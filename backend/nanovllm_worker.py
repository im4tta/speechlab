#!/usr/bin/env python3
"""nanovllm-voxcpm worker — runs INSIDE backend/venv-nanovllm.

Speaks newline-delimited JSON on stdin/stdout, same protocol shape as the
other isolated-engine workers (voxcpm_worker.py, qwen_worker.py, ...). All
human-readable logging goes to STDERR so it never corrupts the stdout
protocol.

Protocol (one JSON object per line):
  stdin  {"op":"load","model_id":..,"devices":[0],"inference_timesteps":10,
          "gpu_memory_utilization":0.9,"max_num_seqs":16,
          "max_num_batched_tokens":8192,"max_model_len":4096}
         {"op":"synth","mode":"clone|design|auto","text":..,"out_wav":<path>,
          "ref_audio":<path?>,"prompt_text":<str?>,"instruct":<str?>,
          "cfg_value":<float?>,"temperature":<float?>,"seed":<int?>,
          "max_generate_length":<int?>,"chunk_max_chars":<int?>,
          "khmer_normalize":<bool?>}
         {"op":"shutdown"}
  stdout {"ok":true,"device":"cuda","sample_rate":16000}                 (load)
         {"ok":true,"sample_rate":..,"duration_sec":..,"inference_ms":..,
          "n_chunks":..}                                                 (synth)
         {"ok":false,"error":".."}                                      (any failure)

Unlike the plain-pytorch VoxCPM worker, this one is async-native end to end:
`nanovllm_voxcpm`'s server pool exposes a genuinely async, concurrency-batched
`generate()`. For a long piece of Khmer text we split it into cluster-safe,
sentence-respecting chunks (core/khmer_text.py) and fire every chunk at the
pool with `asyncio.gather`, letting nanovllm batch them on the GPU instead of
generating one long sequence end to end. This is the whole reason to put
nanovllm underneath VoxCPM2 rather than the reference PyTorch implementation
that `voxcpm_worker.py` drives.

VoxCPM2 expresses voice DESIGN and STYLE STEERING inline as a "(...)" prefix
on the target text (same convention as the reference implementation), so a
style prefix is applied to every chunk individually — each chunk is an
independent request to the pool with no shared decode state.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

# Protocol output. main() replaces this with the REAL stdout and points fd 1
# (Python AND C-level) at stderr, so model-load/tqdm noise can't corrupt the
# newline-delimited JSON the parent reads.
_OUT = sys.stdout

_DEFAULT_SAMPLE_RATE = 16000
_SILENCE_GAP_S = 0.12  # inserted between independently-generated chunks

# core/khmer_text.py lives one directory up (backend/core/); make it
# importable without requiring the isolated venv to install the main package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.khmer_text import chunk_khmer_text, is_khmer_text, normalize_khmer_text  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(obj: dict) -> None:
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


def _write_wav_int16(path: str, arr, sample_rate: int) -> None:
    import numpy as np

    if arr.dtype != np.int16:
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())


def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _build_targets(req: dict) -> tuple[list[str], str]:
    """Split the request into per-chunk target texts. Mirrors the reference
    VoxCPM worker's mode dispatch (auto / design / clone / controllable
    clone / ultimate clone) but fans each mode out over N text chunks.

    Returns (chunk_texts, mode).
    """
    raw_text = (req.get("text") or "").strip()
    if not raw_text:
        raise ValueError("text must be non-empty")
    mode = req.get("mode") or "auto"
    style = (req.get("instruct") or "").strip()
    ref = req.get("ref_audio")

    if mode == "design" and not style:
        mode = "auto"

    khmer_normalize = req.get("khmer_normalize", True)
    if khmer_normalize and is_khmer_text(raw_text):
        raw_text = normalize_khmer_text(raw_text)

    max_chars = int(req.get("chunk_max_chars") or 220)
    pieces = [c.text for c in chunk_khmer_text(raw_text, max_chars=max_chars)]
    if not pieces:
        pieces = [raw_text]

    if style and mode in ("design", "clone"):
        pieces = [f"({style}){p}" for p in pieces]

    if mode == "clone" and not ref:
        raise ValueError("clone mode requires ref_audio")

    return pieces, mode


class _Worker:
    def __init__(self) -> None:
        self._pool = None  # AsyncVoxCPM2ServerPool | AsyncVoxCPMServerPool
        self._sample_rate = _DEFAULT_SAMPLE_RATE
        self._inference_timesteps = 10

    async def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "load":
            return await self._load(req)
        if op == "synth":
            return await self._synth(req)
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op: {op!r}"}

    # -- load -----------------------------------------------------------
    async def _load(self, req: dict) -> dict:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"import torch failed: {exc}"}
        if not torch.cuda.is_available():
            return {
                "ok": False,
                "error": (
                    "nanovllm-voxcpm requires an NVIDIA GPU (CUDA). This engine "
                    "cannot run on CPU/MPS — use VoxCPM2 (the 'voxcpm' engine) "
                    "instead on non-CUDA hardware."
                ),
            }
        try:
            from nanovllm_voxcpm import VoxCPM
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"import nanovllm_voxcpm failed: {exc}"}

        model_id = req.get("model_id") or "openbmb/VoxCPM2"
        self._inference_timesteps = int(req.get("inference_timesteps") or 10)
        try:
            self._pool = VoxCPM.from_pretrained(
                model=model_id,
                inference_timesteps=self._inference_timesteps,
                max_num_batched_tokens=int(req.get("max_num_batched_tokens") or 8192),
                max_num_seqs=int(req.get("max_num_seqs") or 16),
                max_model_len=int(req.get("max_model_len") or 4096),
                gpu_memory_utilization=float(req.get("gpu_memory_utilization") or 0.9),
                enforce_eager=bool(req.get("enforce_eager") or False),
                devices=list(req.get("devices") or [0]),
            )
            await self._pool.wait_for_ready()
        except Exception as exc:  # noqa: BLE001
            self._pool = None
            return {"ok": False, "error": f"load failed: {exc}"}

        try:
            info = await self._pool.get_model_info()
            sr = int(info.get("output_sample_rate") or info.get("sample_rate") or 0)
            if sr > 0:
                self._sample_rate = sr
        except Exception:  # noqa: BLE001
            pass
        _log(f"[nanovllm-worker] model loaded, sr={self._sample_rate}, devices={req.get('devices') or [0]}")
        return {"ok": True, "device": "cuda", "sample_rate": self._sample_rate}

    # -- synth ------------------------------------------------------------
    async def _synth(self, req: dict) -> dict:
        if self._pool is None:
            return {"ok": False, "error": "model not loaded"}
        out_wav = req.get("out_wav")
        if not out_wav:
            return {"ok": False, "error": "out_wav required"}

        try:
            targets, mode = _build_targets(req)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        gen_kwargs: dict[str, Any] = {}
        if req.get("cfg_value") is not None:
            gen_kwargs["cfg_value"] = float(req["cfg_value"])
        if req.get("temperature") is not None:
            gen_kwargs["temperature"] = float(req["temperature"])
        if req.get("max_generate_length") is not None:
            gen_kwargs["max_generate_length"] = int(req["max_generate_length"])
        if req.get("seed") is not None:
            gen_kwargs["seed"] = int(req["seed"])

        prompt_id: str | None = None
        ref_audio_latents: bytes | None = None
        try:
            ref = req.get("ref_audio")
            transcript = (req.get("prompt_text") or "").strip()
            if mode == "clone" and ref:
                wav_bytes = _read_bytes(ref)
                wav_format = Path(ref).suffix.lstrip(".") or "wav"
                if transcript:
                    # Ultimate cloning: transcript-guided continuation. Register
                    # once, reuse the returned prompt_id for every chunk so we
                    # don't re-encode the reference audio per chunk.
                    prompt_id = await self._pool.add_prompt(wav_bytes, wav_format, transcript)
                else:
                    # Reference-audio-only conditioning (independent of any
                    # decode prefix) — no transcript needed or assumed.
                    ref_audio_latents = await self._pool.encode_latents(wav_bytes, wav_format)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"reference audio processing failed: {exc}"}

        t0 = time.perf_counter()
        try:
            # Fire every chunk at the pool concurrently and report progress on
            # the protocol as each one completes, so the backend can surface a
            # live "Chunk N/M" percentage to the UI.
            tasks = {
                asyncio.create_task(
                    self._gen_one(text, prompt_id=prompt_id, ref_audio_latents=ref_audio_latents, **gen_kwargs)
                ): i
                for i, text in enumerate(targets)
            }
            waves: list = [None] * len(tasks)
            done = 0
            for task in asyncio.as_completed(tasks):
                idx = tasks[task]
                waves[idx] = await task
                done += 1
                _reply(
                    {
                        "progress": {
                            "done": done,
                            "total": len(tasks),
                            "stage": f"Chunk {done}/{len(tasks)}",
                        }
                    }
                )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"generate failed: {exc}"}
        finally:
            if prompt_id is not None:
                try:
                    await self._pool.remove_prompt(prompt_id)
                except Exception:  # noqa: BLE001
                    pass
        inference_ms = int((time.perf_counter() - t0) * 1000)

        import numpy as np

        gap = np.zeros(int(_SILENCE_GAP_S * self._sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, w in enumerate(waves):
            if i > 0:
                pieces.append(gap)
            pieces.append(w)
        full = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)

        try:
            _write_wav_int16(out_wav, full, self._sample_rate)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"write wav failed: {exc}"}

        return {
            "ok": True,
            "sample_rate": self._sample_rate,
            "duration_sec": float(full.size) / float(self._sample_rate),
            "inference_ms": inference_ms,
            "n_chunks": len(targets),
        }

    async def _gen_one(
        self,
        target_text: str,
        *,
        prompt_id: str | None,
        ref_audio_latents: bytes | None,
        **gen_kwargs: Any,
    ):
        import numpy as np

        chunks: list[np.ndarray] = []
        async for wav_chunk in self._pool.generate(
            target_text=target_text,
            prompt_id=prompt_id,
            ref_audio_latents=ref_audio_latents,
            **gen_kwargs,
        ):
            chunks.append(np.asarray(wav_chunk, dtype=np.float32).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0)


async def _main_async() -> int:
    loop = asyncio.get_running_loop()
    worker = _Worker()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"bad json: {exc}"})
            continue
        try:
            resp = await worker.handle(req)
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": f"worker exception: {exc}"}
        _reply(resp)
        if req.get("op") == "shutdown":
            if worker._pool is not None:
                try:
                    await worker._pool.stop()
                except Exception:  # noqa: BLE001
                    pass
            break
    return 0


def main() -> int:
    global _OUT
    _OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    try:
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr

    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
