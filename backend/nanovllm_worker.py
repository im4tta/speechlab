#!/usr/bin/env python3
"""nanovllm-voxcpm (Khmer) worker — runs INSIDE backend/venv-nanovllm.

Speaks newline-delimited JSON on stdin/stdout, same protocol shape as the
other isolated-engine workers (voxcpm_worker.py, qwen_worker.py, ...). All
human-readable logging goes to STDERR so it never corrupts the stdout
protocol.

Protocol (one JSON object per line):
  stdin  {"op":"load","model_id":..,"device":"auto|cuda|mps|cpu",
          "backend":"auto|nanovllm|reference","devices":[0],
          "inference_timesteps":10,
          # optional explicit nanovllm sizing (None/absent = auto by VRAM):
          "gpu_memory_utilization":..,"max_num_seqs":..,
          "max_num_batched_tokens":..,"max_model_len":..,"enforce_eager":..}
         {"op":"synth","mode":"clone|design|auto","text":..,"out_wav":<path>,
          "ref_audio":<path?>,"prompt_text":<str?>,"instruct":<str?>,
          "cfg_value":<float?>,"temperature":<float?>,"seed":<int?>,
          "inference_timesteps":<int?>,"max_generate_length":<int?>,
          "chunk_max_chars":<int?>,"khmer_normalize":<bool?>,
          "anchor_voice":<bool?>,"retries":<int?>}
         {"op":"shutdown"}
  stdout {"ok":true,"device":..,"backend":"nanovllm|reference",
          "fallback_reason":..,"profile":..,"sample_rate":..,
          "chunk_max_chars":..,"hardware":{..}}                          (load)
         {"ok":true,"sample_rate":..,"duration_sec":..,"inference_ms":..,
          "n_chunks":..,"retries":..,"anchored":..}                      (synth)
         {"progress":{"done":..,"total":..,"stage":..}}                  (any time)
         {"ok":false,"error":".."}                                      (any failure)

Two runtimes behind one protocol:

* **nanovllm** (NVIDIA Ampere+, ≥ 6 GB): `nanovllm_voxcpm`'s server pool is
  genuinely async and concurrency-batched, so a long Khmer passage is split
  into chunks that are generated *at the same time*. Pool sizing (KV cache,
  batch width, CUDA graphs) is picked from the GPU's VRAM and stepped down if
  a load runs out of memory — that's what lets 6–12 GB cards run it.
* **reference** (Apple Silicon/MPS, CPU, older or smaller NVIDIA GPUs, or no
  flash-attn): the upstream `voxcpm` PyTorch runtime, one chunk at a time.
  Same checkpoint, same Khmer pipeline, just sequential.

The Khmer pipeline itself (normalization, chunking, voice anchoring, bad-case
retries, seamless joining) is shared with the voxcpm worker — see
core/voxcpm_pipeline.py.
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

# Let unsupported MPS ops fall back to CPU instead of crashing (Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Protocol output. main() replaces this with the REAL stdout and points fd 1
# (Python AND C-level) at stderr, so model-load/tqdm noise can't corrupt the
# newline-delimited JSON the parent reads.
_OUT = sys.stdout

_DEFAULT_SAMPLE_RATE = 48000

# core/ lives next to this file (backend/core/); make it importable without
# requiring the isolated venv to install the main package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.voxcpm_pipeline import (  # noqa: E402
    ReferenceBackend,
    build_plan,
    default_progress_writer,
    load_reference_model,
    synthesize_plan,
)
from core.voxcpm_runtime import (  # noqa: E402
    HardwareInfo,
    apply_overrides,
    choose_backend,
    max_generate_length,
    nano_profile_ladder,
    probe_hardware,
    reference_device,
)


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


def _build_targets(req: dict) -> tuple[list[str], str]:
    """Per-chunk target texts (style prefix applied) + resolved mode.

    Thin wrapper over core.voxcpm_pipeline.build_plan, kept for callers/tests
    that only need the texts."""
    plan = build_plan(req)
    return plan.targets, plan.mode


def _nanovllm_importable() -> tuple[bool, str | None]:
    try:
        import nanovllm_voxcpm  # noqa: F401 — imports flash_attn too
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


# ─── nanovllm backend ────────────────────────────────────────────────────────
class _NanoBackend:
    """Concurrent backend over `nanovllm_voxcpm`'s async server pool."""

    name = "nanovllm"
    concurrent = True

    def __init__(self, pool, *, sample_rate: int, feat_dim: int, patch_size: int,
                 max_model_len: int, tokenizer=None) -> None:
        self._pool = pool
        self.sample_rate = sample_rate
        self._feat_dim = max(1, feat_dim)
        self._patch_size = max(1, patch_size)
        self._max_model_len = max_model_len
        self._tok = tokenizer

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            try:
                return len(self._tok.encode(text, add_special_tokens=False))
            except Exception:  # noqa: BLE001
                pass
        # Byte count is an upper bound for a byte-fallback BPE tokenizer.
        return len(text.encode("utf-8"))

    def _patches(self, latents: bytes | None) -> int:
        if not latents:
            return 0
        frames = len(latents) // 4 // self._feat_dim
        return frames // self._patch_size

    async def condition(self, *, ref_path: str | None = None, wav_bytes: bytes | None = None,
                        fmt: str = "wav", transcript: str | None = None) -> dict:
        if ref_path is not None:
            wav_bytes = Path(ref_path).read_bytes()
            fmt = Path(ref_path).suffix.lstrip(".") or "wav"
        latents = await self._pool.encode_latents(wav_bytes, fmt)
        if transcript:
            # Transcript-guided continuation ("ultimate cloning").
            return {"prompt_latents": latents, "prompt_text": transcript}
        # Reference-audio-only conditioning — no transcript needed or assumed.
        return {"ref_audio_latents": latents}

    async def release(self, cond: dict) -> None:
        return None

    def plan_cap(self, text: str, cond: dict | None, requested: int | None) -> int:
        cond = cond or {}
        target_tokens = self._count_tokens(text)
        prompt_len = target_tokens + self._count_tokens(cond.get("prompt_text", "")) + 1
        if cond.get("ref_audio_latents"):
            prompt_len += self._patches(cond["ref_audio_latents"]) + 2
        if cond.get("prompt_latents"):
            prompt_len += self._patches(cond["prompt_latents"])
        cap = max_generate_length(target_tokens)
        if requested:
            cap = min(cap, int(requested))
        cap = min(cap, self._max_model_len - prompt_len - 4)
        if cap < 16:
            raise ValueError(
                f"chunk too long for max_model_len={self._max_model_len} "
                f"(prompt ≈ {prompt_len} tokens) — lower chunk_max_chars or use a shorter reference clip"
            )
        return cap

    async def generate(self, text: str, cond: dict | None, gen: dict):
        import numpy as np

        cap = self.plan_cap(text, cond, gen.get("max_generate_length"))
        kwargs: dict[str, Any] = {"target_text": text, "max_generate_length": cap}
        for key in ("prompt_latents", "prompt_text", "ref_audio_latents"):
            if cond and cond.get(key):
                kwargs[key] = cond[key]
        for key, cast in (("cfg_value", float), ("temperature", float), ("seed", int)):
            if gen.get(key) is not None:
                kwargs[key] = cast(gen[key])
        pieces: list[np.ndarray] = []
        async for wav_chunk in self._pool.generate(**kwargs):
            pieces.append(np.asarray(wav_chunk, dtype=np.float32).reshape(-1))
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return wav, len(pieces) >= cap

    async def stop(self) -> None:
        await self._pool.stop()


def _load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001
        _log(f"[nanovllm-worker] tokenizer unavailable ({exc}); using byte-count estimates")
        return None


class _Worker:
    def __init__(self) -> None:
        self._backend = None  # _NanoBackend | ReferenceBackend
        self._sample_rate = _DEFAULT_SAMPLE_RATE
        self._inference_timesteps = 10
        self._chunk_cap: int | None = None

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
    async def _load_nanovllm(self, req: dict, hw: HardwareInfo) -> tuple[bool, str | None, str | None]:
        """Try nanovllm with the best-fitting profile, stepping down on failure.
        Returns (ok, profile_name, last_error)."""
        from nanovllm_voxcpm import VoxCPM

        model_id = req.get("model_id") or "openbmb/VoxCPM2"
        overrides = {k: req.get(k) for k in ("max_num_seqs", "max_num_batched_tokens", "max_model_len",
                                              "gpu_memory_utilization", "enforce_eager")}
        last_err: str | None = None
        for base in nano_profile_ladder(hw.vram_gb):
            profile = apply_overrides(base, overrides)
            _reply({"progress": {"stage": f"Loading VoxCPM2 on nanovllm ({profile.name} profile)…"}})
            pool = None
            try:
                pool = VoxCPM.from_pretrained(
                    model=model_id,
                    inference_timesteps=self._inference_timesteps,
                    devices=list(req.get("devices") or [0]),
                    **profile.to_load_kwargs(),
                )
                await pool.wait_for_ready()
                info = await pool.get_model_info()
            except Exception as exc:  # noqa: BLE001
                last_err = f"{profile.name}: {exc}"
                _log(f"[nanovllm-worker] load failed with {profile.name} profile: {exc}")
                if pool is not None:
                    try:
                        await pool.stop()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                continue
            sr = int(info.get("output_sample_rate") or info.get("sample_rate") or _DEFAULT_SAMPLE_RATE)
            self._backend = _NanoBackend(
                pool,
                sample_rate=sr,
                feat_dim=int(info.get("feat_dim") or 64),
                patch_size=int(info.get("patch_size") or 1),
                max_model_len=profile.max_model_len,
                tokenizer=_load_tokenizer(info.get("model_path")),
            )
            self._sample_rate = sr
            self._chunk_cap = profile.chunk_max_chars
            return True, profile.name, None
        return False, None, last_err

    async def _load_reference(self, req: dict, hw: HardwareInfo) -> str:
        model_id = req.get("model_id") or "openbmb/VoxCPM2"
        device = reference_device(hw, req.get("device"))
        _reply({"progress": {"stage": f"Loading VoxCPM2 (reference runtime, {device})…"}})
        # Load on the event-loop thread: generation runs there too (see
        # ReferenceBackend — CUDA graphs from torch.compile are per-thread).
        model = load_reference_model(model_id, device, _log)
        self._backend = ReferenceBackend(model, device=device, inference_timesteps=self._inference_timesteps)
        self._sample_rate = self._backend.sample_rate
        self._chunk_cap = None
        return device

    async def _load(self, req: dict) -> dict:
        self._inference_timesteps = int(req.get("inference_timesteps") or 10)
        hw = await asyncio.to_thread(probe_hardware)
        requested = (req.get("backend") or "auto").lower()
        device_req = (req.get("device") or "auto").lower()
        if device_req in ("cpu", "mps"):
            requested = "reference"  # nanovllm is CUDA-only
        importable, import_err = (True, None)
        if requested != "reference":
            importable, import_err = await asyncio.to_thread(_nanovllm_importable)
        backend, reason = choose_backend(hw, requested, nanovllm_importable=importable)
        if requested == "nanovllm" and backend != "nanovllm":
            return {"ok": False, "error": f"nanovllm backend requested but unavailable: {reason or import_err}"}

        profile_name = None
        if backend == "nanovllm":
            ok, profile_name, err = await self._load_nanovllm(req, hw)
            if not ok:
                if requested == "nanovllm":
                    return {"ok": False, "error": f"nanovllm failed to load: {err}"}
                backend, reason = "reference", f"nanovllm failed to load ({err})"
        if backend == "reference":
            if reason:
                _log(f"[nanovllm-worker] using the reference VoxCPM2 runtime: {reason}")
            try:
                device = await self._load_reference(req, hw)
            except ImportError as exc:
                return {
                    "ok": False,
                    "error": (
                        f"Can't run nanovllm here ({reason}) and the reference VoxCPM2 "
                        f"runtime isn't installed ({exc}). Re-run `python studio.py "
                        "install-nanovllm` to install the right runtime for this machine."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"load failed: {exc}"}
        else:
            device = "cuda"

        _log(
            f"[nanovllm-worker] ready: backend={backend} device={device} "
            f"profile={profile_name} sr={self._sample_rate} hw={hw.to_dict()}"
        )
        return {
            "ok": True,
            "device": device,
            "backend": backend,
            "fallback_reason": reason,
            "profile": profile_name,
            "sample_rate": self._sample_rate,
            "chunk_max_chars": self._chunk_cap,
            "hardware": hw.to_dict(),
        }

    # -- synth ------------------------------------------------------------
    async def _synth(self, req: dict) -> dict:
        if self._backend is None:
            return {"ok": False, "error": "model not loaded"}
        out_wav = req.get("out_wav")
        if not out_wav:
            return {"ok": False, "error": "out_wav required"}
        try:
            plan = build_plan(req, max_chars_cap=self._chunk_cap)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        gen: dict[str, Any] = {}
        for key in ("cfg_value", "temperature", "seed", "max_generate_length", "inference_timesteps"):
            if req.get(key) is not None:
                gen[key] = req[key]

        t0 = time.perf_counter()
        try:
            full, stats = await synthesize_plan(
                self._backend,
                plan,
                gen=gen,
                ref_audio=req.get("ref_audio"),
                prompt_text=req.get("prompt_text"),
                anchor_voice=bool(req.get("anchor_voice", True)),
                retries=int(req.get("retries", 2)),
                progress=default_progress_writer(_reply),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"generate failed: {exc}"}
        inference_ms = int((time.perf_counter() - t0) * 1000)

        try:
            _write_wav_int16(out_wav, full, self._sample_rate)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"write wav failed: {exc}"}

        return {
            "ok": True,
            "sample_rate": self._sample_rate,
            "duration_sec": float(full.size) / float(self._sample_rate),
            "inference_ms": inference_ms,
            "n_chunks": stats.n_chunks,
            "retries": stats.retries,
            "anchored": stats.anchored,
        }

    async def stop(self) -> None:
        stop = getattr(self._backend, "stop", None)
        if stop is not None:
            try:
                await stop()
            except Exception:  # noqa: BLE001
                pass


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
            await worker.stop()
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
