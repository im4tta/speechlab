#!/usr/bin/env python3
"""VoxCPM worker — runs INSIDE backend/venv-voxcpm.

Speaks newline-delimited JSON on stdin/stdout. The parent process
(backend/core/engines/voxcpm_engine.py) drives it. All human-readable
logging goes to STDERR so it never corrupts the stdout protocol.

Protocol (one JSON object per line):
  stdin  {"op":"load","device":"auto|cuda|mps|cpu","model_id":"openbmb/VoxCPM2"}
         {"op":"synth","mode":"clone|design|auto","text":..,"out_wav":<path>,
          "ref_audio":<path?>,"prompt_text":<str?>,"instruct":<str?>,
          "cfg_value":<float?>,"inference_timesteps":<int?>,"seed":<int?>,
          "chunk_max_chars":<int?>,"khmer_normalize":<bool?>,
          "anchor_voice":<bool?>}
         {"op":"shutdown"}
  stdout {"ok":true,"device":..,"device_mode":..,"dtype":..,"vram_gb":..}  (load)
         {"ok":true,"sample_rate":48000,"duration_sec":..,"inference_ms":..,
          "n_chunks":..}                                                  (synth)
         {"progress":{..}}                                                (any time)
         {"ok":false,"error":".."}                                        (any failure)

VoxCPM expresses voice DESIGN and STYLE STEERING inline as a "(...)" prefix in
the text (NOT a separate argument), so this worker composes the prefixed text.
Long text (and all Khmer text) goes through the shared chunked pipeline in
core/voxcpm_pipeline.py — Khmer normalization, sentence chunking, voice
anchoring across chunks and seamless joining — run sequentially here. The
generated audio is written to out_wav (16-bit PCM mono WAV at 48 kHz); only
metadata travels over the pipe.

Device selection (auto): a CUDA GPU with enough VRAM, else Apple Silicon MPS,
else CPU. voxcpm >= 2.0 takes the device explicitly and forces float32 on MPS.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

# Let unsupported MPS ops fall back to CPU instead of crashing (Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Protocol output. main() replaces this with the REAL stdout and points fd 1
# (Python AND C-level) at stderr, so model-load/tqdm noise can't corrupt the
# newline-delimited JSON the parent reads.
_OUT = sys.stdout

_DEFAULT_SAMPLE_RATE = 48000

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.voxcpm_pipeline import (  # noqa: E402
    ReferenceBackend,
    build_plan,
    default_progress_writer,
    load_reference_model,
    synthesize_plan,
)
from core.voxcpm_runtime import REFERENCE_MIN_VRAM_GB, probe_hardware, reference_device  # noqa: E402


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(obj: dict) -> None:
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


def _write_wav_int16(path: str, samples, sample_rate: int) -> None:
    """Write a mono 16-bit PCM WAV from a float or int16 numpy array."""
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())


#: Kept for backwards compatibility with callers/tests; the real threshold
#: lives in core/voxcpm_runtime.py.
_GPU_VRAM_GB_THRESHOLD = REFERENCE_MIN_VRAM_GB


def _resolve_device_config(requested: str | None = "auto") -> tuple[str, str, float | None]:
    """Pick (device, device_mode, vram_gb).

    device_mode is "cuda" | "mps" | "cpu" | "cpu_offload" — the last meaning a
    CUDA GPU exists but is too small for VoxCPM2, so the model runs on CPU."""
    hw = probe_hardware()
    device = reference_device(hw, requested)
    mode = device
    if device == "cpu" and hw.cuda:
        mode = "cpu_offload"
    return device, mode, hw.vram_gb


def _build_generate_kwargs(req: dict) -> tuple[dict, str]:
    """Translate a single-chunk synth request into voxcpm.generate(**kwargs).

    Dispatch table (mode, has_ref, has_style, has_transcript):
      auto              -> generate(text)
      design            -> generate("(style)text")
      clone             -> generate(text, reference_wav_path=ref)
      controllable      -> generate("(style)text", reference_wav_path=ref)
      ultimate          -> generate(text, prompt_wav_path=ref, prompt_text=tr,
                                    reference_wav_path=ref)
    An empty design style downgrades to auto. (Multi-chunk requests go through
    core.voxcpm_pipeline, which applies the same table per chunk.)
    """
    text = (req.get("text") or "").strip()
    mode = req.get("mode") or "auto"
    style = (req.get("instruct") or "").strip()
    ref = req.get("ref_audio")
    transcript = (req.get("prompt_text") or "").strip()

    if mode == "design" and not style:
        mode = "auto"

    # Inline "(style)" prefix for design + controllable-clone only.
    prefixed = f"({style}){text}" if style and mode in ("design", "clone") else text
    kwargs: dict = {"text": prefixed}

    if req.get("cfg_value") is not None:
        kwargs["cfg_value"] = float(req["cfg_value"])
    if req.get("inference_timesteps") is not None:
        kwargs["inference_timesteps"] = int(req["inference_timesteps"])

    if mode == "clone":
        if not ref:
            raise ValueError("clone mode requires ref_audio")
        kwargs["reference_wav_path"] = ref
        if transcript:
            # Ultimate cloning: continuation guided by the reference transcript.
            kwargs["prompt_wav_path"] = ref
            kwargs["prompt_text"] = transcript
    return kwargs, mode


class _Worker:
    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "load":
            return self._load(req)
        if op == "synth":
            return self._synth(req)
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op: {op!r}"}

    def _load(self, req: dict) -> dict:
        model_id = req.get("model_id") or "openbmb/VoxCPM2"
        try:
            import voxcpm  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"import voxcpm failed: {exc}"}
        device, device_mode, vram_gb = _resolve_device_config(req.get("device"))
        _reply({"progress": {"stage": f"Loading model from {model_id} on {device}…"}})
        try:
            self._model = load_reference_model(model_id, device, _log)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"load failed: {exc}"}
        _reply({"progress": {"stage": "Model ready", "done": 1, "total": 1}})
        self._device = device
        try:
            sr = int(self._model.tts_model.sample_rate)
            if sr > 0:
                self._sample_rate = sr
        except Exception:  # noqa: BLE001
            pass
        dtype = "float32" if device == "mps" else None
        try:
            dtype = str(self._model.tts_model.config.dtype)
        except Exception:  # noqa: BLE001
            pass
        _log(
            f"[voxcpm-worker] model loaded (requested device={req.get('device')!r} → "
            f"{device}, mode={device_mode}, dtype={dtype}), sr={self._sample_rate}"
        )
        return {
            "ok": True,
            "device": device,
            "device_mode": device_mode,
            "dtype": dtype or "bfloat16",
            "vram_gb": vram_gb,
        }

    def _synth(self, req: dict) -> dict:
        if self._model is None:
            return {"ok": False, "error": "model not loaded"}
        out_wav = req.get("out_wav")
        if not (req.get("text") or "").strip():
            return {"ok": False, "error": "text must be non-empty"}
        if not out_wav:
            return {"ok": False, "error": "out_wav required"}
        try:
            plan = build_plan(req)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        backend = ReferenceBackend(self._model, device=self._device,
                                   inference_timesteps=int(req.get("inference_timesteps") or 10))
        self._sample_rate = backend.sample_rate or self._sample_rate
        gen = {k: req[k] for k in ("cfg_value", "inference_timesteps", "seed") if req.get(k) is not None}
        t0 = time.perf_counter()
        _reply({"progress": {"stage": "Synthesizing…"}})
        try:
            full, stats = asyncio.run(
                synthesize_plan(
                    backend,
                    plan,
                    gen=gen,
                    ref_audio=req.get("ref_audio"),
                    prompt_text=req.get("prompt_text"),
                    anchor_voice=bool(req.get("anchor_voice", True)),
                    retries=int(req.get("retries", 1)),
                    progress=default_progress_writer(_reply) if len(plan.bodies) > 1 else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"generate failed: {exc}"}
        inference_ms = int((time.perf_counter() - t0) * 1000)
        _reply({"progress": {"stage": "Writing audio…"}})
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
        }


def main() -> int:
    global _OUT
    _OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    try:
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr

    worker = _Worker()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"bad json: {exc}"})
            continue
        try:
            resp = worker.handle(req)
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": f"worker exception: {exc}"}
        _reply(resp)
        if req.get("op") == "shutdown":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
