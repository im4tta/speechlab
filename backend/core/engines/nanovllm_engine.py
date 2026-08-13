"""nanovllm-voxcpm Khmer engine — ISOLATED-ENV PROXY.

`nano-vllm-voxcpm` needs an NVIDIA GPU, CUDA >= 12, torch >= 2.5, and
flash-attn — a heavier and more version-sensitive stack than the rest of the
main venv — so (like Chatterbox / OmniVoice / VoxCPM / Qwen) it never runs in
this process. This class is a thin proxy that drives
`backend/nanovllm_worker.py` inside its own venv (`backend/venv-nanovllm`).
It keeps the exact same `Engine` surface as every other engine, so
`EngineManager` and `SynthService` need no special-casing.

This is the **recommended engine for Khmer**: same VoxCPM2 checkpoint as the
`voxcpm` engine, but served through nanovllm's batching runtime for much
higher throughput, plus Khmer-aware text normalization and cluster-safe,
sentence-respecting chunking (`core/khmer_text.py`) so long Khmer paragraphs
are split into several requests that nanovllm generates *concurrently*
instead of one long sequential decode. Communication is newline-delimited
JSON over the worker's stdin/stdout; generated audio is written by the
worker to a temp WAV this process reads (mirrors `voxcpm_engine.py`).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import Engine, EngineResult, EngineSynthRequest

log = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/


def _default_worker_python() -> Path:
    venv = _BACKEND_ROOT / "venv-nanovllm"
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _default_worker_script() -> Path:
    return _BACKEND_ROOT / "nanovllm_worker.py"


class NanoVllmKhmerEngine(Engine):
    """Proxy to a nanovllm-voxcpm worker running in backend/venv-nanovllm."""

    name = "nanovllm_km"
    display_name = "VoxCPM2 (nanovllm · Khmer)"
    license = "MIT (runtime) + Apache-2.0 (VoxCPM2 weights)"
    model_url = "https://github.com/GeeeekExplorer/nano-vllm"
    description = (
        "OpenBMB's VoxCPM2 served through nanovllm's concurrent-batching "
        "runtime, tuned for Khmer: Khmer-aware text normalization (numbers, "
        "symbols) and cluster-safe sentence chunking so long passages are "
        "generated as several concurrent requests instead of one long "
        "decode. Voice design, cloning, and transcript-guided cloning. "
        "Requires an NVIDIA GPU. Runs in its own isolated environment."
    )

    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        devices: list[int] | None = None,
        inference_timesteps: int = 10,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 8192,
        max_model_len: int = 4096,
        chunk_max_chars: int = 220,
        worker_python: Path | None = None,
        worker_script: Path | None = None,
    ) -> None:
        self._model_id = model_id
        self._devices = list(devices) if devices else [0]
        self._inference_timesteps = inference_timesteps
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_model_len = max_model_len
        self._chunk_max_chars = chunk_max_chars
        self._worker_python = Path(worker_python) if worker_python else _default_worker_python()
        self._worker_script = Path(worker_script) if worker_script else _default_worker_script()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None
        self._resolved_sample_rate: int | None = None
        self._device_request = "cuda"  # nanovllm requires CUDA; no auto/cpu path

    # -- lifecycle
    def load(self) -> None:
        with self._load_lock:
            if self.is_loaded():
                return
            if not self._worker_python.is_file():
                raise RuntimeError(
                    "nanovllm-voxcpm isn't installed in its isolated environment. "
                    "Run `python studio.py install-nanovllm` (or click Install in the UI)."
                )
            env = dict(os.environ)
            models_dir = _BACKEND_ROOT / "models"
            env["HF_HOME"] = str(models_dir)
            env["HUGGINGFACE_HUB_CACHE"] = str(models_dir / "hub")
            log.info("Spawning nanovllm worker: %s %s", self._worker_python, self._worker_script)
            self._proc = subprocess.Popen(
                [str(self._worker_python), str(self._worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            self._start_stderr_drain()
            resp = self._exchange(
                {
                    "op": "load",
                    "model_id": self._model_id,
                    "devices": self._devices,
                    "inference_timesteps": self._inference_timesteps,
                    "gpu_memory_utilization": self._gpu_memory_utilization,
                    "max_num_seqs": self._max_num_seqs,
                    "max_num_batched_tokens": self._max_num_batched_tokens,
                    "max_model_len": self._max_model_len,
                },
                timeout=900,  # first load downloads + compiles CUDA graphs
            )
            if not resp.get("ok"):
                err = resp.get("error", "unknown error")
                self._kill()
                raise RuntimeError(f"nanovllm worker failed to load: {err}")
            self._resolved_sample_rate = resp.get("sample_rate")

    def unload(self) -> None:
        if self._proc is None:
            return
        # Never block the switch on a worker that's mid-load: the background
        # warmup thread holds `_lock` across a long model download/load. If we
        # can't grab the lock quickly, skip the graceful handshake and kill.
        if self._lock.acquire(timeout=1.0):
            try:
                if self._proc.poll() is None:
                    self._proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._lock.release()
        self._kill()

    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def installed(self) -> bool:
        return self._ready_marker().is_file()

    def _ready_marker(self) -> Path:
        return self._worker_python.parent.parent / ".nanovllm-ready"

    def downloaded(self) -> bool:
        from ..model_cache import model_downloaded

        return model_downloaded(self._model_id)

    def engine_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "device": self._device_request,
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        }

    # -- capabilities
    def sample_rate(self) -> int:
        return self._resolved_sample_rate or 16000

    def max_speakers(self) -> int:
        return 1

    def supports_voice_cloning(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def supports_voice_modes(self) -> bool:
        return True

    def supports_style_clone(self) -> bool:
        return True

    def default_cfg_scale(self) -> float | None:
        return 2.0

    def languages(self) -> list[dict[str, str]]:
        # VoxCPM2 is multilingual; Khmer is the reason this engine exists, so
        # surface it first. "" (Auto) lets the model infer from the script.
        return [
            {"code": "", "label": "Auto-detect"},
            {"code": "km", "label": "Khmer (ខ្មែរ)"},
            {"code": "en", "label": "English"},
            {"code": "zh", "label": "Chinese"},
        ]

    def available_voices(self) -> list:
        return []

    # -- synthesis
    def _build_synth_msg(self, req: EngineSynthRequest, out_wav: str) -> dict:
        text = (req.text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")
        mode = req.voice_mode or ("clone" if req.reference_audio else "auto")
        style = (req.instruct or "").strip()
        if mode == "design" and not style:
            mode = "auto"
        msg: dict[str, Any] = {
            "op": "synth",
            "mode": mode,
            "text": text,
            "out_wav": out_wav,
            "chunk_max_chars": self._chunk_max_chars,
            "khmer_normalize": True,
        }
        if mode == "clone":
            if not req.reference_audio:
                raise ValueError("nanovllm Khmer engine clone mode requires a reference voice.")
            msg["ref_audio"] = req.reference_audio
            transcript = (req.reference_text or "").strip()
            if transcript:
                msg["prompt_text"] = transcript
        if style and mode in ("design", "clone"):
            msg["instruct"] = style
        if req.cfg_scale is not None:
            msg["cfg_value"] = float(req.cfg_scale)
        steps = req.inference_steps if req.inference_steps is not None else self._inference_timesteps
        if steps is not None:
            msg["inference_timesteps"] = int(steps)
        if req.temperature is not None:
            msg["temperature"] = float(req.temperature)
        if req.seed is not None:
            msg["seed"] = int(req.seed)
        return msg

    def synthesize(self, req: EngineSynthRequest) -> EngineResult:
        if not self.is_loaded():
            raise RuntimeError("nanovllm worker is not loaded")
        fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="nanovllm-km-")
        os.close(fd)
        try:
            msg = self._build_synth_msg(req, out_wav)
            resp = self._exchange(msg, timeout=300)
            if not resp.get("ok"):
                raise RuntimeError(f"nanovllm synth failed: {resp.get('error', 'unknown error')}")
            wav_bytes = Path(out_wav).read_bytes()
        finally:
            try:
                os.unlink(out_wav)
            except OSError:
                pass
        return EngineResult(
            wav_bytes=wav_bytes,
            sample_rate=int(resp.get("sample_rate", self.sample_rate())),
            duration_sec=float(resp.get("duration_sec", 0.0)),
            inference_ms=int(resp.get("inference_ms", 0)),
        )

    # -- internals (mirrors voxcpm_engine.py's stdio protocol driver)
    def _exchange(self, msg: dict, expect_reply: bool = True, timeout: float | None = None) -> dict:
        with self._lock:
            if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("nanovllm worker is not running")
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._kill()
                raise RuntimeError(f"nanovllm worker pipe broke: {exc}") from exc
            if not expect_reply:
                return {"ok": True}
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    if self._stderr_thread is not None:
                        self._stderr_thread.join(timeout=1.0)
                    stderr = self._recent_stderr()
                    self._kill()
                    raise RuntimeError(
                        "nanovllm worker closed unexpectedly" + (f": {stderr}" if stderr else "")
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    log.debug("nanovllm worker non-protocol stdout: %s", stripped[:200])
                    continue
                if isinstance(parsed, dict) and "progress" in parsed:
                    # Live chunk progress — forward to the shared tracker and
                    # keep reading for the real reply.
                    self._report_progress(parsed.get("progress"))
                    continue
                return parsed

    def _report_progress(self, data: Any) -> None:
        """Forward a worker progress message to the global synthesis tracker."""
        if not isinstance(data, dict):
            return
        try:
            from ...services.synth_progress import synth_progress

            synth_progress.update(
                done=data.get("done"),
                total=data.get("total"),
                stage=data.get("stage"),
            )
        except Exception:  # noqa: BLE001 — progress is best-effort
            pass

    def _start_stderr_drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        self._stderr_tail.clear()

        def _drain(stream, sink) -> None:
            try:
                for line in stream:
                    sink.append(line.rstrip("\n"))
            except Exception:  # noqa: BLE001
                pass

        thread = threading.Thread(target=_drain, args=(proc.stderr, self._stderr_tail), daemon=True)
        thread.start()
        self._stderr_thread = thread

    def _recent_stderr(self) -> str:
        return "\n".join(self._stderr_tail).strip()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass
