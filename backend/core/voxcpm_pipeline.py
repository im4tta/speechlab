"""Chunked VoxCPM2 synthesis pipeline shared by the nanovllm + voxcpm workers.

Both isolated workers (`backend/nanovllm_worker.py`, `backend/voxcpm_worker.py`)
drive the same VoxCPM2 checkpoint through different runtimes. Everything that
is *not* runtime-specific lives here, so the Khmer quality work applies to
whichever runtime the machine can actually run:

1. **Plan** (`build_plan`): split an explicit leading "(style)" off the text,
   normalize Khmer (numbers, dates, ៗ …; see core/khmer_text.py), neutralize
   stray parentheses (VoxCPM would read them as style prompts), and cut the
   text into cluster-safe, sentence-respecting chunks tagged with the kind of
   break that ends each one.
2. **Voice anchoring**: in auto/design modes there's no reference voice, so
   independently generated chunks would each invent a *different* speaker.
   The first chunk is generated alone, and its audio then conditions every
   remaining chunk (reference-audio cloning), so a long passage keeps one voice.
3. **Bad-case guard**: each chunk's generation length is capped from its token
   count, and a chunk that runs away (hits the cap) or comes back implausibly
   short is regenerated (up to `retries` times, re-seeded).
4. **Join** (core/audio_join.py): trim, loudness-match, boundary-aware pauses,
   fades.

A backend implements three coroutines — `condition`, `release`, `generate` —
plus `name`, `sample_rate` and `concurrent`. `ReferenceBackend` (plain
`voxcpm` PyTorch; CUDA / Apple MPS / CPU) lives here; the nanovllm backend
lives in the nanovllm worker since it needs that runtime's server pool.

Imported inside the worker venvs via a sys.path insert of `backend/`, so it
depends only on the stdlib + numpy (torch is imported lazily).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from .khmer_text import (
    BOUNDARY_PARAGRAPH,
    chunk_khmer_text,
    clean_khmer_orthography,
    is_khmer_text,
    neutralize_parentheses,
    normalize_khmer_text,
    split_style_prefix,
)

#: Default characters per chunk (overridable per request / capped per runtime).
DEFAULT_CHUNK_MAX_CHARS = 220
#: Anchor clip length used to condition the remaining chunks (seconds).
ANCHOR_MAX_SECONDS = 10.0
#: Faster than any human reads — a chunk shorter than len(text)/this is a
#: truncated/garbled generation.
_MAX_PLAUSIBLE_CHARS_PER_SEC = 40.0


@dataclass
class Plan:
    mode: str                 # auto | design | clone
    style: str                # "" when none
    bodies: list[str]         # normalized chunk texts (no style prefix)
    boundaries: list[str]     # break after each chunk
    khmer: bool = False

    @property
    def targets(self) -> list[str]:
        """Chunk texts as sent to the model (style prefix on every chunk —
        each chunk is an independent request with no shared decode state)."""
        if self.style and self.mode in ("design", "clone"):
            return [f"({self.style}){b}" for b in self.bodies]
        return list(self.bodies)


def build_plan(req: dict, *, max_chars_cap: int | None = None) -> Plan:
    """Turn a worker `synth` request into a chunk plan. Raises ValueError on
    invalid input (empty text, clone without reference)."""
    raw_text = (req.get("text") or "").strip()
    if not raw_text:
        raise ValueError("text must be non-empty")
    mode = req.get("mode") or ("clone" if req.get("ref_audio") else "auto")
    req_style = (req.get("instruct") or "").strip()
    inline_style, body = split_style_prefix(raw_text)
    style = ", ".join(s for s in (req_style, inline_style) if s)
    if mode == "auto" and inline_style:
        mode = "design"  # the user typed VoxCPM's inline "(style)" syntax
    if mode == "design" and not style:
        mode = "auto"
    if mode == "clone" and not req.get("ref_audio"):
        raise ValueError("clone mode requires ref_audio")

    body = body.strip() or raw_text
    khmer = is_khmer_text(body)
    if req.get("khmer_normalize", True) and khmer:
        body = normalize_khmer_text(body)
    else:
        body = neutralize_parentheses(clean_khmer_orthography(body)).strip()
    if not body:
        raise ValueError("text must be non-empty")

    max_chars = int(req.get("chunk_max_chars") or DEFAULT_CHUNK_MAX_CHARS)
    if max_chars_cap:
        max_chars = min(max_chars, int(max_chars_cap))
    max_chars = max(8, max_chars)  # guard against absurd configs (1-char chunks)
    chunks = chunk_khmer_text(body, max_chars=max_chars)
    if not chunks:
        return Plan(mode, style, [body], [BOUNDARY_PARAGRAPH], khmer)
    return Plan(mode, style, [c.text for c in chunks], [c.boundary for c in chunks], khmer)


@dataclass
class SynthStats:
    n_chunks: int = 0
    retries: int = 0
    anchored: bool = False
    notes: list[str] = field(default_factory=list)


ProgressFn = Callable[[int, int, str], None]


def _looks_truncated(n_samples: int, sample_rate: int, body: str) -> bool:
    chars = len(body.strip())
    if chars < 20 or sample_rate <= 0:
        return False
    return (n_samples / sample_rate) < chars / _MAX_PLAUSIBLE_CHARS_PER_SEC


async def _generate_checked(backend, text: str, body: str, cond: dict | None, gen: dict, retries: int, stats: SynthStats):
    """Generate one chunk, regenerating runaway / truncated results."""
    import numpy as np

    candidates: list[tuple[bool, Any]] = []  # (hit_cap, wav)
    base_seed = gen.get("seed")
    for attempt in range(retries + 1):
        g = dict(gen)
        if attempt and base_seed is not None:
            g["seed"] = int(base_seed) + attempt  # same seed would repeat the bad case
        wav, hit_cap = await backend.generate(text, cond, g)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        truncated = _looks_truncated(wav.size, backend.sample_rate, body)
        if not hit_cap and not truncated:
            return wav
        candidates.append((hit_cap, wav))
        if attempt < retries:
            stats.retries += 1
    # Every attempt was suspicious: prefer one that stopped on its own (the
    # longest of those), else the shortest runaway.
    stopped = [w for hit, w in candidates if not hit]
    if stopped:
        return max(stopped, key=lambda w: w.size)
    return min((w for _, w in candidates), key=lambda w: w.size)


async def synthesize_plan(
    backend,
    plan: Plan,
    *,
    gen: dict | None = None,
    ref_audio: str | None = None,
    prompt_text: str | None = None,
    anchor_voice: bool = True,
    retries: int = 2,
    progress: ProgressFn | None = None,
):
    """Run a plan on a backend. Returns (joined_wave, SynthStats)."""
    from .audio_join import float_to_wav_bytes, join_chunks, trim_silence

    gen = dict(gen or {})
    targets = plan.targets
    n = len(targets)
    stats = SynthStats(n_chunks=n)
    waves: list = [None] * n
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, n, f"Chunk {done}/{n}")

    cond: dict | None = None
    try:
        if plan.mode == "clone" and ref_audio:
            cond = await backend.condition(ref_path=ref_audio, transcript=(prompt_text or "").strip() or None)

        pending = list(range(n))
        if anchor_voice and plan.mode in ("auto", "design") and n > 1:
            waves[0] = await _generate_checked(backend, targets[0], plan.bodies[0], None, gen, retries, stats)
            _tick()
            anchor = trim_silence(waves[0], backend.sample_rate)
            anchor = anchor[: int(ANCHOR_MAX_SECONDS * backend.sample_rate)]
            if anchor.size >= int(0.5 * backend.sample_rate):
                cond = await backend.condition(wav_bytes=float_to_wav_bytes(anchor, backend.sample_rate), fmt="wav")
                stats.anchored = True
            pending = pending[1:]

        async def _one(i: int) -> None:
            waves[i] = await _generate_checked(backend, targets[i], plan.bodies[i], cond, gen, retries, stats)
            _tick()

        if backend.concurrent:
            await asyncio.gather(*(_one(i) for i in pending))
        else:
            for i in pending:
                await _one(i)
    finally:
        if cond is not None:
            try:
                await backend.release(cond)
            except Exception:  # noqa: BLE001
                pass

    full = join_chunks(waves, backend.sample_rate, plan.boundaries)
    return full, stats


# ─── Reference runtime (plain `voxcpm` PyTorch: CUDA / MPS / CPU) ────────────
class ReferenceBackend:
    """Sequential backend around `voxcpm.VoxCPM` (the upstream PyTorch runtime).

    Works everywhere VoxCPM2 itself does — CUDA, Apple Silicon (MPS, float32)
    and CPU — so it's the fallback when nanovllm can't run.

    Generation deliberately runs *inline* on the calling (event-loop) thread,
    the same thread the model was loaded on: on CUDA voxcpm wraps its steps in
    torch.compile(mode="reduce-overhead"), whose CUDA graphs are per-thread, so
    hopping between pool threads would re-record or break them. Nothing else
    runs concurrently in a worker, so blocking the loop between chunks is fine."""

    name = "reference"
    concurrent = False

    def __init__(self, model, *, device: str = "cpu", inference_timesteps: int = 10) -> None:
        self._model = model
        self.device = device
        self._inference_timesteps = inference_timesteps
        try:
            self.sample_rate = int(model.tts_model.sample_rate)
        except Exception:  # noqa: BLE001
            self.sample_rate = 48000

    async def condition(self, *, ref_path: str | None = None, wav_bytes: bytes | None = None,
                        fmt: str = "wav", transcript: str | None = None) -> dict:
        tmp = None
        if ref_path is None:
            fd, tmp = tempfile.mkstemp(suffix=f".{fmt or 'wav'}", prefix="voxcpm-anchor-")
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes or b"")
            ref_path = tmp
        cond: dict = {"reference_wav_path": ref_path, "_tmp": tmp}
        if transcript:
            # Ultimate cloning: continuation guided by the reference transcript.
            cond["prompt_wav_path"] = ref_path
            cond["prompt_text"] = transcript
        return cond

    async def release(self, cond: dict) -> None:
        tmp = cond.get("_tmp")
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _kwargs(self, text: str, cond: dict | None, gen: dict) -> dict:
        kwargs: dict = {"text": text}
        if gen.get("cfg_value") is not None:
            kwargs["cfg_value"] = float(gen["cfg_value"])
        steps = gen.get("inference_timesteps")
        kwargs["inference_timesteps"] = int(steps) if steps is not None else self._inference_timesteps
        for key in ("reference_wav_path", "prompt_wav_path", "prompt_text"):
            if cond and cond.get(key):
                kwargs[key] = cond[key]
        return kwargs

    def _generate_sync(self, kwargs: dict, seed: int | None):
        if seed is not None:
            try:
                import torch

                torch.manual_seed(int(seed))
            except Exception:  # noqa: BLE001
                pass
        return self._model.generate(**kwargs)

    async def generate(self, text: str, cond: dict | None, gen: dict):
        import numpy as np

        kwargs = self._kwargs(text, cond, gen)
        audio = self._generate_sync(kwargs, gen.get("seed"))
        arr = audio[0] if isinstance(audio, (list, tuple)) else audio
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().float().numpy()
        # voxcpm runs its own bad-case retry (retry_badcase), so no cap signal.
        return np.asarray(arr, dtype=np.float32).reshape(-1), False


def load_reference_model(model_id: str, device: str, log: Callable[[str], Any] | None = None):
    """Load `voxcpm.VoxCPM` on an explicit device.

    torch.compile (`optimize`) only works on CUDA; elsewhere it would just burn
    time on a warm-up generation, so it's disabled. voxcpm itself forces
    float32 on MPS (lower precision glitches the diffusion loop)."""
    from voxcpm import VoxCPM

    kwargs: dict = {"load_denoiser": False, "optimize": device == "cuda", "device": device}
    try:
        return VoxCPM.from_pretrained(model_id, **kwargs)
    except TypeError:
        # voxcpm < 2.0 has no device/optimize kwargs.
        if log:
            log("[voxcpm] old voxcpm without device selection — using its default device")
        return VoxCPM.from_pretrained(model_id, load_denoiser=False)


def default_progress_writer(reply: Callable[[dict], Any]) -> ProgressFn:
    def _p(done: int, total: int, stage: str) -> None:
        reply({"progress": {"done": done, "total": total, "stage": stage}})

    return _p

