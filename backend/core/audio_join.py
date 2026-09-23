"""Seamless joining of independently generated TTS chunks.

Long Khmer passages are synthesized as several independent requests (see
core/khmer_text.py). Naively concatenating the raw waveforms with a fixed gap
gives audible seams: each chunk carries its own random amount of leading /
trailing silence, its own loudness, and hard edges that click. This module
fixes all three:

- `trim_silence` cuts each chunk's leading/trailing silence down to a short pad
  (relative to the chunk's own level, so quiet voices aren't eaten).
- `match_loudness` nudges every chunk toward the median speech level (bounded
  gain, so a whispered chunk isn't pumped into noise).
- `join_chunks` inserts a pause sized by the *kind* of break that ended the
  chunk (paragraph > sentence > clause > mid-clause) and applies short fades
  so there are no clicks.

numpy-only on purpose: it's imported by the isolated worker venvs
(nanovllm / voxcpm) via a sys.path insert, not just the main backend.
"""

from __future__ import annotations

import numpy as np

#: Pause (seconds) inserted after a chunk, keyed by the boundary that ended it
#: (see core.khmer_text.BOUNDARY_*). Tuned for natural Khmer read speech.
PAUSE_SECONDS: dict[str, float] = {
    "paragraph": 0.55,
    "sentence": 0.32,
    "clause": 0.16,
    "none": 0.05,
}

_FRAME_S = 0.01


def pause_for_boundary(boundary: str | None) -> float:
    return PAUSE_SECONDS.get(boundary or "sentence", PAUSE_SECONDS["sentence"])


def _frame_rms(wav: np.ndarray, frame: int) -> np.ndarray:
    n = len(wav) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = wav[: n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)


def trim_silence(
    wav: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -40.0,
    pad_s: float = 0.05,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping `pad_s` of breathing room.

    Silence = 10 ms frames whose RMS is `threshold_db` below the loudest frame.
    Returns the input unchanged when it's empty or entirely below threshold
    (never turn a quiet chunk into nothing)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame = max(1, int(sample_rate * _FRAME_S))
    rms = _frame_rms(wav, frame)
    if rms.size == 0 or float(rms.max()) <= 1e-6:
        return wav
    thresh = float(rms.max()) * (10.0 ** (threshold_db / 20.0))
    voiced = np.nonzero(rms > thresh)[0]
    if voiced.size == 0:
        return wav
    pad = int(pad_s * sample_rate)
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(len(wav), (int(voiced[-1]) + 1) * frame + pad)
    return wav[start:end]


def speech_level(wav: np.ndarray, sample_rate: int) -> float:
    """RMS of the voiced frames only (so pauses don't drag the level down)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame = max(1, int(sample_rate * _FRAME_S))
    rms = _frame_rms(wav, frame)
    if rms.size == 0 or float(rms.max()) <= 1e-6:
        return 0.0
    voiced = rms[rms > float(rms.max()) * 0.1]
    return float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2))) if voiced.size else 0.0


def match_loudness(
    waves: list[np.ndarray],
    sample_rate: int,
    *,
    max_gain_db: float = 6.0,
) -> list[np.ndarray]:
    """Scale each chunk toward the median speech level of all chunks.

    Gain is clamped to ±max_gain_db so an intentionally soft/loud delivery is
    evened out, not flattened."""
    if len(waves) < 2:
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
    levels = [speech_level(w, sample_rate) for w in waves]
    audible = [lv for lv in levels if lv > 1e-5]
    if not audible:
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
    target = float(np.median(audible))
    max_gain = 10.0 ** (max_gain_db / 20.0)
    out: list[np.ndarray] = []
    for w, lv in zip(waves, levels):
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if lv <= 1e-5:
            out.append(w)
            continue
        gain = min(max_gain, max(1.0 / max_gain, target / lv))
        out.append(w * np.float32(gain))
    return out


def _fade(wav: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(wav) < 2 * n:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def join_chunks(
    waves: list[np.ndarray],
    sample_rate: int,
    boundaries: list[str] | None = None,
    *,
    trim: bool = True,
    level: bool = True,
    fade_s: float = 0.008,
    peak: float = 0.97,
) -> np.ndarray:
    """Join chunk waveforms into one natural-sounding take.

    `boundaries[i]` is the break that ended chunk i (paragraph/sentence/clause/
    none) and decides the pause inserted after it. With a single chunk only the
    trim + peak guard apply."""
    if not waves:
        return np.zeros(0, dtype=np.float32)
    boundaries = list(boundaries or [])
    clean = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
    if trim:
        clean = [trim_silence(w, sample_rate) for w in clean]
    if level:
        clean = match_loudness(clean, sample_rate)
    fade_n = int(fade_s * sample_rate)
    pieces: list[np.ndarray] = []
    for i, w in enumerate(clean):
        pieces.append(_fade(w, fade_n) if len(clean) > 1 else w)
        if i < len(clean) - 1:
            b = boundaries[i] if i < len(boundaries) else None
            pieces.append(np.zeros(int(pause_for_boundary(b) * sample_rate), dtype=np.float32))
    full = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    top = float(np.max(np.abs(full))) if full.size else 0.0
    if top > peak:  # loudness matching can push a peak over full scale
        full = full * np.float32(peak / top)
    return full.astype(np.float32, copy=False)


def float_to_wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float waveform as 16-bit PCM WAV bytes (in memory)."""
    import io
    import wave

    arr = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
