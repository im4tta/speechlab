"""core/audio_join.py: trimming, loudness matching, boundary-aware joins."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.audio_join import (  # noqa: E402
    PAUSE_SECONDS,
    float_to_wav_bytes,
    join_chunks,
    match_loudness,
    speech_level,
    trim_silence,
)

SR = 16000


def _tone(seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _padded(seconds: float, lead: float, tail: float, amp: float = 0.5) -> np.ndarray:
    return np.concatenate([np.zeros(int(lead * SR), np.float32), _tone(seconds, amp),
                           np.zeros(int(tail * SR), np.float32)])


def test_trim_silence_keeps_short_pad():
    wav = _padded(1.0, lead=0.8, tail=1.2)
    out = trim_silence(wav, SR, pad_s=0.05)
    assert 1.0 <= len(out) / SR <= 1.15


def test_trim_silence_leaves_all_silent_input_alone():
    wav = np.zeros(SR, np.float32)
    assert len(trim_silence(wav, SR)) == SR


def test_match_loudness_evens_levels_within_bounds():
    loud, soft = _tone(1.0, 0.6), _tone(1.0, 0.3)
    a, b = match_loudness([loud, soft], SR)
    assert abs(speech_level(a, SR) - speech_level(b, SR)) < 0.05
    # ±6 dB cap: a 40 dB quieter chunk is only raised by 2x, not flattened.
    whisper = _tone(1.0, 0.006)
    _, w2 = match_loudness([loud, whisper], SR)
    assert speech_level(w2, SR) <= speech_level(whisper, SR) * 2.01


def test_join_inserts_boundary_pauses():
    waves = [_tone(0.5), _tone(0.5), _tone(0.5)]
    out = join_chunks(waves, SR, ["sentence", "paragraph", "paragraph"], trim=False, level=False)
    expected = 1.5 + PAUSE_SECONDS["sentence"] + PAUSE_SECONDS["paragraph"]
    assert abs(len(out) / SR - expected) < 0.01


def test_join_single_chunk_and_peak_guard():
    out = join_chunks([_tone(0.5, amp=1.5)], SR)
    assert np.max(np.abs(out)) <= 0.971
    assert join_chunks([], SR).size == 0


def test_float_to_wav_bytes_roundtrip_header():
    data = float_to_wav_bytes(_tone(0.1), SR)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
