import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from backend.services.dub import _reconstruct_timeline

SR = 24000


def _pcm(n_samples: int) -> bytes:
    return np.ones(n_samples, dtype=np.int16).tobytes()  # non-zero so we can find it


def _samples(pcm: bytes) -> int:
    return len(pcm) // 2


def test_leading_silence_equals_first_start():
    # first segment starts at 1.0s; 0.5s of audio
    out = _reconstruct_timeline([(1.0, 1.5, _pcm(SR // 2))], SR)
    total = _samples(out)
    assert total == SR + SR // 2  # 1.0s silence + 0.5s audio


def test_inter_segment_gap_is_original_pause():
    # seg A: [1.0, 1.5] with 0.5s audio; seg B starts at 2.0 (0.5s pause after A's end)
    pieces = [(1.0, 1.5, _pcm(SR // 2)), (2.0, 2.4, _pcm(SR // 4))]
    out = _reconstruct_timeline(pieces, SR)
    # 1.0s lead + 0.5s A + 0.5s gap (2.0-1.5) + 0.25s B
    expected = SR + SR // 2 + SR // 2 + SR // 4
    assert _samples(out) == expected


def test_segment_audio_is_natural_length_not_stretched():
    # original span is 5s but synthesized audio is only 1s — we keep 1s (no stretch)
    out = _reconstruct_timeline([(0.0, 5.0, _pcm(SR))], SR)
    assert _samples(out) == SR  # exactly the 1s of audio, no lead (start=0)


def test_overlapping_or_negative_gap_clamped_to_zero():
    # seg B starts before A ends (bad timing) -> gap clamps to 0, no negative silence
    pieces = [(0.0, 2.0, _pcm(SR)), (1.0, 3.0, _pcm(SR))]
    out = _reconstruct_timeline(pieces, SR)
    assert _samples(out) == SR + SR  # 1s A + 0 gap + 1s B


def test_empty_pieces_returns_empty():
    assert _reconstruct_timeline([], SR) == b""
