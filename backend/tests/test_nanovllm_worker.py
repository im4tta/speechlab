"""Tests for nanovllm_worker.py's `_build_targets` (pure logic, no torch/asyncio)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from nanovllm_worker import _build_targets  # noqa: E402


def test_auto_mode_single_short_sentence():
    pieces, mode = _build_targets({"text": "សួស្តី។", "mode": "auto"})
    assert mode == "auto"
    assert pieces == ["សួស្តី។"]


def test_design_mode_prefixes_every_chunk_with_style():
    req = {
        "text": "ប្រយោគទីមួយ។ ប្រយោគទីពីរ។",
        "mode": "design",
        "instruct": "young woman, warm",
        "chunk_max_chars": 12,
    }
    pieces, mode = _build_targets(req)
    assert mode == "design"
    assert len(pieces) >= 2
    for p in pieces:
        assert p.startswith("(young woman, warm)")


def test_design_mode_without_style_falls_back_to_auto():
    pieces, mode = _build_targets({"text": "hi there", "mode": "design", "instruct": ""})
    assert mode == "auto"
    assert not pieces[0].startswith("(")


def test_clone_mode_requires_ref_audio():
    with pytest.raises(ValueError):
        _build_targets({"text": "hi", "mode": "clone"})


def test_clone_mode_with_style_prefixes_chunks():
    req = {
        "text": "សួស្តី",
        "mode": "clone",
        "ref_audio": "/tmp/ref.wav",
        "instruct": "cheerful",
    }
    pieces, mode = _build_targets(req)
    assert mode == "clone"
    assert pieces[0].startswith("(cheerful)")


def test_clone_mode_without_style_leaves_text_unprefixed():
    req = {"text": "សួស្តី", "mode": "clone", "ref_audio": "/tmp/ref.wav"}
    pieces, mode = _build_targets(req)
    assert pieces == ["សួស្តី"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        _build_targets({"text": "   "})


def test_khmer_normalize_can_be_disabled():
    req = {"text": "តម្លៃ 100 ដុល្លារ", "khmer_normalize": False}
    pieces, _mode = _build_targets(req)
    assert "100" in pieces[0]


def test_khmer_normalize_default_on_expands_numbers():
    req = {"text": "តម្លៃ 100 ដុល្លារ"}
    pieces, _mode = _build_targets(req)
    assert "100" not in pieces[0]


def test_long_text_splits_into_multiple_chunks():
    req = {"text": "ប្រយោគមួយខ្លីៗ។ " * 30, "chunk_max_chars": 60}
    pieces, _mode = _build_targets(req)
    assert len(pieces) > 1
    for p in pieces:
        assert len(p) <= 60
