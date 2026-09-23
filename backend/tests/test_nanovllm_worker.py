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


# ─── runtime selection (simulated hardware, no torch/nanovllm needed) ─────────

import asyncio  # noqa: E402

import nanovllm_worker  # noqa: E402
from core.voxcpm_runtime import HardwareInfo  # noqa: E402


class _FakeRefModel:
    class tts_model:
        sample_rate = 48000

    def generate(self, **kwargs):
        import numpy as np

        t = np.arange(2 * 48000) / 48000  # 2 s of tone — a plausible take
        return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _patch_hw(monkeypatch, hw, *, nano_importable=False):
    loaded = {}

    def _fake_load(model_id, device, log=None):
        loaded["device"] = device
        return _FakeRefModel()

    monkeypatch.setattr(nanovllm_worker, "probe_hardware", lambda: hw)
    monkeypatch.setattr(nanovllm_worker, "_nanovllm_importable", lambda: (nano_importable, None))
    monkeypatch.setattr(nanovllm_worker, "load_reference_model", _fake_load)
    monkeypatch.setattr(nanovllm_worker, "_reply", lambda obj: None)
    return loaded


def test_mac_loads_reference_runtime_on_mps(monkeypatch):
    loaded = _patch_hw(monkeypatch, HardwareInfo(cuda=False, mps=True))
    resp = asyncio.run(nanovllm_worker._Worker().handle({"op": "load"}))
    assert resp["ok"] is True
    assert resp["backend"] == "reference"
    assert resp["device"] == "mps" and loaded["device"] == "mps"
    assert "Apple Silicon" in resp["fallback_reason"]
    assert resp["sample_rate"] == 48000


def test_small_gpu_falls_back_to_cpu_reference(monkeypatch):
    hw = HardwareInfo(cuda=True, vram_gb=4.0, compute_capability=(8, 6))
    _patch_hw(monkeypatch, hw, nano_importable=True)
    resp = asyncio.run(nanovllm_worker._Worker().handle({"op": "load"}))
    assert resp["backend"] == "reference" and resp["device"] == "cpu"


def test_forcing_nanovllm_on_a_mac_is_an_error(monkeypatch):
    _patch_hw(monkeypatch, HardwareInfo(cuda=False, mps=True))
    resp = asyncio.run(nanovllm_worker._Worker().handle({"op": "load", "backend": "nanovllm"}))
    assert resp["ok"] is False


def test_reference_runtime_synthesizes_khmer_end_to_end(monkeypatch, tmp_path):
    _patch_hw(monkeypatch, HardwareInfo(cuda=False, mps=True))
    worker = nanovllm_worker._Worker()
    asyncio.run(worker.handle({"op": "load"}))
    out = tmp_path / "o.wav"
    resp = asyncio.run(worker.handle({
        "op": "synth", "text": "សួស្តី។ " * 40, "out_wav": str(out), "chunk_max_chars": 60,
    }))
    assert resp["ok"] is True
    assert resp["n_chunks"] > 1
    assert resp["anchored"] is True
    assert out.is_file() and out.stat().st_size > 44


def test_nano_backend_caps_generation_to_fit_context():
    b = nanovllm_worker._NanoBackend(
        pool=None, sample_rate=48000, feat_dim=64, patch_size=4, max_model_len=2048,
    )
    # Byte-count token estimate for Khmer (3 bytes/char) without a tokenizer.
    assert b.plan_cap("ក" * 10, None, None) == min(2000, 30 * 6 + 10)
    # A reference clip's latent patches eat into the context budget.
    latents = b"\0" * (4 * 64 * 4 * 500)  # 500 patches
    cap = b.plan_cap("ក" * 200, {"ref_audio_latents": latents}, None)
    assert cap <= 2048 - (600 + 1 + 502)
    with pytest.raises(ValueError):
        b.plan_cap("ក" * 1000, None, None)
