"""NanoVllmKhmerEngine proxy tests: capabilities, message building (no subprocess)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.engines import EngineSynthRequest  # noqa: E402
from backend.core.engines.nanovllm_engine import NanoVllmKhmerEngine  # noqa: E402


def _eng():
    return NanoVllmKhmerEngine()


def test_capabilities():
    e = _eng()
    assert e.name == "nanovllm_km"
    assert e.max_speakers() == 1
    assert e.supports_voice_cloning() is True
    assert e.supports_voice_modes() is True
    assert e.supports_style_clone() is True
    assert e.supports_streaming() is False
    assert e.default_cfg_scale() == 2.0
    assert e.sample_rate() == 48000  # VoxCPM2 output rate, before load()


def test_languages_lead_with_khmer():
    langs = _eng().languages()
    codes = [l["code"] for l in langs]
    assert codes[0] == ""  # Auto-detect first
    assert "km" in codes
    assert codes.index("km") < codes.index("en")


def test_available_voices_empty_by_default():
    # No built-in curated voices; voices come from the shared voice library.
    assert _eng().available_voices() == []


def test_build_msg_auto_mode_default():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(text="សួស្តី", voice_id=""), "/tmp/o.wav"
    )
    assert msg["op"] == "synth"
    assert msg["mode"] == "auto"
    assert msg["text"] == "សួស្តី"
    assert msg["khmer_normalize"] is True
    assert "ref_audio" not in msg
    assert "instruct" not in msg


def test_build_msg_design_mode_requires_no_ref():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(
            text="សួស្តី", voice_id="", voice_mode="design",
            instruct="young woman, warm, gentle",
        ),
        "/tmp/o.wav",
    )
    assert msg["mode"] == "design"
    assert msg["instruct"] == "young woman, warm, gentle"


def test_build_msg_design_mode_falls_back_to_auto_without_style():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(text="hi", voice_id="", voice_mode="design", instruct=""),
        "/tmp/o.wav",
    )
    assert msg["mode"] == "auto"


def test_build_msg_clone_mode_requires_ref_audio():
    with pytest.raises(ValueError):
        _eng()._build_synth_msg(
            EngineSynthRequest(text="hi", voice_id="", voice_mode="clone"),
            "/tmp/o.wav",
        )


def test_build_msg_clone_mode_with_ref_and_transcript():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(
            text="hi", voice_id="", voice_mode="clone",
            reference_audio="/tmp/ref.wav", reference_text="hello there",
        ),
        "/tmp/o.wav",
    )
    assert msg["mode"] == "clone"
    assert msg["ref_audio"] == "/tmp/ref.wav"
    assert msg["prompt_text"] == "hello there"


def test_build_msg_clone_infers_from_reference_audio_when_mode_unset():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(text="hi", voice_id="", reference_audio="/tmp/ref.wav"),
        "/tmp/o.wav",
    )
    assert msg["mode"] == "clone"


def test_build_msg_forwards_generation_params():
    msg = _eng()._build_synth_msg(
        EngineSynthRequest(
            text="hi", voice_id="", cfg_scale=2.5, inference_steps=15,
            temperature=0.8, seed=42,
        ),
        "/tmp/o.wav",
    )
    assert msg["cfg_value"] == 2.5
    assert msg["inference_timesteps"] == 15
    assert msg["temperature"] == 0.8
    assert msg["seed"] == 42


def test_build_msg_requires_nonempty_text():
    with pytest.raises(ValueError):
        _eng()._build_synth_msg(EngineSynthRequest(text="   ", voice_id=""), "/tmp/o.wav")


def test_ready_marker_path_matches_venv_layout():
    e = _eng()
    marker = e._ready_marker()
    assert marker.name == ".nanovllm-ready"
    assert marker.parent.name == "venv-nanovllm"


def test_load_msg_defaults_to_auto_runtime_and_omits_unset_sizing():
    msg = NanoVllmKhmerEngine()._load_msg()
    assert msg["device"] == "auto"
    assert msg["backend"] == "auto"
    for key in ("max_num_seqs", "gpu_memory_utilization", "max_model_len", "enforce_eager"):
        assert key not in msg  # None = let the worker size it from VRAM


def test_load_msg_forwards_explicit_sizing_and_device():
    msg = NanoVllmKhmerEngine(
        max_num_seqs=4, gpu_memory_utilization=0.8, enforce_eager=True,
        device_request="mps", backend="reference",
    )._load_msg()
    assert msg["max_num_seqs"] == 4
    assert msg["gpu_memory_utilization"] == 0.8
    assert msg["enforce_eager"] is True
    assert msg["device"] == "mps" and msg["backend"] == "reference"


def test_engine_info_before_load_reports_request():
    info = NanoVllmKhmerEngine(device_request="auto").engine_info()
    assert info["device"] == "auto"
    assert info["runtime"] is None
