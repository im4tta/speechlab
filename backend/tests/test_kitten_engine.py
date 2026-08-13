"""Kitten TTS engine — metadata, voices, registration.

No real synthesis here (that needs the ONNX weights); the E2E covers that.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from backend.core.engines.kitten_engine import (  # noqa: E402
    MODEL_ID,
    KittenEngine,
    _KITTEN_VOICES,
    _voice_spec,
)
from backend.tests.test_smoke import _make_client  # noqa: E402


def test_kitten_metadata_and_capabilities():
    e = KittenEngine()
    assert e.name == "kitten"
    assert e.display_name == "Kitten TTS Mini"
    assert e.license == "Apache-2.0"
    assert MODEL_ID == "KittenML/kitten-tts-mini-0.8"
    assert e.sample_rate() == 24000
    assert e.max_speakers() == 1
    assert e.supports_voice_cloning() is False
    assert e.default_cfg_scale() is None
    assert e.is_loaded() is False
    assert e.languages() == [{"code": "en", "label": "English"}]


def test_kitten_has_eight_aliased_voices():
    assert len(_KITTEN_VOICES) == 8
    names = {v.name for v in _KITTEN_VOICES}
    # The model's own config.json aliases, not invented.
    assert names == {"Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"}
    # 4 women, 4 men; every id is a raw expr-voice-* the model expects.
    assert sum(v.gender == "woman" for v in _KITTEN_VOICES) == 4
    assert sum(v.gender == "man" for v in _KITTEN_VOICES) == 4
    assert all(v.id.startswith("expr-voice-") for v in _KITTEN_VOICES)


def test_kitten_available_voices_shape():
    voices = KittenEngine().available_voices()
    assert len(voices) == 8
    v = next(x for x in voices if x.name == "Bella")
    assert v.id == "expr-voice-2-f"
    assert v.gender == "woman"
    assert v.source == "builtin"
    assert v.sample_rate == 24000


def test_kitten_voice_spec_lookup():
    assert _voice_spec("expr-voice-2-f").name == "Bella"
    assert _voice_spec("nope") is None


def test_kitten_unknown_voice_raises_not_degrades():
    """generate() silently degrades on a bad voice, so synthesize must reject it
    before ever calling the model."""
    from backend.core.engines import EngineSynthRequest

    e = KittenEngine()
    with pytest.raises(ValueError, match="unknown Kitten TTS voice"):
        e.synthesize(EngineSynthRequest(text="hi", voice_id="Jasper-not-a-real-id"))


def test_kitten_registered_and_downloadable_deletable():
    from backend.scripts.download_models import MODEL_CATALOG
    from backend.services.model_download import DOWNLOADABLE
    from backend.services.model_delete import DELETABLE

    assert MODEL_CATALOG["kitten"]["repo_id"] == "KittenML/kitten-tts-mini-0.8"
    assert "kitten" in DOWNLOADABLE
    assert "kitten" in DELETABLE


def test_kitten_in_engine_selector(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    engines = {e["name"]: e for e in client.get("/api/engines").json()["engines"]}
    assert "kitten" in engines
    k = engines["kitten"]
    assert k["license"] == "Apache-2.0"
    assert k["max_speakers"] == 1
    assert k["supports_voice_cloning"] is False


def test_kitten_voices_listed(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    voices = [v for v in client.get("/api/voices").json()["voices"] if v["engine"] == "kitten"]
    assert len(voices) == 8
    assert {v["name"] for v in voices} == {
        "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
    }
