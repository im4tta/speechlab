import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.translate.m2m100_translator import M2M100Translator


def test_m2m_metadata_and_langs():
    t = M2M100Translator()
    assert t.name == "m2m100"
    assert t.display_name == "M2M-100 (418M)"
    assert t.license == "MIT"
    assert t.needs_source_lang() is True
    langs = t.languages()
    assert {"code": "ur", "label": "Urdu"} in langs
    assert {"code": "en", "label": "English"} in langs
    assert not t.is_loaded()


def test_m2m_large_identity_is_parameterized():
    # The 1.2B checkpoint reuses the same class with a distinct identity.
    t = M2M100Translator(
        model_id="facebook/m2m100_1.2B", name="m2m100_large",
        display_name="M2M-100 (1.2B)", model_url="https://huggingface.co/facebook/m2m100_1.2B")
    assert t.name == "m2m100_large"
    assert t.display_name == "M2M-100 (1.2B)"
    assert t.needs_source_lang() is True
    # same 100-language set as the 418M model
    assert {"code": "ur", "label": "Urdu"} in t.languages()
