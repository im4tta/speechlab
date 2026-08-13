import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.translate.lang_codes import normalize_source, M2M100_LANGS


def test_normalize_passes_known_iso_codes():
    assert normalize_source("en") == "en"
    assert normalize_source("ur") == "ur"


def test_normalize_maps_whisper_aliases():
    # Whisper emits some codes that differ from M2M's; the map fixes them.
    assert normalize_source("iw") == "he"   # legacy Hebrew
    assert normalize_source("jw") == "jv"   # Javanese
    assert normalize_source("Zh") == "zh"   # case-insensitive


def test_normalize_none_and_blank():
    assert normalize_source(None) is None
    assert normalize_source("") is None
    assert normalize_source("auto") is None


def test_language_lists_are_code_label_pairs():
    assert ("en", "English") in M2M100_LANGS
    assert ("ur", "Urdu") in M2M100_LANGS
    assert all(len(pair) == 2 for pair in M2M100_LANGS)
    assert len(M2M100_LANGS) >= 90
