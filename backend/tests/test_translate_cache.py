import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.translate_cache import TranslateCache


def test_roundtrip(tmp_path):
    c = TranslateCache(tmp_path / "tr", enabled=True)
    assert c.get("hello", "en", "ur", "m2m100") is None
    c.put("hello", "en", "ur", "m2m100", "ہیلو")
    assert c.get("hello", "en", "ur", "m2m100") == "ہیلو"


def test_key_folds_all_fields(tmp_path):
    c = TranslateCache(tmp_path / "tr", enabled=True)
    c.put("hello", "en", "ur", "m2m100", "A")
    assert c.get("hello", "en", "fr", "m2m100") is None      # target differs
    assert c.get("hello", "en", "ur", "madlad") is None      # model differs
    assert c.get("hello", "de", "ur", "m2m100") is None      # source differs


def test_disabled_is_noop(tmp_path):
    c = TranslateCache(tmp_path / "tr", enabled=False)
    c.put("hello", "en", "ur", "m2m100", "x")
    assert c.get("hello", "en", "ur", "m2m100") is None
