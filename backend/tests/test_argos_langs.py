import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.translate.argos_langs import ARGOS_LANGS, required_packages


def test_lang_list_has_urdu_and_is_pairs():
    assert ("ur", "Urdu") in ARGOS_LANGS
    assert ("en", "English") in ARGOS_LANGS
    assert all(len(p) == 2 for p in ARGOS_LANGS)
    assert len(ARGOS_LANGS) >= 45


def test_required_packages_english_pivot():
    # non-English source and target -> two packages via English
    assert required_packages("fr", "ur") == [("fr", "en"), ("en", "ur")]


def test_required_packages_from_english():
    assert required_packages("en", "ur") == [("en", "ur")]


def test_required_packages_to_english():
    assert required_packages("fr", "en") == [("fr", "en")]


def test_required_packages_same_language_is_empty():
    assert required_packages("en", "en") == []
    assert required_packages("fr", "fr") == [("fr", "en"), ("en", "fr")]
