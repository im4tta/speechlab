import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.translate import Translator
from backend.core.translate.argos_translator import ArgosTranslator


def test_metadata_and_capabilities(tmp_path):
    t = ArgosTranslator(packages_dir=tmp_path)
    assert t.name == "argos"
    assert t.on_demand() is True
    assert t.needs_source_lang() is True
    assert {"code": "ur", "label": "Urdu"} in t.languages()
    assert not t.is_loaded()


def test_base_translator_on_demand_defaults_false():
    class _T(Translator):
        def load(self): ...
        def unload(self): ...
        def is_loaded(self): return False
        def translate(self, req): raise NotImplementedError
    assert _T().on_demand() is False


def test_required_packages_used_for_pair():
    # The translator resolves the same package set as the helper.
    from backend.core.translate.argos_langs import required_packages
    assert required_packages("fr", "ur") == [("fr", "en"), ("en", "ur")]
