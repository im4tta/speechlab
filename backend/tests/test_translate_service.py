import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest

from backend.core.exceptions import BackendError
from backend.core.translate import TranslateRequest, TranslateResult, Translator
from backend.services.translate import TranslateService
from backend.services.translate_cache import TranslateCache


class _Stub(Translator):
    def __init__(self, name, needs_source=False):
        self.name = name
        self.display_name = name
        self._needs = needs_source
        self.calls = []
        self._loaded = False

    def load(self): self._loaded = True
    def unload(self): self._loaded = False
    def is_loaded(self): return self._loaded
    def downloaded(self): return True
    def needs_source_lang(self): return self._needs
    def languages(self): return [{"code": "ur", "label": "Urdu"}]

    def translate(self, req: TranslateRequest) -> TranslateResult:
        self.calls.append(req)
        return TranslateResult(texts=[t.upper() for t in req.texts],
                               source_lang=req.source_lang or "auto",
                               target_lang=req.target_lang, inference_ms=1)


def _svc(tmp_path, **kw):
    trs = {"m2m100": _Stub("m2m100", needs_source=True), "madlad": _Stub("madlad")}
    return TranslateService(translators=trs, default="m2m100",
                            cache=TranslateCache(tmp_path / "c"), **kw), trs


def test_translate_batches_and_returns(tmp_path):
    svc, trs = _svc(tmp_path)
    r = svc.translate(["hi", "bye"], source_lang="en", target_lang="ur")
    assert r.texts == ["HI", "BYE"]
    assert len(trs["m2m100"].calls) == 1  # one batched call


def test_second_identical_call_hits_cache(tmp_path):
    svc, trs = _svc(tmp_path)
    svc.translate(["hi"], source_lang="en", target_lang="ur")
    svc.translate(["hi"], source_lang="en", target_lang="ur")
    assert len(trs["m2m100"].calls) == 1  # second served from cache


def test_model_override_selects_translator(tmp_path):
    svc, trs = _svc(tmp_path)
    svc.translate(["hi"], source_lang="en", target_lang="ur", model="madlad")
    assert len(trs["madlad"].calls) == 1
    assert len(trs["m2m100"].calls) == 0


def test_set_active_switches_and_unloads_previous(tmp_path):
    svc, trs = _svc(tmp_path)
    svc.translate(["hi"], source_lang="en", target_lang="ur")  # loads m2m100
    trs["m2m100"].load()
    svc.set_active("madlad")
    assert svc.active_name == "madlad"
    assert trs["m2m100"].is_loaded() is False


def test_missing_source_for_m2m_raises_400(tmp_path):
    svc, _ = _svc(tmp_path)
    with pytest.raises(BackendError) as e:
        svc.translate(["hi"], source_lang=None, target_lang="ur")
    assert e.value.http_status == 400


def test_status_lists_models_and_active(tmp_path):
    svc, _ = _svc(tmp_path)
    st = svc.status()
    assert st["active"] == "m2m100"
    names = {m["name"] for m in st["models"]}
    assert names == {"m2m100", "madlad"}
