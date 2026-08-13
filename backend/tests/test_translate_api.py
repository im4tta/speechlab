import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.translate import TranslateRequest, TranslateResult, Translator
from backend.tests.test_smoke import _make_client


class _Stub(Translator):
    name = "m2m100"
    display_name = "M2M"
    _loaded = False

    def load(self): self._loaded = True
    def unload(self): self._loaded = False
    def is_loaded(self): return self._loaded
    def downloaded(self): return True
    def needs_source_lang(self): return True
    def languages(self): return [{"code": "ur", "label": "Urdu"}]

    def translate(self, req: TranslateRequest) -> TranslateResult:
        return TranslateResult([t.upper() for t in req.texts], req.source_lang or "auto", req.target_lang, 1)


def _client(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    from backend.services.translate import TranslateService
    svc = TranslateService(translators={"m2m100": _Stub()}, default="m2m100")
    client.app.state.translate_service = svc
    return client


def test_status_lists_models(tmp_path):
    r = _client(tmp_path).get("/api/translate/status")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == "m2m100"
    assert body["models"][0]["languages"] == [{"code": "ur", "label": "Urdu"}]


def test_translate_segments(tmp_path):
    r = _client(tmp_path).post("/api/translate", json={
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "source_lang": "en", "target_lang": "ur"})
    assert r.status_code == 200, r.text
    assert r.json()["segments"][0]["text"] == "HELLO"


def test_translate_missing_source_400(tmp_path):
    r = _client(tmp_path).post("/api/translate", json={
        "texts": ["hello"], "target_lang": "ur"})
    assert r.status_code == 400


def test_translators_absent_from_engines(tmp_path):
    r = _client(tmp_path).get("/api/engines")
    names = {e["name"] for e in r.json()["engines"]}
    assert "m2m100" not in names and "madlad" not in names
