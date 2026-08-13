import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.tests.test_smoke import _make_client


def test_argos_in_status_with_on_demand(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    body = client.get("/api/translate/status").json()
    models = {m["name"]: m for m in body["models"]}
    assert "argos" in models
    assert models["argos"]["on_demand"] is True
    assert models["m2m100"]["on_demand"] is False   # local weights, not on-demand
    assert {"code": "ur", "label": "Urdu"} in models["argos"]["languages"]


def test_argos_absent_from_engines(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    names = {e["name"] for e in client.get("/api/engines").json()["engines"]}
    assert "argos" not in names
