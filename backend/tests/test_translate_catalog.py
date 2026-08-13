import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.download_models import MODEL_CATALOG
from backend.services.model_download import DOWNLOADABLE
from backend.services.model_delete import DELETABLE


def test_translation_models_are_downloadable_and_deletable():
    for key, repo in (("m2m100", "facebook/m2m100_418M"), ("m2m100_large", "facebook/m2m100_1.2B")):
        assert MODEL_CATALOG[key]["repo_id"] == repo
        assert key in DOWNLOADABLE
        assert key in DELETABLE
        # non-PyTorch weight formats are skipped so we don't fetch dead gigabytes
        assert "*.ot" in MODEL_CATALOG[key]["ignore"]
