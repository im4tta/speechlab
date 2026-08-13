import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from backend.services.audio_pcm import pcm16_from_wav, pcm16_to_wav


def test_roundtrip_wav_pcm_wav():
    pcm = (np.zeros(2400, dtype=np.int16)).tobytes()
    wav = pcm16_to_wav(pcm, 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    back_pcm, sr, n = pcm16_from_wav(wav)
    assert sr == 24000
    assert n == 2400
    assert back_pcm == pcm


def test_download_module_reexports_them():
    # api/download.py must keep working via the shared helpers.
    from backend.api import download
    assert download._pcm16_from_wav is pcm16_from_wav
    assert download._pcm16_to_wav is pcm16_to_wav
