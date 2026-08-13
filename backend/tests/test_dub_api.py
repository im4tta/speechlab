import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from backend.services.audio_pcm import pcm16_to_wav
from backend.tests.test_smoke import _make_client


class _StubSynth:
    def synthesize(self, req):
        from backend.services.synthesize import SynthResult

        pcm = np.zeros(12000, dtype=np.int16).tobytes()  # 0.5s @ 24k
        return SynthResult(wav_bytes=pcm16_to_wav(pcm, 24000), sample_rate=24000,
                           duration_sec=0.5, inference_ms=1, engine="stub")


def _client(tmp_path):
    client = _make_client(tmp_path / "v", tmp_path / "u")
    # Inject the stub synth into the real DubService so its real DubCache
    # (which round-trips) drives the cache-hit path.
    client.app.state.dub_service._synth = _StubSynth()
    return client


def _body():
    return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"},
                         {"start": 2.0, "end": 3.0, "text": "world"}],
            "voice": "af_heart", "engine": "kokoro"}


def test_dub_returns_wav(tmp_path):
    r = _client(tmp_path).post("/api/dub", json=_body())
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.headers["X-Sample-Rate"] == "24000"
    assert r.headers["X-Cache"] == "miss"
    assert r.headers["X-Cache-Hash"].startswith("dub-")


def test_dub_second_call_hits_cache(tmp_path):
    client = _client(tmp_path)
    client.post("/api/dub", json=_body())
    r2 = client.post("/api/dub", json=_body())
    assert r2.headers["X-Cache"] == "hit"


def test_dub_empty_voice_clone_400(tmp_path):
    # Empty voice with no design/auto mode = clone with no reference -> 400.
    b = _body(); b["voice"] = ""
    assert _client(tmp_path).post("/api/dub", json=b).status_code == 400


def test_dub_design_mode_allows_empty_voice(tmp_path):
    # design/auto carry no reference voice, so an empty voice is fine.
    b = _body(); b["voice"] = ""; b["voice_mode"] = "design"; b["instruct"] = "young adult"
    r = _client(tmp_path).post("/api/dub", json=b)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"


def test_dub_no_segments_422(tmp_path):
    b = _body(); b["segments"] = []
    assert _client(tmp_path).post("/api/dub", json=b).status_code == 422


def test_dub_all_blank_segments_400(tmp_path):
    b = _body(); b["segments"] = [{"start": 0.0, "end": 1.0, "text": "  "}]
    assert _client(tmp_path).post("/api/dub", json=b).status_code == 400
