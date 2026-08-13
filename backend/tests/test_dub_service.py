import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from backend.core.exceptions import BackendError
from backend.services.audio_pcm import pcm16_to_wav
from backend.services.dub import DubSegment, DubService


class _StubSynth:
    """Stub SynthService: returns 0.5s of silence per call, records the calls."""

    def __init__(self, sr=24000):
        self.sr = sr
        self.calls = []

    def synthesize(self, req):
        from backend.services.synthesize import SynthResult

        self.calls.append(req)
        pcm = np.zeros(self.sr // 2, dtype=np.int16).tobytes()  # 0.5s
        return SynthResult(
            wav_bytes=pcm16_to_wav(pcm, self.sr), sample_rate=self.sr,
            duration_sec=0.5, inference_ms=1, engine=req.engine or "stub",
        )


def _segs():
    return [DubSegment(0.0, 1.0, "hello"), DubSegment(2.0, 3.0, "world")]


def test_dub_synthesizes_each_nonempty_segment():
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    r = svc.dub(_segs(), voice="af_heart", engine="kokoro")
    assert len(synth.calls) == 2
    assert synth.calls[0].text == "hello"
    assert synth.calls[0].speakers[0].voice_id == "af_heart"
    assert synth.calls[0].engine == "kokoro"
    assert r.sample_rate == 24000
    assert r.wav_bytes[:4] == b"RIFF"
    # 0s lead + 0.5s + (2.0-1.0=1.0s gap) + 0.5s = 2.0s
    assert abs(r.duration_sec - 2.0) < 0.01


def test_empty_text_segments_are_skipped():
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    segs = [DubSegment(0.0, 1.0, "hi"), DubSegment(1.0, 2.0, "   "), DubSegment(2.0, 3.0, "bye")]
    svc.dub(segs, voice="v")
    assert len(synth.calls) == 2  # the blank one is skipped


def test_no_nonempty_segments_raises_400():
    svc = DubService(synth=_StubSynth(), cache=None)
    with pytest.raises(BackendError) as e:
        svc.dub([DubSegment(0.0, 1.0, "  ")], voice="v")
    assert e.value.http_status == 400


def test_empty_voice_raises_400():
    svc = DubService(synth=_StubSynth(), cache=None)
    with pytest.raises(BackendError) as e:
        svc.dub(_segs(), voice="")
    assert e.value.http_status == 400


def test_design_mode_allows_empty_voice():
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    svc.dub([DubSegment(0.0, 1.0, "hi")], voice="", voice_mode="design", instruct="young adult")
    assert len(synth.calls) == 1
    assert synth.calls[0].speakers[0].voice_mode == "design"
    assert synth.calls[0].speakers[0].instruct == "young adult"


def test_design_mode_synthesizes_full_text_in_one_call():
    # No reference voice -> one call for the whole transcript (consistent voice),
    # not one per segment (which would drift).
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    segs = [DubSegment(0.0, 1.0, "hello"), DubSegment(2.0, 3.0, "world"), DubSegment(4.0, 5.0, "bye")]
    svc.dub(segs, voice="", voice_mode="design", instruct="x")
    assert len(synth.calls) == 1
    assert synth.calls[0].text == "hello world bye"


def test_auto_mode_synthesizes_in_one_call():
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    svc.dub([DubSegment(0.0, 1.0, "a"), DubSegment(2.0, 3.0, "b")], voice="", voice_mode="auto")
    assert len(synth.calls) == 1


def test_clone_mode_still_synthesizes_per_segment():
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=None)
    svc.dub([DubSegment(0.0, 1.0, "a"), DubSegment(2.0, 3.0, "b")], voice="v", voice_mode="clone")
    assert len(synth.calls) == 2


def test_voice_mode_and_instruct_change_hash():
    svc = DubService(synth=_StubSynth(), cache=None)
    base = svc.dub([DubSegment(0.0, 1.0, "hi")], voice="v").cache_hash
    design = svc.dub([DubSegment(0.0, 1.0, "hi")], voice="", voice_mode="design", instruct="a").cache_hash
    design2 = svc.dub([DubSegment(0.0, 1.0, "hi")], voice="", voice_mode="design", instruct="b").cache_hash
    assert base != design != design2 and design != design2


class _StubCache:
    def __init__(self, tmp):
        self.enabled = True
        self.store = {}
        self._tmp = tmp

    def get(self, h):
        return self.store.get(h)

    def put(self, join_hash, wav_bytes, sample_rate, duration_sec, inference_ms, segment_hashes=None):
        from types import SimpleNamespace

        p = self._tmp / f"{join_hash}.wav"
        p.write_bytes(wav_bytes)
        self.store[join_hash] = SimpleNamespace(
            wav_path=p, sample_rate=sample_rate, duration_sec=duration_sec, inference_ms=inference_ms)


def test_identical_request_hits_cache(tmp_path):
    synth = _StubSynth()
    svc = DubService(synth=synth, cache=_StubCache(tmp_path))
    r1 = svc.dub(_segs(), voice="v", engine="kokoro")
    assert r1.cache_hit is False
    n_after_first = len(synth.calls)
    r2 = svc.dub(_segs(), voice="v", engine="kokoro")
    assert r2.cache_hit is True
    assert r2.cache_hash == r1.cache_hash
    assert len(synth.calls) == n_after_first  # no re-synthesis


def test_editing_text_changes_hash(tmp_path):
    svc = DubService(synth=_StubSynth(), cache=_StubCache(tmp_path))
    h1 = svc.dub([DubSegment(0.0, 1.0, "hello")], voice="v").cache_hash
    h2 = svc.dub([DubSegment(0.0, 1.0, "hullo")], voice="v").cache_hash
    assert h1 != h2
