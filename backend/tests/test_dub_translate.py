import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from backend.services.audio_pcm import pcm16_to_wav
from backend.services.dub import DubSegment, DubService


class _StubSynth:
    def __init__(self): self.texts = []
    def synthesize(self, req):
        from backend.services.synthesize import SynthResult
        self.texts.append(req.text)
        pcm = np.zeros(12000, dtype=np.int16).tobytes()
        return SynthResult(wav_bytes=pcm16_to_wav(pcm, 24000), sample_rate=24000,
                           duration_sec=0.5, inference_ms=1, engine="stub")


class _StubTranslate:
    def __init__(self): self.calls = []
    def translate(self, texts, source_lang, target_lang, model=None):
        from backend.core.translate import TranslateResult
        self.calls.append((tuple(texts), source_lang, target_lang, model))
        return TranslateResult([f"{t}-{target_lang}" for t in texts], source_lang or "auto", target_lang, 1)


def _segs():
    return [DubSegment(0.0, 1.0, "hello"), DubSegment(2.0, 3.0, "world")]


def test_target_language_translates_before_synth():
    synth, tr = _StubSynth(), _StubTranslate()
    svc = DubService(synth=synth, cache=None, translate=tr)
    svc.dub(_segs(), voice="v", engine="kokoro", source_language="en", target_language="ur")
    assert len(tr.calls) == 1
    assert synth.texts == ["hello-ur", "world-ur"]  # translated text reached synth


def test_no_target_language_skips_translation():
    synth, tr = _StubSynth(), _StubTranslate()
    svc = DubService(synth=synth, cache=None, translate=tr)
    svc.dub(_segs(), voice="v", engine="kokoro")
    assert tr.calls == []
    assert synth.texts == ["hello", "world"]


def test_target_equal_source_skips_translation():
    synth, tr = _StubSynth(), _StubTranslate()
    svc = DubService(synth=synth, cache=None, translate=tr)
    svc.dub(_segs(), voice="v", engine="kokoro", source_language="en", target_language="en")
    assert tr.calls == []


def test_target_language_changes_cache_hash():
    svc = DubService(synth=_StubSynth(), cache=None, translate=_StubTranslate())
    h_plain = svc.dub(_segs(), voice="v", engine="kokoro").cache_hash
    h_tr = svc.dub(_segs(), voice="v", engine="kokoro", source_language="en", target_language="ur").cache_hash
    assert h_plain != h_tr


def test_design_mode_translates_whole_transcript_once():
    synth, tr = _StubSynth(), _StubTranslate()
    svc = DubService(synth=synth, cache=None, translate=tr)
    svc.dub(_segs(), voice="", engine="omnivoice", voice_mode="design", instruct="x",
            source_language="en", target_language="ur")
    assert len(tr.calls) == 1
    assert len(synth.texts) == 1  # single synth call on the translated joined text
