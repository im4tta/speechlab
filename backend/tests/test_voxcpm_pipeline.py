"""core/voxcpm_pipeline.py: plan building + chunked synthesis with a fake backend."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.voxcpm_pipeline import build_plan, synthesize_plan  # noqa: E402

SR = 16000
LONG_KM = "ប្រយោគទីមួយនេះវែងបន្តិច។ ប្រយោគទីពីរក៏វែងដែរ។ ប្រយោគទីបីបញ្ចប់។"


class FakeBackend:
    """Records calls; returns ~1 s of tone per chunk."""

    name = "fake"
    sample_rate = SR

    def __init__(self, concurrent=False, hit_cap_first=0):
        self.concurrent = concurrent
        self.calls = []          # (text, cond)
        self.conditions = []     # kwargs passed to condition()
        self.released = []
        self._hit_cap_left = hit_cap_first

    async def condition(self, **kw):
        cond = {"id": len(self.conditions), **{k: v for k, v in kw.items() if k != "wav_bytes"}}
        self.conditions.append(kw)
        return cond

    async def release(self, cond):
        self.released.append(cond)

    async def generate(self, text, cond, gen):
        self.calls.append((text, cond, dict(gen)))
        t = np.arange(SR) / SR
        wav = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        if self._hit_cap_left > 0:
            self._hit_cap_left -= 1
            return wav, True
        return wav, False


def _run(coro):
    return asyncio.run(coro)


def test_plan_inline_style_applies_to_every_chunk():
    plan = build_plan({"text": "(calm, slow) " + LONG_KM, "chunk_max_chars": 30})
    assert plan.mode == "design"
    assert plan.style == "calm, slow"
    assert len(plan.targets) > 1
    assert all(t.startswith("(calm, slow)") for t in plan.targets)


def test_plan_combines_request_and_inline_style():
    plan = build_plan({"text": "(slow) សួស្តី", "mode": "design", "instruct": "warm"})
    assert plan.style == "warm, slow"


def test_plan_neutralizes_parentheses_in_non_khmer_text():
    plan = build_plan({"text": "Hello there. (aside) more text", "chunk_max_chars": 14})
    assert all(not t.lstrip().startswith("(") for t in plan.targets)


def test_plan_clone_requires_reference():
    with pytest.raises(ValueError):
        build_plan({"text": "hi", "mode": "clone"})


def test_auto_mode_anchors_voice_after_first_chunk():
    backend = FakeBackend()
    plan = build_plan({"text": LONG_KM, "chunk_max_chars": 30})
    assert len(plan.targets) >= 2
    wav, stats = _run(synthesize_plan(backend, plan))
    assert stats.anchored is True
    # First chunk unconditioned, the rest conditioned on the anchor clip.
    assert backend.calls[0][1] is None
    assert all(c[1] is not None for c in backend.calls[1:])
    assert backend.conditions[0].get("wav_bytes")  # anchor built from audio
    assert backend.released  # conditioning released afterwards
    assert wav.size > SR * len(plan.targets) * 0.9


def test_anchor_can_be_disabled():
    backend = FakeBackend()
    plan = build_plan({"text": LONG_KM, "chunk_max_chars": 30})
    _wav, stats = _run(synthesize_plan(backend, plan, anchor_voice=False))
    assert stats.anchored is False
    assert all(c[1] is None for c in backend.calls)


def test_clone_mode_conditions_every_chunk_on_reference_and_transcript():
    backend = FakeBackend(concurrent=True)
    plan = build_plan({"text": LONG_KM, "mode": "clone", "ref_audio": "/tmp/r.wav", "chunk_max_chars": 30})
    _wav, stats = _run(synthesize_plan(backend, plan, ref_audio="/tmp/r.wav", prompt_text="hello"))
    assert stats.anchored is False
    assert backend.conditions == [{"ref_path": "/tmp/r.wav", "transcript": "hello"}]
    assert all(c[1] is not None for c in backend.calls)


def test_runaway_chunk_is_regenerated_with_new_seed():
    backend = FakeBackend(hit_cap_first=1)
    plan = build_plan({"text": "សួស្តី"})
    _wav, stats = _run(synthesize_plan(backend, plan, gen={"seed": 7}))
    assert stats.retries == 1
    assert [c[2].get("seed") for c in backend.calls] == [7, 8]


def test_progress_reported_per_chunk():
    backend = FakeBackend()
    plan = build_plan({"text": LONG_KM, "chunk_max_chars": 30})
    seen = []
    _run(synthesize_plan(backend, plan, progress=lambda d, t, s: seen.append((d, t))))
    assert seen[-1] == (len(plan.targets), len(plan.targets))
