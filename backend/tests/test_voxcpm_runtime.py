"""core/voxcpm_runtime.py: hardware → runtime/profile decisions (no torch needed)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.voxcpm_runtime import (  # noqa: E402
    HardwareInfo,
    apply_overrides,
    choose_backend,
    max_generate_length,
    nano_profile_ladder,
    nanovllm_blocker,
    reference_device,
)

BIG = HardwareInfo(cuda=True, vram_gb=24.0, compute_capability=(8, 9))
MID = HardwareInfo(cuda=True, vram_gb=12.0, compute_capability=(8, 6))
SMALL = HardwareInfo(cuda=True, vram_gb=6.4, compute_capability=(8, 6))
TURING = HardwareInfo(cuda=True, vram_gb=11.0, compute_capability=(7, 5))
TINY = HardwareInfo(cuda=True, vram_gb=4.0, compute_capability=(8, 6))
MAC = HardwareInfo(cuda=False, mps=True)
CPU = HardwareInfo()


def test_profile_ladder_starts_at_the_fitting_tier_and_steps_down():
    assert [p.name for p in nano_profile_ladder(24)] == ["large", "medium", "small", "tiny"]
    assert [p.name for p in nano_profile_ladder(12)] == ["medium", "small", "tiny"]
    assert [p.name for p in nano_profile_ladder(8.5)] == ["small", "tiny"]
    assert [p.name for p in nano_profile_ladder(6.4)] == ["tiny"]


def test_small_profiles_skip_cuda_graphs_and_shrink_chunks():
    tiny = nano_profile_ladder(6.4)[0]
    assert tiny.enforce_eager is True
    assert tiny.max_model_len <= 2048
    assert tiny.chunk_max_chars < 220
    assert tiny.max_num_batched_tokens >= tiny.max_model_len


def test_explicit_overrides_win_and_keep_nanovllm_invariant():
    p = apply_overrides(nano_profile_ladder(24)[0], {"max_num_seqs": 3, "max_model_len": 8192})
    assert p.max_num_seqs == 3
    assert p.max_model_len == 8192
    assert p.max_num_batched_tokens >= 8192  # nanovllm asserts batched >= model_len


def test_nanovllm_blockers():
    assert nanovllm_blocker(BIG) is None
    assert nanovllm_blocker(SMALL) is None
    assert "compute capability" in nanovllm_blocker(TURING)
    assert "VRAM" in nanovllm_blocker(TINY)
    assert "Apple Silicon" in nanovllm_blocker(MAC)
    assert "no NVIDIA" in nanovllm_blocker(CPU)


def test_choose_backend():
    assert choose_backend(BIG) == ("nanovllm", None)
    assert choose_backend(MAC)[0] == "reference"
    assert choose_backend(BIG, nanovllm_importable=False)[0] == "reference"
    assert choose_backend(BIG, "reference")[0] == "reference"


def test_reference_device():
    assert reference_device(MAC) == "mps"
    assert reference_device(CPU) == "cpu"
    assert reference_device(BIG) == "cuda"
    assert reference_device(TURING) == "cuda"  # pre-Ampere but big enough
    assert reference_device(TINY) == "cpu"     # too small for the 2B model
    assert reference_device(BIG, "cpu") == "cpu"
    assert reference_device(CPU, "mps") == "cpu"  # requested but not present


def test_max_generate_length_scales_with_text_and_is_capped():
    assert max_generate_length(10) == 70
    assert max_generate_length(0) == 10
    assert max_generate_length(10_000) == 2000
