"""Hardware → runtime decisions for VoxCPM2 (the `nanovllm_km` + `voxcpm` engines).

VoxCPM2 is a ~2B model. How it can run depends heavily on the machine:

- **Big NVIDIA GPU (Ampere+ / ≥ 8 GB):** nano-vllm-voxcpm's concurrent server
  pool — fastest, fans chunks out in parallel.
- **Smaller or older NVIDIA GPU:** nanovllm still works on 6–8 GB if its KV
  cache, batch width and CUDA-graph capture are scaled down; flash-attn (a
  hard nanovllm dependency) needs compute capability ≥ 8.0, so Turing/Pascal
  cards (GTX 16xx, RTX 20xx, T4) can't use it at all.
- **Apple Silicon (MPS), CPU, or tiny GPUs:** the reference `voxcpm` PyTorch
  runtime, one chunk at a time. On MPS it runs in float32 (the library forces
  that — lower precision glitches the diffusion loop).

This module holds the *pure* decision logic (no torch import at module load),
so it is unit-testable and importable from the isolated worker venvs, which
put `backend/` on sys.path. Probing the actual hardware (`probe_hardware`)
imports torch lazily.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: flash-attn 2 (required by nano-vllm-voxcpm) supports Ampere (sm80) and newer.
NANOVLLM_MIN_COMPUTE_CAPABILITY = (8, 0)
#: Below this much VRAM even the smallest nanovllm profile can't hold VoxCPM2's
#: weights plus a usable KV cache.
NANOVLLM_MIN_VRAM_GB = 6.0
#: The reference runtime allocates a static KV cache for the model's full
#: context on top of the weights; below this it doesn't fit on the GPU and we
#: run it on CPU instead (slow, but it works).
REFERENCE_MIN_VRAM_GB = 7.5


@dataclass(frozen=True)
class HardwareInfo:
    cuda: bool = False
    vram_gb: float | None = None
    compute_capability: tuple[int, int] | None = None
    gpu_name: str | None = None
    mps: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.compute_capability is not None:
            d["compute_capability"] = f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return d


@dataclass(frozen=True)
class NanoProfile:
    """nanovllm server-pool sizing for one VRAM tier."""

    name: str
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    #: Upper bound on characters per chunk, so prompt + generation always fit
    #: in max_model_len for this tier.
    chunk_max_chars: int

    def to_load_kwargs(self) -> dict:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
        }


#: Largest first. `min_vram_gb` is the total VRAM the tier is chosen for.
NANO_PROFILES: list[tuple[float, NanoProfile]] = [
    (20.0, NanoProfile("large", 16, 8192, 4096, 0.90, False, 220)),
    (12.0, NanoProfile("medium", 8, 4096, 4096, 0.90, False, 220)),
    # CUDA-graph capture costs a few hundred MB per batch size — skip it on
    # small cards and spend the memory on KV cache instead.
    (8.0, NanoProfile("small", 4, 4096, 4096, 0.92, True, 200)),
    (0.0, NanoProfile("tiny", 2, 2048, 2048, 0.94, True, 160)),
]


def nano_profile_ladder(vram_gb: float | None) -> list[NanoProfile]:
    """Profiles to try, best-fitting first, then progressively smaller ones
    (a load that OOMs on one tier is retried on the next)."""
    profiles = [p for _, p in NANO_PROFILES]
    if vram_gb is None:
        return profiles[1:]  # unknown VRAM: start from "medium"
    for i, (min_gb, _) in enumerate(NANO_PROFILES):
        if vram_gb >= min_gb:
            return profiles[i:]
    return profiles[-1:]


def apply_overrides(profile: NanoProfile, overrides: dict | None) -> NanoProfile:
    """Explicit user settings (e.g. NANOVLLM_MAX_NUM_SEQS) win over the tier."""
    if not overrides:
        return profile
    fields = asdict(profile)
    for key in ("max_num_seqs", "max_num_batched_tokens", "max_model_len",
                "gpu_memory_utilization", "enforce_eager", "chunk_max_chars"):
        val = overrides.get(key)
        if val is not None:
            fields[key] = type(fields[key])(val)
    # nanovllm asserts max_num_batched_tokens >= max_model_len.
    fields["max_num_batched_tokens"] = max(fields["max_num_batched_tokens"], fields["max_model_len"])
    return NanoProfile(**fields)


def nanovllm_blocker(hw: HardwareInfo) -> str | None:
    """Why nanovllm can't run on this hardware (None = it can)."""
    if not hw.cuda:
        if hw.mps:
            return "Apple Silicon (MPS) has no CUDA — nanovllm needs an NVIDIA GPU"
        return "no NVIDIA GPU (CUDA) detected"
    cc = hw.compute_capability
    if cc is not None and tuple(cc) < NANOVLLM_MIN_COMPUTE_CAPABILITY:
        return (
            f"GPU compute capability {cc[0]}.{cc[1]} is below 8.0 — flash-attn "
            "(required by nanovllm) needs an Ampere (RTX 30xx / A-series) or newer GPU"
        )
    if hw.vram_gb is not None and hw.vram_gb < NANOVLLM_MIN_VRAM_GB:
        return f"only {hw.vram_gb:.1f} GB VRAM (nanovllm needs ≥ {NANOVLLM_MIN_VRAM_GB:.0f} GB)"
    return None


def reference_device(hw: HardwareInfo, requested: str | None = "auto") -> str:
    """Device for the reference (plain PyTorch) VoxCPM runtime.

    An explicit cpu/mps/cuda request is honoured when that device exists; auto
    prefers a CUDA GPU with enough VRAM, then Apple MPS, then CPU."""
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        return "mps" if hw.mps else "cpu"
    if req.startswith("cuda"):
        return "cuda" if hw.cuda else ("mps" if hw.mps else "cpu")
    if hw.cuda and (hw.vram_gb is None or hw.vram_gb >= REFERENCE_MIN_VRAM_GB):
        return "cuda"
    if hw.mps:
        return "mps"
    return "cpu"


def choose_backend(hw: HardwareInfo, requested: str = "auto", *, nanovllm_importable: bool = True) -> tuple[str, str | None]:
    """Pick "nanovllm" or "reference". Returns (backend, reason_for_fallback)."""
    req = (requested or "auto").lower()
    if req == "reference":
        return "reference", "reference runtime requested"
    blocker = nanovllm_blocker(hw)
    if blocker is None and not nanovllm_importable:
        blocker = "nano-vllm-voxcpm / flash-attn isn't installed in this environment"
    if blocker is None:
        return "nanovllm", None
    return "reference", blocker


def max_generate_length(target_tokens: int, *, ratio: float = 6.0, floor: int = 10, hard_cap: int = 2000) -> int:
    """Cap on generated audio patches for one chunk.

    Same heuristic as the reference VoxCPM runtime's bad-case guard
    (tokens × 6 + 10): generous for any real reading speed, but stops the
    runaway "never emits stop" generations low-resource languages like Khmer
    are prone to, instead of letting them babble to the global limit."""
    return max(floor, min(hard_cap, int(target_tokens * ratio + floor)))


def probe_hardware() -> HardwareInfo:
    """Inspect the local torch runtime (imports torch lazily)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return HardwareInfo()
    mps = bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return HardwareInfo(
                cuda=True,
                vram_gb=round(props.total_memory / 1e9, 2),
                compute_capability=(int(props.major), int(props.minor)),
                gpu_name=str(props.name),
                mps=mps,
            )
    except Exception:  # noqa: BLE001
        pass
    return HardwareInfo(cuda=False, mps=mps)
