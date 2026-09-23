"""Stdlib-only environment detection for the Voice Studio launcher.

Imported by ``studio.py`` BEFORE any venv exists, so it must not import
any third-party package.
"""

from __future__ import annotations

import re
import shutil
import subprocess

CUDA_TAG_TO_INDEX: dict[str, str] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
}


def parse_nvidia_smi_cuda_version(text: str) -> str | None:
    """Extract the ``CUDA Version: X.Y`` field from nvidia-smi output."""
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", text)
    return m.group(1) if m else None


def cuda_version_to_tag(version: str | None) -> str | None:
    """Map a CUDA runtime version (e.g. '12.4') to a PyTorch wheel tag."""
    if not version:
        return None
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except ValueError:
        return None
    # cu124 is the newest wheel build we ship. Drivers reporting CUDA 12.4+
    # (including 13.x — modern drivers report e.g. "CUDA Version: 13.2") run
    # cu124 wheels natively; only 12.0–12.3 lack the 12.4 runtime → cu121.
    if major >= 13:
        return "cu124"
    if major == 12:
        return "cu124" if minor >= 4 else "cu121"
    if major == 11:
        return "cu118"
    return None


def torch_index_url(tag: str | None) -> str | None:
    """Map a wheel tag to a ``--index-url``; None means the default wheel."""
    if tag in (None, "cpu", "mps"):
        return None
    return CUDA_TAG_TO_INDEX.get(tag)


def _run_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def detect_cuda_tag(runner=None) -> str | None:
    """Detect the best CUDA wheel tag. ``runner`` is injectable for tests."""
    run = runner or _run_nvidia_smi
    text = run()
    if text is None:
        return None
    return cuda_version_to_tag(parse_nvidia_smi_cuda_version(text))


def cuda_version_to_omnivoice_tag(version: str | None) -> str | None:
    """Map a CUDA runtime version to a torch 2.8 wheel tag for OmniVoice.

    OmniVoice needs torch 2.8, whose CUDA builds are cu126 and cu128 (there is
    no cu124 torch-2.8 wheel). Drivers below CUDA 12.6 fall back to the CPU
    build. This is separate from `cuda_version_to_tag`, which targets the
    torch 2.6 builds used by the main/Chatterbox venvs.
    """
    if not version:
        return None
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except ValueError:
        return None
    if major >= 13:
        return "cu128"
    if major == 12:
        if minor >= 8:
            return "cu128"
        if minor >= 6:
            return "cu126"
    return None


def detect_omnivoice_cuda_tag(runner=None) -> str | None:
    """Detect the torch-2.8 CUDA wheel tag for OmniVoice. `runner` is injectable."""
    run = runner or _run_nvidia_smi
    text = run()
    if text is None:
        return None
    return cuda_version_to_omnivoice_tag(parse_nvidia_smi_cuda_version(text))


def cuda_version_to_voxcpm_tag(version: str | None) -> str | None:
    """Map a CUDA runtime version to a torch wheel tag for VoxCPM.

    VoxCPM needs torch>=2.5; we install a torch 2.8 CUDA build whose wheels are
    cu126/cu128 (same as OmniVoice). Drivers below CUDA 12.6 fall back to CPU.
    """
    return cuda_version_to_omnivoice_tag(version)


def detect_voxcpm_cuda_tag(runner=None) -> str | None:
    """Detect the torch CUDA wheel tag for VoxCPM. `runner` is injectable."""
    run = runner or _run_nvidia_smi
    text = run()
    if text is None:
        return None
    return cuda_version_to_voxcpm_tag(parse_nvidia_smi_cuda_version(text))


def cuda_version_to_qwen_tag(version: str | None) -> str | None:
    """Map a CUDA runtime version to a torch wheel tag for Qwen3-TTS.

    qwen-tts needs a modern torch (transformers==4.57.3); we install a torch
    2.8 CUDA build (cu126/cu128, same as OmniVoice/VoxCPM). Below 12.6 → CPU.
    """
    return cuda_version_to_omnivoice_tag(version)


def detect_qwen_cuda_tag(runner=None) -> str | None:
    """Detect the torch CUDA wheel tag for Qwen. `runner` is injectable."""
    run = runner or _run_nvidia_smi
    text = run()
    if text is None:
        return None
    return cuda_version_to_qwen_tag(parse_nvidia_smi_cuda_version(text))


def cuda_version_to_nanovllm_tag(version: str | None) -> str | None:
    """Map a CUDA runtime version to a torch wheel tag for nanovllm-voxcpm.

    nano-vllm-voxcpm needs torch>=2.5 and flash-attn (CUDA-only, no CPU
    path); we install a torch 2.8 CUDA build (cu126/cu128, same tag scheme
    as OmniVoice/VoxCPM/Qwen) so flash-attn wheels resolve consistently.
    """
    return cuda_version_to_omnivoice_tag(version)


def detect_nanovllm_cuda_tag(runner=None) -> str | None:
    """Detect the torch CUDA wheel tag for nanovllm-voxcpm. `runner` is injectable."""
    run = runner or _run_nvidia_smi
    text = run()
    if text is None:
        return None
    return cuda_version_to_nanovllm_tag(parse_nvidia_smi_cuda_version(text))


# ─── NVIDIA GPU details + nanovllm (Khmer) install profile ─────────────────

def _run_nvidia_smi_query() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    # Drivers older than ~510 reject the compute_cap field — retry without it.
    for fields in ("name,memory.total,compute_cap", "name,memory.total"):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    return None


def parse_nvidia_gpu_query(text: str | None) -> dict | None:
    """Parse the first line of
    `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits`
    into {"name", "vram_gb", "compute_capability": (major, minor) | None}.

    Older drivers don't know `compute_cap` (the field comes back as
    "[N/A]" or the whole query fails) — that just leaves it None."""
    if not text:
        return None
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_gb = round(float(parts[1]) / 1024.0, 1)  # MiB -> GiB
    except ValueError:
        return None
    cc = None
    if len(parts) >= 3:
        m = re.fullmatch(r"(\d+)\.(\d+)", parts[2])
        if m:
            cc = (int(m.group(1)), int(m.group(2)))
    return {"name": parts[0], "vram_gb": vram_gb, "compute_capability": cc}


def detect_nvidia_gpu(runner=None) -> dict | None:
    """Name / VRAM / compute capability of GPU 0, or None. `runner` is injectable."""
    run = runner or _run_nvidia_smi_query
    return parse_nvidia_gpu_query(run())


#: flash-attn 2 (required by nano-vllm-voxcpm) needs Ampere (sm80) or newer,
#: and VoxCPM2 + a usable KV cache need ~6 GB. Mirrors
#: backend/core/voxcpm_runtime.py, which makes the same call at runtime.
NANOVLLM_MIN_COMPUTE_CAPABILITY = (8, 0)
NANOVLLM_MIN_VRAM_GB = 6.0


def nanovllm_install_profile(cuda_tag: str | None, gpu: dict | None,
                             override: str | None = None) -> tuple[str, str]:
    """Which runtime to install into backend/venv-nanovllm.

    "full" = nano-vllm-voxcpm (+ flash-attn) AND the reference voxcpm runtime
             as an automatic fallback — for NVIDIA Ampere+ GPUs with ≥ 6 GB
             and a CUDA 12.6+ driver.
    "lite" = only the reference voxcpm runtime — Apple Silicon (MPS), CPU-only
             machines, and older/smaller NVIDIA GPUs. flash-attn can't be
             built/used there, so installing nanovllm would just fail.
    Returns (profile, human-readable reason). `override` ("full"/"lite",
    e.g. from NANOVLLM_INSTALL_PROFILE) wins.
    """
    if override in ("full", "lite"):
        return override, f"forced by NANOVLLM_INSTALL_PROFILE={override}"
    if not cuda_tag:
        if gpu:
            return "lite", "NVIDIA driver is older than CUDA 12.6 (nanovllm needs torch 2.8 cu126/cu128)"
        return "lite", "no NVIDIA GPU (Apple Silicon / CPU) — using the reference VoxCPM2 runtime"
    if gpu:
        cc = gpu.get("compute_capability")
        if cc is not None and tuple(cc) < NANOVLLM_MIN_COMPUTE_CAPABILITY:
            return "lite", (f"{gpu.get('name', 'GPU')} is compute capability {cc[0]}.{cc[1]}; "
                            "flash-attn needs 8.0+ (RTX 30xx / A-series or newer)")
        vram = gpu.get("vram_gb")
        if vram is not None and vram < NANOVLLM_MIN_VRAM_GB:
            return "lite", f"{gpu.get('name', 'GPU')} has {vram} GB VRAM; nanovllm needs ≥ 6 GB"
    return "full", "NVIDIA GPU supports nanovllm (reference runtime installed as fallback)"
