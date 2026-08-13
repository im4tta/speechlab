"""Health and config endpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..config import BACKEND_ROOT, Settings, get_settings
from ..core.engine_manager import EngineManager
from ..core.model import ModelManager
from ..core.model_cache import resolve_model_dir
from ..core.version import get_version
from .deps import get_engine_manager, get_model_manager
from .schemas import ConfigResponse, EngineInfoModel, HealthResponse

router = APIRouter(tags=["health"])


class VoxcpmModelPathBody(BaseModel):
    path: str | None = None  # None or "" clears the local path


class VoxcpmModelPathResponse(BaseModel):
    path: str | None
    effective: bool  # True once a backend restart has applied it
    requires_restart: bool = True
    downloaded: bool = False
    exists: bool = False


def _env_file() -> Path:
    return BACKEND_ROOT / ".env"


def _set_env_var(key: str, value: str | None) -> None:
    """Set/update `key` in backend/.env, preserving the rest of the file."""
    path = _env_file()
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    out: list[str] = []
    matched = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix) or (
            stripped.startswith("#") and stripped[1:].lstrip().startswith(prefix)
        ):
            matched = True
            if value is not None:
                out.append(f"{key}={value}")
            continue
        out.append(line)
    if value is not None and not matched:
        out.append(f"{key}={value}")
    if value is None and not matched:
        return  # nothing to change
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


@router.post("/api/config/voxcpm-model-path", response_model=VoxcpmModelPathResponse)
def set_voxcpm_model_path(
    body: VoxcpmModelPathBody,
    settings: Settings = Depends(get_settings),
) -> VoxcpmModelPathResponse:
    """Point the `voxcpm` / `nanovllm_km` engines at a local VoxCPM2 folder.

    Persists VOXCPM_MODEL_PATH to backend/.env. Takes effect on the next
    backend start. If the folder is empty, the engines report it as "not
    downloaded" so the Download button fetches the weights straight into it.
    """
    value = (body.path or "").strip() or None
    _set_env_var("VOXCPM_MODEL_PATH", value)
    resolved = resolve_model_dir(value) if value else None
    return VoxcpmModelPathResponse(
        path=str(resolved) if resolved else None,
        effective=bool(settings.voxcpm_model_path),
        requires_restart=True,
        downloaded=resolved is not None and any(
            p.suffix in (".safetensors", ".bin", ".pt", ".ckpt") for p in resolved.rglob("*")
        ),
        exists=resolved is not None and resolved.is_dir(),
    )


@router.post("/api/config/choose-voxcpm-folder", response_model=VoxcpmModelPathResponse)
def choose_voxcpm_folder(
    settings: Settings = Depends(get_settings),
) -> VoxcpmModelPathResponse:
    """Open a native folder picker on the server and use the chosen directory.

    The dialog runs on the machine hosting the backend (the machine that has
    the model files), so a browser-only folder selection is not required.
    """
    chosen = _native_folder_picker("Select your local VoxCPM2 model folder")
    if not chosen:
        return VoxcpmModelPathResponse(
            path=str(settings.voxcpm_model_path) if settings.voxcpm_model_path else None,
            effective=bool(settings.voxcpm_model_path),
            exists=settings.voxcpm_model_path is not None and settings.voxcpm_model_path.is_dir(),
        )
    return set_voxcpm_model_path(VoxcpmModelPathBody(path=chosen), settings)


def _native_folder_picker(description: str) -> str | None:
    """Cross-platform native directory chooser. Returns the picked path or None."""
    if os.name == "nt":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.Description = '{description}'; "
            "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ $f.SelectedPath }"
        )
        for exe in ("powershell.exe", "pwsh.exe"):
            try:
                proc = subprocess.run(
                    [exe, "-NoProfile", "-STA", "-Command", script],
                    capture_output=True, text=True, timeout=300,
                )
                out = (proc.stdout or "").strip()
                if proc.returncode == 0 and out:
                    return out
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None
    if sys.platform == "darwin":
        script = (
            'tell application "System Events" to set theFolder to '
            f'choose folder with prompt "{description}"'
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=300,
            )
            out = (proc.stdout or "").strip().strip("\r\n")
            return out
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
    # Linux: zenity, then kdialog.
    for exe, args in (("zenity", ["--file-selection", "--directory", "--title", description]),
                      ("kdialog", ["--getexistingdirectory", description])):
        try:
            proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=300)
            out = (proc.stdout or "").strip()
            if proc.returncode == 0 and out:
                return out
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


@router.get("/api/health", response_model=HealthResponse)
def health(
    em: EngineManager = Depends(get_engine_manager),
) -> HealthResponse:
    """Liveness probe. Returns 200 when the active engine is loaded."""
    engine = em.active_engine
    info = engine.engine_info()
    return HealthResponse(
        status="ok" if engine.is_loaded() else "loading",
        model_loaded=engine.is_loaded(),
        device=info.get("device", "unknown"),
        version=get_version(),
    )


@router.get("/api/config", response_model=ConfigResponse)
def config(
    em: EngineManager = Depends(get_engine_manager),
    mm: ModelManager | None = Depends(get_model_manager),
    settings: Settings = Depends(get_settings),
) -> ConfigResponse:
    """Server-wide configuration + active engine's runtime details.

    `model_id`, `device`, `dtype`, `attn_implementation`, and
    `sampling_rate` reflect the *active* engine. When the active engine
    is VibeVoice, we source those from its ModelManager directly (the
    most accurate values). For other engines, we delegate to the
    engine's `engine_info()` method, which each engine overrides with
    its own honest values (or "unknown" if it doesn't track them).
    """
    engine = em.active_engine
    if engine.name == "vibevoice" and mm is not None and mm.is_loaded:
        device_name = mm.device_name
        dtype_name = mm.dtype_name
        attn_impl = mm.attn_impl
        sample_rate = mm.sampling_rate
        model_id = mm.model_id
    else:
        info = engine.engine_info()
        device_name = info["device"]
        dtype_name = info["dtype"]
        attn_impl = info["attn_implementation"]
        sample_rate = engine.sample_rate() if engine.is_loaded() else 0
        model_id = info["model_id"]

    return ConfigResponse(
        version=get_version(),
        model_id=model_id,
        device=device_name,
        dtype=dtype_name,
        attn_implementation=attn_impl,
        sampling_rate=sample_rate,
        default_cfg_scale=settings.default_cfg_scale,
        max_text_chars=settings.max_text_chars,
        voices_dir=str(settings.voices_dir),
        uploads_dir=str(settings.uploads_dir),
        active_engine=em.active_name,
        voxcpm_model_path=str(settings.voxcpm_model_path) if settings.voxcpm_model_path else None,
        engines=[
            EngineInfoModel(
                name=info["name"],
                display_name=info["display_name"],
                description=info["description"],
                license=info.get("license", "unknown"),
                model_url=info.get("model_url", ""),
                loaded=info["loaded"],
                supports_voice_cloning=info["supports_voice_cloning"],
                supports_streaming=info.get("supports_streaming", False),
                sample_rate=info.get("sample_rate"),
                max_speakers=info["max_speakers"],
                default_cfg_scale=info["default_cfg_scale"],
                active=info.get("active", False),
                supports_voice_modes=info.get("supports_voice_modes", False),
                supports_style_clone=info.get("supports_style_clone", False),
                supports_style_prompt=info.get("supports_style_prompt", False),
            )
            for info in em.info()
        ],
    )