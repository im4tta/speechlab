"""Host hardware snapshot, captured lazily and cached per process.

Stored on every synthesis cache entry so each generation records the machine
that produced it (helpful when the same cache is moved between machines, or
when debugging which GPU/CPU path generated a take).
"""

from __future__ import annotations

import platform
import threading
from typing import Any

_MACHINE: dict[str, Any] | None = None
_LOCK = threading.Lock()


def _cpu_name() -> str | None:
    # Windows: read the friendly CPU brand from the registry.
    try:
        if platform.system() == "Windows":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                if name and name.strip():
                    return name.strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        name = platform.processor()
        if name and "family" not in name.lower():
            return name
    except Exception:  # noqa: BLE001
        pass
    return platform.machine()


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        return None


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{name} ({vram_gb:.1f} GB)"
    except Exception:  # noqa: BLE001
        pass
    return None


def machine_spec() -> dict[str, Any]:
    """A compact snapshot of the host that produced a generation."""
    global _MACHINE
    with _LOCK:
        if _MACHINE is None:
            _MACHINE = {
                "os": platform.system(),
                "cpu": _cpu_name(),
                "ram_gb": _ram_gb(),
                "gpu": _gpu_name(),
                "python": platform.python_version(),
            }
        return dict(_MACHINE)
