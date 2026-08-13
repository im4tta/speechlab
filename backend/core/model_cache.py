"""Detect whether a model repo's weights are present in the local HF cache.

`snapshot_download(..., local_files_only=True)` returns the snapshot path when
every file of the repo's current revision is already cached, and raises
otherwise — so it doubles as a "fully downloaded?" probe with no network call.
"""

from __future__ import annotations

from pathlib import Path

#: Weight-file extensions that mark a local model directory as populated.
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".ckpt")


def resolve_model_dir(path: str | Path) -> Path:
    """Return the actual directory whose files an engine should load.

    Handles HuggingFace-cache layouts transparently: pointing at a
    `models--org--name` cache root, its `snapshots` folder, or a snapshot
    revision folder all resolve down to the directory that directly holds the
    model files (``config.json`` + weights). An ordinary model folder is
    returned unchanged; a nonexistent path is returned as-is so callers can
    still report "not downloaded" / offer to fetch into it.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return p
    if (p / "config.json").is_file():
        return p
    snap = p if p.name == "snapshots" else p / "snapshots"
    if snap.is_dir():
        revs = sorted(
            (d for d in snap.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for rev in revs:
            if (rev / "config.json").is_file():
                return rev
    return p


def model_downloaded(repo_id: str) -> bool:
    """True if every file of `repo_id`'s current revision is cached locally.

    A `repo_id` that is an existing local directory (HF-cache path or a plain
    model folder) is treated as downloaded only when it actually holds weight
    files — an empty folder still counts as "not downloaded" so the UI can
    offer to fetch the weights into it.
    """
    local = Path(repo_id)
    if local.is_dir():
        resolved = resolve_model_dir(local)
        return any(resolved.rglob(f"*{ext}") for ext in _WEIGHT_EXTS)
    try:
        # Imported lazily so this module is import-safe before the HF cache
        # dir is configured (see backend/core/hf_paths.py).
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — any failure means "not fully cached"
        return False
