"""Background model-weight downloader with live progress.

Runs `snapshot_download` on a daemon thread inside the backend, folding byte
deltas into a shared progress snapshot that the UI polls. One download at a
time (weights are large). State machine: idle -> downloading -> done | error.

The downloader logic here is engine-agnostic; the network parts live in the
injectable `runner` (default `_default_runner`) so tests drive a fake.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from backend.scripts.download_models import MODEL_CATALOG

#: Engines whose weights this downloader can fetch (in-process engines).
DOWNLOADABLE: frozenset[str] = frozenset(
    {"vibevoice", "kokoro", "kitten", "omnivoice", "voxcpm", "qwen", "whisper",
     "m2m100", "m2m100_large"}
)

_MAX_LOG_LINES = 500
_SPEED_WINDOW = 30  # number of (ts, bytes) samples kept for speed/ETA


class DownloadCancelled(Exception):
    """Raised inside the runner when the user cancels an in-progress download."""

# A runner downloads `repo_id`, reporting progress via the given Progress.
Runner = Callable[[str, "Progress"], None]


class Progress:
    """The callback surface a runner uses to report download progress."""

    def __init__(self, downloader: "ModelDownloader") -> None:
        self._d = downloader

    def set_total(self, total: int) -> None:
        self._d._set_total(total)

    def add_bytes(self, n: int, current_file: Optional[str] = None) -> None:
        self._d._add_bytes(n, current_file)

    def log(self, line: str) -> None:
        self._d._log(line)

    def ignore_patterns(self) -> list[str]:
        return list(self._d._ignore)

    def local_dir(self) -> str | None:
        return self._d._local_dir

    def should_cancel(self) -> bool:
        return self._d._cancel_event.is_set()


class ModelDownloader:
    """Thread-safe, single-flight model download with a progress snapshot."""

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._runner: Runner = runner or _default_runner
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._engine: str | None = None
        self._state = "idle"
        self._downloaded = 0
        self._total: int | None = None
        self._current_file: str | None = None
        self._log_lines: list[str] = []
        self._error: str | None = None
        self._returncode: int | None = None
        self._local_dir: str | None = None
        self._repo_id: str | None = None
        self._samples: Deque[Tuple[float, int]] = deque(maxlen=_SPEED_WINDOW)
        self._cancel_event = threading.Event()
        self._ignore: list[str] = []

    # -- public API
    def status(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def start(self, engine: str, local_dir: str | None = None, repo_id: str | None = None) -> dict:
        if engine not in DOWNLOADABLE:
            raise ValueError(f"{engine} is not downloadable")
        with self._lock:
            if self._state == "downloading":
                if self._engine == engine:
                    return self._snapshot_locked()  # coalesce onto the same job
                raise ValueError(
                    f"a download for {self._engine} is already in progress"
                )
            # `repo_id` overrides the catalog default (used for ASR model variants).
            if repo_id:
                r_id = repo_id
                ignore = list(MODEL_CATALOG[engine].get("ignore") or [])
            else:
                r_id = MODEL_CATALOG[engine]["repo_id"]
                ignore = list(MODEL_CATALOG[engine].get("ignore") or [])
            self._engine = engine
            self._state = "downloading"
            self._downloaded = 0
            self._total = None
            self._current_file = None
            self._log_lines = []
            self._error = None
            self._returncode = None
            self._local_dir = local_dir
            self._repo_id = r_id
            self._ignore = ignore
            self._cancel_event.clear()
            self._samples.clear()
            self._samples.append((self._clock(), 0))
            self._thread = threading.Thread(
                target=self._run, args=(r_id,), daemon=True
            )
            self._thread.start()
            return self._snapshot_locked()

    def cancel(self, engine: str | None = None) -> dict:
        """Signal an in-progress download to stop. No-op if idle/done."""
        with self._lock:
            if self._state == "downloading" and (engine is None or engine == self._engine):
                self._cancel_event.set()
                self._log_lines.append("Cancelling…")
            return self._snapshot_locked()

    # -- internals
    def _run(self, repo_id: str) -> None:
        try:
            self._runner(repo_id, Progress(self))
        except DownloadCancelled:
            with self._lock:
                self._log_lines.append("Download cancelled.")
                self._state = "cancelled"
                self._returncode = 1
            return
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = str(exc)
                self._log_lines.append(f"[download error] {exc}")
                self._state = "error"
                self._returncode = -1
            return
        with self._lock:
            if self._total is not None:
                self._downloaded = self._total
            self._state = "done"
            self._returncode = 0

    def _set_total(self, total: int) -> None:
        with self._lock:
            self._total = int(total) if total and total > 0 else None

    def _add_bytes(self, n: int, current_file: Optional[str]) -> None:
        with self._lock:
            self._downloaded += int(n)
            if current_file:
                self._current_file = current_file
            self._samples.append((self._clock(), self._downloaded))

    def _log(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line)
            if len(self._log_lines) > _MAX_LOG_LINES:
                del self._log_lines[: len(self._log_lines) - _MAX_LOG_LINES]

    def _speed_bps_locked(self) -> float | None:
        if len(self._samples) < 2:
            return None
        t0, b0 = self._samples[0]
        t1, b1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return max(0.0, (b1 - b0) / dt)

    def _snapshot_locked(self) -> dict:
        speed = self._speed_bps_locked()
        percent: float | None = None
        if self._total:
            percent = min(100.0, self._downloaded * 100.0 / self._total)
        eta: float | None = None
        if speed and speed > 0 and self._total:
            eta = max(0, self._total - self._downloaded) / speed
        return {
            "engine": self._engine,
            "state": self._state,
            "percent": percent,
            "downloaded_bytes": self._downloaded,
            "total_bytes": self._total,
            "speed_bps": speed,
            "eta_sec": eta,
            "current_file": self._current_file,
            "local_dir": self._local_dir,
            "repo_id": self._repo_id,
            "log": list(self._log_lines),
            "error": self._error,
            "returncode": self._returncode,
        }


def _repo_total_bytes(repo_id: str, ignore: list[str] | None = None) -> int | None:
    """Best-effort total download size for a repo, excluding ignored files."""
    import fnmatch

    ignore = ignore or []
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, timeout=10)
        total = 0
        for sib in info.siblings or []:
            name = sib.rfilename
            if any(fnmatch.fnmatch(name, pat) for pat in ignore):
                continue
            size = getattr(sib, "size", None)
            if size is None:
                lfs = getattr(sib, "lfs", None)
                size = getattr(lfs, "size", None) if lfs else None
            if size:
                total += int(size)
        return total or None
    except Exception:  # noqa: BLE001
        return None


# The download runs in a CHILD PROCESS so cancellation can terminate it
# mid-file (snapshot_download is a blocking call a daemon thread can't interrupt).
# The child reads repo_id + optional local_dir + ignore patterns from argv and
# inherits HF_HOME.
_CHILD_SRC = (
    "import sys; from huggingface_hub import snapshot_download; "
    "local_dir = sys.argv[2] or None; "
    "snapshot_download(sys.argv[1], local_dir=local_dir, "
    "ignore_patterns=(sys.argv[3:] or None))"
)


def _default_runner(repo_id: str, progress: Progress) -> None:
    """Download `repo_id` into the local HF cache (or `local_dir`) with progress.

    When the engine is configured with a local model folder (`local_dir`), the
    weights are fetched straight into that folder instead of the HF cache. Runs
    snapshot_download in a subprocess so the user can cancel it (a blocking call
    in a daemon thread can't be interrupted). We poll the destination directory
    to advance the UI bar, and terminate the child on cancel.
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path as _Path

    from huggingface_hub.constants import HF_HUB_CACHE

    ignore = progress.ignore_patterns()
    local_dir = progress.local_dir()
    progress.log(f"Resolving {repo_id} …")
    total = _repo_total_bytes(repo_id, ignore)
    if total:
        progress.set_total(total)
        progress.log(f"Total download size: {total / (1024 * 1024):.0f} MB")
    else:
        progress.log("Total size unknown; reporting bytes downloaded.")

    if local_dir:
        # Weights land directly in the configured local model folder.
        dest_dir = _Path(local_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        # HF stores blobs (complete + *.incomplete in-progress) at:
        # {HF_HUB_CACHE}/models--{org}--{name}/blobs/
        dest_dir = _Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "blobs"

    def _dir_bytes(path: _Path) -> int:
        try:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except OSError:
            return 0

    baseline = _dir_bytes(dest_dir)  # bytes already present (resume scenario)
    if baseline > 0:
        progress.add_bytes(baseline)
    polled = 0

    log_file = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    log_path = _Path(log_file.name)
    log_file.close()
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SRC, repo_id, local_dir or "", *ignore],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        env=dict(os.environ),
    )
    cancelled = False
    try:
        while proc.poll() is None:
            if progress.should_cancel():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            written = _dir_bytes(dest_dir) - baseline
            delta = written - polled
            if delta > 0:
                progress.add_bytes(delta)
                polled = written
            time.sleep(0.5)
    finally:
        if proc.poll() is None:
            proc.kill()

    if cancelled:
        raise DownloadCancelled()
    if proc.returncode not in (0, None):
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
        except OSError:
            pass
        raise RuntimeError(f"snapshot_download exited {proc.returncode}. {tail}".strip())
    progress.log("Download complete.")
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass
