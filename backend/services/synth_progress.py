"""Live synthesis progress tracker (thread-safe singleton).

Engines that can report granular progress (e.g. the nanovllm worker reporting
chunks as they finish) push updates via `synth_progress.update(...)`; the
`/api/synthesize` route brackets the whole request with `begin`/`finish`. The
frontend polls `GET /api/synthesis/progress` for a live snapshot while it is
busy generating.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class SynthProgress:
    """Single-flight tracker for the in-flight synthesis request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def begin(self, engine: str | None, text: str | None) -> None:
        with self._lock:
            self._job = {
                "engine": engine,
                "text": (text or "")[:2000],
                "started_at": time.monotonic(),
                "done": 0,
                "total": None,
                "stage": "starting",
            }

    def update(self, *, done: int | None = None, total: int | None = None, stage: str | None = None) -> None:
        with self._lock:
            job = self._job
            if job is None:
                return
            if done is not None:
                job["done"] = int(done)
            if total is not None:
                job["total"] = int(total)
            if stage is not None:
                job["stage"] = stage

    def finish(self) -> None:
        with self._lock:
            self._job = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {
                    "state": "idle",
                    "engine": None,
                    "text": None,
                    "elapsed_sec": 0.0,
                    "done": 0,
                    "total": None,
                    "percent": None,
                    "eta_sec": None,
                    "stage": None,
                }
            elapsed = time.monotonic() - job["started_at"]
            total = job["total"]
            percent: float | None = None
            eta: float | None = None
            if total:
                percent = min(100.0, job["done"] * 100.0 / total)
                if percent > 0:
                    eta = elapsed / percent * (100.0 - percent)
            return {
                "state": "synthesizing",
                "engine": job["engine"],
                "text": job["text"],
                "elapsed_sec": round(elapsed, 1),
                "done": job["done"],
                "total": total,
                "percent": percent,
                "eta_sec": round(eta, 1) if eta is not None else None,
                "stage": job["stage"],
            }


#: Process-wide singleton the API + engines share.
synth_progress = SynthProgress()
