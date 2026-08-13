"""Persistent YouTube transcript history (one JSON file, LRU-capped).

Stores past YouTube scribe results so the user can reopen a transcript without
re-fetching, and to record whether each came from manual captions, auto-generated
CC, or our Whisper ASR fallback.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MAX_ENTRIES = 60


class YoutubeHistory:
    def __init__(self, dir_path: Path, max_entries: int = _MAX_ENTRIES) -> None:
        self._dir = Path(dir_path)
        self._file = self._dir / "history.json"
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    # -- storage
    def _load(self) -> None:
        try:
            if self._file.is_file():
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._entries = data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to load YouTube history: %s", exc)
            self._entries = {}

    def _persist(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to save YouTube history: %s", exc)

    # -- public
    def save(self, entry: dict[str, Any]) -> None:
        video_id = entry.get("video_id")
        if not video_id:
            return
        with self._lock:
            self._entries[video_id] = {**entry, "created_at": time.time()}
            # LRU-cap: drop the oldest entries beyond the limit.
            if len(self._entries) > self._max_entries:
                for vid in sorted(
                    self._entries, key=lambda v: self._entries[v].get("created_at", 0)
                )[: len(self._entries) - self._max_entries]:
                    self._entries.pop(vid, None)
            self._persist()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda e: e.get("created_at", 0),
                reverse=True,
            )

    def get(self, video_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(video_id)

    def delete(self, video_id: str) -> bool:
        with self._lock:
            if video_id not in self._entries:
                return False
            del self._entries[video_id]
            self._persist()
            return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries = {}
            self._persist()
            return n
