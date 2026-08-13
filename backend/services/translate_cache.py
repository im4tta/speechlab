"""On-disk translation cache: one JSON per (text, source, target, model) hash."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


class TranslateCache:
    def __init__(self, cache_dir: Path | str, enabled: bool = True) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _hash(self, text: str, source: str | None, target: str, model: str) -> str:
        canonical = json.dumps({"t": text, "s": source or "", "g": target, "m": model},
                               sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def get(self, text: str, source: str | None, target: str, model: str) -> str | None:
        if not self._enabled:
            return None
        p = self._dir / f"{self._hash(text, source, target, model)}.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["translated"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, text: str, source: str | None, target: str, model: str, translated: str) -> None:
        if not self._enabled:
            return
        p = self._dir / f"{self._hash(text, source, target, model)}.json"
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps({"translated": translated}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
        except OSError as exc:
            log.debug("translate cache put failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
