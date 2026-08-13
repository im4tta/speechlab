"""On-disk cache for dubbed audio: one WAV + JSON sidecar per hash.

Deliberately not JoinCache: JoinCache stores its entries under downloads/ but
delegates reads to SynthCache.get, which looks for the meta JSON in the cache
root — so a freshly-put join entry doesn't round-trip within the same process.
DubCache is a simple, reliable store (like AsrCache) that round-trips.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)


class DubCache:
    def __init__(self, cache_dir: Path | str, enabled: bool = True) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dir(self) -> Path:
        return self._dir

    def _wav(self, content_hash: str) -> Path:
        return self._dir / f"{content_hash}.wav"

    def get(self, content_hash: str):
        """Return an entry with `.wav_path`/`.sample_rate`/`.duration_sec`/
        `.inference_ms`, or None. Shape matches what DubService reads."""
        if not self._enabled:
            return None
        wav = self._wav(content_hash)
        meta = wav.with_suffix(".json")
        if not wav.is_file() or not meta.is_file():
            return None
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return SimpleNamespace(
            wav_path=wav, sample_rate=int(d["sample_rate"]),
            duration_sec=float(d["duration_sec"]), inference_ms=int(d["inference_ms"]),
        )

    def put(self, join_hash: str, wav_bytes: bytes, sample_rate: int,
            duration_sec: float, inference_ms: int, segment_hashes=None) -> None:
        """Write atomically (temp + replace) so a reader never sees a partial WAV.
        `join_hash`/`segment_hashes` names kept to match the DubService call site."""
        if not self._enabled:
            return
        wav = self._wav(join_hash)
        meta = wav.with_suffix(".json")
        try:
            self._atomic_write(wav, wav_bytes)
            self._atomic_write(meta, json.dumps({
                "sample_rate": sample_rate, "duration_sec": duration_sec,
                "inference_ms": inference_ms,
            }).encode("utf-8"))
        except OSError as exc:
            log.debug("dub cache put failed: %s", exc)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
