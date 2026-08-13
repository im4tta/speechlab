"""Engine-agnostic translation service: registry, active model, batching, cache."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.exceptions import TextInvalid
from ..core.translate import TranslateRequest, TranslateResult, Translator
from ..core.translate.lang_codes import normalize_source

log = logging.getLogger(__name__)


class TranslateService:
    def __init__(self, translators: dict[str, Translator], default: str,
                 gate=None, cache=None, active_marker: Path | None = None) -> None:
        self._translators = translators
        self._cache = cache
        self._gate = gate
        self._marker = active_marker
        self._active = self._restore_active(default)

    def _restore_active(self, default: str) -> str:
        if self._marker and self._marker.is_file():
            try:
                name = self._marker.read_text(encoding="utf-8").strip()
                if name in self._translators:
                    return name
            except OSError:
                pass
        return default

    @property
    def active_name(self) -> str:
        return self._active

    def set_active(self, name: str) -> None:
        if name not in self._translators:
            raise TextInvalid(f"unknown translation model: {name}")
        if name == self._active:
            return
        prev = self._translators.get(self._active)
        if prev is not None:
            try:
                prev.unload()
            except Exception:  # noqa: BLE001
                log.exception("failed to unload translator %s", self._active)
        self._active = name
        if self._marker:
            try:
                self._marker.write_text(name, encoding="utf-8")
            except OSError:
                pass

    def get(self, name: str) -> Translator:
        return self._translators[name]

    def translate(self, texts: list[str], source_lang: str | None,
                  target_lang: str, model: str | None = None) -> TranslateResult:
        name = model or self._active
        if name not in self._translators:
            raise TextInvalid(f"unknown translation model: {name}")
        tr = self._translators[name]
        src = normalize_source(source_lang)
        if tr.needs_source_lang() and not src:
            raise TextInvalid("this translation model needs a source language; set it or auto-detect first")

        # Serve as much as possible from cache; translate only the misses.
        cached: list[str | None] = [None] * len(texts)
        if self._cache is not None and self._cache.enabled:
            for i, t in enumerate(texts):
                cached[i] = self._cache.get(t, src, target_lang, name)
        misses = [i for i, c in enumerate(cached) if c is None]

        out = list(cached)
        if misses:
            req = TranslateRequest(texts=[texts[i] for i in misses],
                                   source_lang=src, target_lang=target_lang)
            res = self._run(tr, req)
            for j, i in enumerate(misses):
                out[i] = res.texts[j]
                if self._cache is not None and self._cache.enabled:
                    self._cache.put(texts[i], src, target_lang, name, res.texts[j])
        return TranslateResult(texts=[o or "" for o in out],
                               source_lang=src or "auto", target_lang=target_lang, inference_ms=0)

    def _run(self, tr: Translator, req: TranslateRequest) -> TranslateResult:
        if self._gate is not None:
            with self._gate.exclusive():
                return tr.translate(req)
        return tr.translate(req)

    def status(self) -> dict:
        models = []
        for name, tr in self._translators.items():
            models.append({
                "name": name, "display_name": tr.display_name, "description": tr.description,
                "license": tr.license, "model_url": tr.model_url,
                "loaded": tr.is_loaded(), "downloaded": tr.downloaded(),
                "languages": tr.languages(), "needs_source_lang": tr.needs_source_lang(),
                "on_demand": tr.on_demand(),
            })
        return {"active": self._active, "models": models}
