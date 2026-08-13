"""Argos Translate — offline, in-process translator with on-demand language packs."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from . import TranslateRequest, TranslateResult, Translator
from .argos_langs import ARGOS_LANGS, required_packages
from .lang_codes import normalize_source

log = logging.getLogger(__name__)


class ArgosTranslator(Translator):
    name = "argos"
    display_name = "Argos Translate (offline)"
    description = "Open-source offline translator (the engine LibreTranslate uses). Installs language packs on demand."
    license = "MIT"
    model_url = "https://github.com/argosopentech/argos-translate"

    def __init__(self, packages_dir: str | Path, device_request: str = "auto") -> None:
        self._packages_dir = Path(packages_dir)
        self._device_request = device_request
        self._loaded = False

    def needs_source_lang(self) -> bool:
        return True

    def on_demand(self) -> bool:
        return True

    def languages(self) -> list[dict[str, str]]:
        return [{"code": c, "label": n} for c, n in ARGOS_LANGS]

    def downloaded(self) -> bool:
        try:
            import argostranslate.translate  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        # Point Argos's package dir at our project-local path BEFORE importing its
        # settings module (which reads the env var at import time).
        self._packages_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ARGOS_PACKAGES_DIR"] = str(self._packages_dir)
        os.environ.setdefault("ARGOS_DEVICE_TYPE",
                              "cuda" if self._device_request in ("auto", "cuda") else "cpu")
        import argostranslate.translate  # noqa: F401  (import check)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _ensure_packages(self, source: str, target: str) -> None:
        import argostranslate.package as pkg
        from ..exceptions import TextInvalid
        needed = required_packages(source, target)
        installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
        missing = [pair for pair in needed if pair not in installed]
        if not missing:
            return
        log.info("Argos: installing language packs %s", missing)
        pkg.update_package_index()
        available = {(a.from_code, a.to_code): a for a in pkg.get_available_packages()}
        for pair in missing:
            ap = available.get(pair)
            if ap is None:
                raise TextInvalid(
                    f"Argos has no language pack for {pair[0]}→{pair[1]}; that language isn't supported.")
            try:
                ap.install()  # download + install
            except Exception as exc:  # noqa: BLE001
                raise TextInvalid(
                    f"Couldn't fetch the Argos language pack {pair[0]}→{pair[1]}: {exc}") from exc

    def translate(self, req: TranslateRequest) -> TranslateResult:
        self.load()
        from ..exceptions import TextInvalid
        src = normalize_source(req.source_lang)
        if not src:
            raise TextInvalid("Argos needs a source language; set it or auto-detect first.")
        codes = {c for c, _ in ARGOS_LANGS}
        for code in (src, req.target_lang):
            if code != "en" and code not in codes:
                raise TextInvalid(f"Argos doesn't support the language '{code}'.")
        self._ensure_packages(src, req.target_lang)
        import argostranslate.translate as tr
        t0 = time.perf_counter()
        out = [tr.translate(text, src, req.target_lang) for text in req.texts]
        ms = int((time.perf_counter() - t0) * 1000)
        return TranslateResult(texts=out, source_lang=src, target_lang=req.target_lang, inference_ms=ms)
