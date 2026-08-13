"""M2M-100 418M translator (facebook/m2m100_418M, MIT). Any-to-any, needs source."""

from __future__ import annotations

import logging
import time

from . import TranslateRequest, TranslateResult, Translator
from .lang_codes import M2M100_LANGS, normalize_source

log = logging.getLogger(__name__)

_MODEL_ID = "facebook/m2m100_418M"


class M2M100Translator(Translator):
    """M2M-100 in-process translator. Identity is parameterizable so the 418M
    and 1.2B checkpoints (same architecture, same 100 languages) share this code."""

    license = "MIT"

    def __init__(self, model_id: str = _MODEL_ID, device_request: str = "auto", *,
                 name: str = "m2m100", display_name: str = "M2M-100 (418M)",
                 description: str = "Meta's 100-language translator. Small, fast, MIT-licensed.",
                 model_url: str = "https://huggingface.co/facebook/m2m100_418M") -> None:
        self.name = name
        self.display_name = display_name
        self.description = description
        self.model_url = model_url
        self._model_id = model_id
        self._device_request = device_request
        self._model = None
        self._tok = None
        self._device = "cpu"

    def is_loaded(self) -> bool:
        return self._model is not None

    def downloaded(self) -> bool:
        from ..model_cache import model_downloaded
        return model_downloaded(self._model_id)

    def languages(self) -> list[dict[str, str]]:
        return [{"code": c, "label": n} for c, n in M2M100_LANGS]

    def needs_source_lang(self) -> bool:
        return True

    @staticmethod
    def lang_id_ok(code: str) -> bool:
        return any(code == c for c, _ in M2M100_LANGS)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        self._tok = M2M100Tokenizer.from_pretrained(self._model_id)
        self._model = M2M100ForConditionalGeneration.from_pretrained(self._model_id)
        self._device = "cuda" if (self._device_request in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self._model.to(self._device).eval()
        log.info("M2M-100 loaded on %s", self._device)

    def unload(self) -> None:
        self._model = None
        self._tok = None

    def translate(self, req: TranslateRequest) -> TranslateResult:
        import torch
        self.load()
        src = normalize_source(req.source_lang)
        if not src:
            from ..exceptions import TextInvalid
            raise TextInvalid("M2M-100 needs a source language; set it or auto-detect first")
        t0 = time.perf_counter()
        self._tok.src_lang = src
        out: list[str] = []
        bos = self._tok.get_lang_id(req.target_lang)
        for text in req.texts:
            enc = self._tok(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                gen = self._model.generate(**enc, forced_bos_token_id=bos, max_length=1024)
            out.append(self._tok.batch_decode(gen, skip_special_tokens=True)[0])
        ms = int((time.perf_counter() - t0) * 1000)
        return TranslateResult(texts=out, source_lang=src, target_lang=req.target_lang, inference_ms=ms)
