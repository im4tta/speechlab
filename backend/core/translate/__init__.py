"""Machine-translation abstraction.

Text -> text, so deliberately separate from core.engines.Engine (text -> audio)
and core.asr.AsrEngine (audio -> text). Constructed at app start, loaded lazily
on first use, and never registered in EngineManager.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class TranslateRequest:
    texts: list[str]
    source_lang: str | None   # None/"auto" => model auto-detects (MADLAD); M2M requires it
    target_lang: str


@dataclass
class TranslateResult:
    texts: list[str]          # same order & length as the request
    source_lang: str
    target_lang: str
    inference_ms: int


class Translator(abc.ABC):
    name: str = "translator"
    display_name: str = "Translator"
    description: str = ""
    license: str = ""
    model_url: str = ""

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def unload(self) -> None: ...

    @abc.abstractmethod
    def is_loaded(self) -> bool: ...

    @abc.abstractmethod
    def translate(self, req: TranslateRequest) -> TranslateResult: ...

    def downloaded(self) -> bool:
        return True

    def languages(self) -> list[dict[str, str]]:
        return []

    def needs_source_lang(self) -> bool:
        return False

    def on_demand(self) -> bool:
        """True when the model fetches assets per-use (Argos language packs)
        rather than as one downloadable weight — the UI hides the Download
        button and notes on-demand installation."""
        return False


__all__ = ["Translator", "TranslateRequest", "TranslateResult"]
