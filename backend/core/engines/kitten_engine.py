"""Kitten TTS Mini engine — ultra-light Apache-licensed TTS from KittenML.

See https://huggingface.co/KittenML/kitten-tts-mini-0.8 and
https://github.com/KittenML/KittenTTS.

Kitten TTS Mini is an ~80M-parameter StyleTTS2 model that runs on **ONNX
Runtime** (not torch/transformers), so it loads IN-PROCESS in the main venv
like Kokoro — no isolated worker. ~79 MB of weights, 24 kHz, 8 built-in
English voices, **no voice cloning**. Its whole appeal is running on low-end
hardware with no GPU.

Two quirks, both handled here:
  * The model exposes raw voice ids (`expr-voice-2-f` …) but ships friendly
    aliases in its `config.json` (`Bella`, `Jasper`, …). We surface the
    aliases as the display name and keep the raw id for the model call.
  * `generate()` with an unknown voice **silently degrades** rather than
    raising, so `synthesize()` validates the voice against the known set.

The `kittentts` import is lazy so the backend still boots when it isn't
installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from . import Engine, EngineResult, EngineSynthRequest, wrap_pcm_as_wav

log = logging.getLogger(__name__)

MODEL_ID = "KittenML/kitten-tts-mini-0.8"
_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class _KittenVoiceSpec:
    id: str      # what the model wants, e.g. "expr-voice-2-f"
    name: str    # friendly alias from the model's config.json
    gender: str  # "man" | "woman"


# The 8 voices, from the repo's config.json `voice_aliases` map. `-f` = woman,
# `-m` = man. These aliases are the model's own, not invented here.
_KITTEN_VOICES: tuple[_KittenVoiceSpec, ...] = (
    _KittenVoiceSpec("expr-voice-2-f", "Bella",  "woman"),
    _KittenVoiceSpec("expr-voice-2-m", "Jasper", "man"),
    _KittenVoiceSpec("expr-voice-3-f", "Luna",   "woman"),
    _KittenVoiceSpec("expr-voice-3-m", "Bruno",  "man"),
    _KittenVoiceSpec("expr-voice-4-f", "Rosie",  "woman"),
    _KittenVoiceSpec("expr-voice-4-m", "Hugo",   "man"),
    _KittenVoiceSpec("expr-voice-5-f", "Kiki",   "woman"),
    _KittenVoiceSpec("expr-voice-5-m", "Leo",    "man"),
)


def _voice_spec(voice_id: str) -> _KittenVoiceSpec | None:
    for v in _KITTEN_VOICES:
        if v.id == voice_id:
            return v
    return None


class KittenEngine(Engine):
    name = "kitten"
    display_name = "Kitten TTS Mini"
    license = "Apache-2.0"
    model_url = "https://huggingface.co/KittenML/kitten-tts-mini-0.8"
    description = "KittenML's ~80M ONNX TTS. ~79 MB, 24 kHz, 8 voices, runs on CPU."

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._available: bool | None = None  # cached import check

    # -- lifecycle
    def load(self) -> None:
        self._ensure_kittentts()
        if self._model is None:
            from kittentts import KittenTTS

            t0 = time.perf_counter()
            # Constructing downloads config + onnx + voices from the HF repo
            # (cached; pre-fetched by the model downloader). Respects HF_HOME,
            # which app.py sets before any engine import.
            self._model = KittenTTS(self._model_id)
            log.info("Kitten TTS loaded in %.1fs", time.perf_counter() - t0)

    def unload(self) -> None:
        self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def downloaded(self) -> bool:
        from ..model_cache import model_downloaded

        return model_downloaded(self._model_id)

    def engine_info(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "device": "cpu",  # ONNX Runtime CPU provider
            "dtype": "float32",
            "attn_implementation": "onnx",
        }

    # -- capabilities
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    def max_speakers(self) -> int:
        return 1

    def supports_voice_cloning(self) -> bool:
        return False

    def default_cfg_scale(self) -> float | None:
        return None

    def available_voices(self) -> list:
        from ...services.voices import VoiceInfo

        return [
            VoiceInfo(
                id=v.id,
                name=v.name,
                gender=v.gender,
                language="en",
                source="builtin",
                size_bytes=None,
                duration_sec=None,
                sample_rate=_SAMPLE_RATE,
            )
            for v in _KITTEN_VOICES
        ]

    def languages(self) -> list[dict[str, str]]:
        return [{"code": "en", "label": "English"}]

    # -- synthesis
    def synthesize(self, req: EngineSynthRequest) -> EngineResult:
        import numpy as np

        if not req.voice_id:
            raise ValueError("voice_id is required for Kitten TTS")
        spec = _voice_spec(req.voice_id)
        if spec is None:
            # generate() would silently produce degraded audio for an unknown
            # voice, so fail loudly instead.
            raise ValueError(f"unknown Kitten TTS voice: {req.voice_id}")

        text = (req.text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")

        if self._model is None:
            self.load()

        t0 = time.perf_counter()
        audio = self._model.generate(text, voice=spec.id)
        inference_ms = int((time.perf_counter() - t0) * 1000)

        wav = np.asarray(audio, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError("Kitten TTS produced no audio")
        return EngineResult(
            wav_bytes=wrap_pcm_as_wav(wav, _SAMPLE_RATE),
            sample_rate=_SAMPLE_RATE,
            duration_sec=float(wav.size) / float(_SAMPLE_RATE),
            inference_ms=inference_ms,
        )

    # -- internals
    def _ensure_kittentts(self) -> None:
        if self._available is False:
            raise RuntimeError(
                "The 'kittentts' package is not installed. "
                "Run `pip install -r backend/requirements.txt`."
            )
        if self._available is None:
            try:
                import kittentts  # noqa: F401

                self._available = True
            except ImportError as exc:
                self._available = False
                raise RuntimeError(
                    "The 'kittentts' package is not installed. "
                    "Run `pip install -r backend/requirements.txt`. "
                    f"(import error: {exc})"
                ) from exc
