"""Speech-to-text model catalog.

All entries are standard `whisper`-architecture checkpoints, so the in-process
`WhisperEngine` (AutoModelForSpeechSeq2Seq + the transformers ASR pipeline) can
load any of them with zero code changes. The Khmer-tuned ones are fine-tunes of
Whisper on Khmer audio and are dramatically better for Khmer than the stock
OpenAI checkpoints (Whisper is weak on low-resource Khmer and its tokenizer
inserts spaces where Khmer has none).

Order matters: the UI lists them in this order, with `best_for: ["khmer"]`
flagging the ones worth picking for Khmer input.
"""

from __future__ import annotations

from typing import Any

ASR_MODELS: dict[str, dict[str, Any]] = {
    "openai/whisper-large-v3-turbo": {
        "label": "Whisper large-v3-turbo",
        "size": "~1.6 GB",
        "description": "OpenAI's default multilingual model (99 languages). Fast, but weak on Khmer.",
        "model_url": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "license": "MIT",
        "best_for": ["all"],
    },
    "seanghay/whisper-small-khmer-v2": {
        "label": "Whisper small · Khmer v2",
        "size": "~460 MB",
        "description": "seanghay's Khmer fine-tune — the best speed/accuracy balance for Khmer.",
        "model_url": "https://huggingface.co/seanghay/whisper-small-khmer-v2",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "1morecupofhottea/whisper-turbo-khmer-v9": {
        "label": "Whisper turbo · Khmer v9",
        "size": "~1.6 GB",
        "description": "Popular Khmer fine-tune of whisper-turbo (very fast, strong Khmer).",
        "model_url": "https://huggingface.co/1morecupofhottea/whisper-turbo-khmer-v9",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "1morecupofhottea/whisper-turbo-khmer-v7-stable": {
        "label": "Whisper turbo · Khmer v7 (stable)",
        "size": "~1.6 GB",
        "description": "Stable release of the Khmer turbo fine-tune by 1morecupofhottea.",
        "model_url": "https://huggingface.co/1morecupofhottea/whisper-turbo-khmer-v7-stable",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "seanghay/whisper-medium-khmer": {
        "label": "Whisper medium · Khmer",
        "size": "~1.5 GB",
        "description": "seanghay's Khmer fine-tune of whisper-medium — higher Khmer quality, slower.",
        "model_url": "https://huggingface.co/seanghay/whisper-medium-khmer",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "ken0997/old-whisper-khmer-medium": {
        "label": "Whisper medium · Khmer (ken0997)",
        "size": "~1.5 GB",
        "description": "Another Khmer whisper-medium fine-tune by ken0997.",
        "model_url": "https://huggingface.co/ken0997/old-whisper-khmer-medium",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "ksoky/whisper-large-khmer-asr": {
        "label": "Whisper large · Khmer ASR",
        "size": "~3.0 GB",
        "description": "Khmer fine-tune of whisper-large by ksoky — top quality, heavy + slow.",
        "model_url": "https://huggingface.co/ksoky/whisper-large-khmer-asr",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "steja/whisper-large-khmer": {
        "label": "Whisper large · Khmer (steja)",
        "size": "~3.0 GB",
        "description": "Khmer whisper-large fine-tune by steja.",
        "model_url": "https://huggingface.co/steja/whisper-large-khmer",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "steja/whisper-small-khmer": {
        "label": "Whisper small · Khmer (steja)",
        "size": "~460 MB",
        "description": "Light Khmer whisper-small fine-tune by steja.",
        "model_url": "https://huggingface.co/steja/whisper-small-khmer",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "seanghay/whisper-small-khmer-unspaced": {
        "label": "Whisper small · Khmer (unspaced)",
        "size": "~460 MB",
        "description": "Khmer fine-tune that outputs raw Khmer script with no inserted word spaces (Khmer has none).",
        "model_url": "https://huggingface.co/seanghay/whisper-small-khmer-unspaced",
        "license": "MIT",
        "best_for": ["khmer"],
    },
    "openai/whisper-large-v3": {
        "label": "Whisper large-v3",
        "size": "~3.0 GB",
        "description": "Highest-quality OpenAI multilingual model (heavier; Khmer still limited).",
        "model_url": "https://huggingface.co/openai/whisper-large-v3",
        "license": "MIT",
        "best_for": ["all"],
    },
}
