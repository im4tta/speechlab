"""POST /api/asr/transcribe + GET /api/asr/status + POST /api/asr/model.

Weight downloads for the DEFAULT whisper model deliberately have no route here:
`api/engines.py`'s `/{name}/download` validates against `DOWNLOADABLE` rather
than the engine registry, so `/api/engines/whisper/download` already drives the
shared ModelDownloader. This module adds a per-model switcher for the Khmer
ASR variants, which reuses the same downloader with a `repo_id` override.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..core.asr.models import ASR_MODELS
from ..services.synth_cache import SynthCache
from .deps import get_asr_service, get_model_downloader, get_synth_cache
from .schemas import AsrStatusResponse, AsrTranscribeResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["asr"])


class AsrModelBody(BaseModel):
    model_id: str = Body(..., description="ASR model id from the catalog")


@router.get("/api/asr/status", response_model=AsrStatusResponse)
def asr_status(svc=Depends(get_asr_service)) -> AsrStatusResponse:
    return AsrStatusResponse(**svc.status())


@router.post("/api/asr/model")
def activate_asr_model(
    body: AsrModelBody,
    svc=Depends(get_asr_service),
    downloader=Depends(get_model_downloader),
) -> dict:
    """Switch the active speech-to-text model (e.g. a Khmer fine-tune).

    Downloads the checkpoint first if it isn't cached; returns
    `{downloading: true}` so the UI can poll `/api/engines/whisper/download`
    for progress and re-activate when the weights land.
    """
    model_id = body.model_id.strip()
    meta = ASR_MODELS.get(model_id)
    if meta is None:
        raise HTTPException(status_code=400, detail=f"unknown ASR model: {model_id}")

    engine = svc.engine
    if getattr(engine, "_model_id", None) != model_id:
        engine.set_model(
            model_id,
            label=meta["label"],
            description=meta["description"],
            model_url=meta["model_url"],
            license=meta.get("license"),
        )

    if not engine.downloaded():
        try:
            st = downloader.start("whisper", repo_id=model_id)
        except ValueError as exc:
            # Already downloading the same model — return the live snapshot.
            if "already in progress" in str(exc):
                st = downloader.status()
            else:
                raise HTTPException(status_code=400, detail=str(exc))
        return {
            "model_id": model_id,
            "label": meta["label"],
            "downloading": True,
            "percent": st.get("percent"),
        }
    # Remember the choice so it survives restarts (like the active TTS engine).
    try:
        from ..core.asr.whisper_engine import persist_model

        persist_model(model_id)
    except Exception:  # noqa: BLE001
        pass
    return {"model_id": model_id, "label": meta["label"], "downloading": False, "active": True}


@router.post("/api/asr/transcribe", response_model=AsrTranscribeResponse)
def transcribe(
    file: UploadFile | None = File(None),
    cache_hash: str | None = Form(None),
    language: str | None = Form(None),
    timestamps: bool = Form(False),
    svc=Depends(get_asr_service),
    cache: SynthCache = Depends(get_synth_cache),
) -> AsrTranscribeResponse:
    """Transcribe an uploaded file, or audio already in the synthesis cache.

    Exactly one of `file` / `cache_hash` must be provided. The `cache_hash`
    form is how subtitles are produced for generated audio without a re-upload.
    """
    has_file = file is not None and bool(file.filename)
    has_hash = bool(cache_hash)
    if has_file == has_hash:
        raise HTTPException(
            status_code=422,
            detail="provide exactly one of 'file' or 'cache_hash'",
        )

    if has_hash:
        entry = cache.get(cache_hash)  # type: ignore[arg-type]
        if entry is None or not entry.wav_path.is_file():
            raise HTTPException(status_code=404, detail=f"clip not found: {cache_hash}")
        result = svc.transcribe_file(
            str(entry.wav_path), language=language, timestamps=timestamps
        )
        return AsrTranscribeResponse(**result.__dict__)

    # Uploaded file: persist to a temp path with its original suffix so the
    # service's extension check and librosa's decoder both see the real format.
    suffix = Path(file.filename or "").suffix  # type: ignore[union-attr]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)  # type: ignore[union-attr]
    try:
        result = svc.transcribe_file(tmp_path, language=language, timestamps=timestamps)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    return AsrTranscribeResponse(**result.__dict__)
