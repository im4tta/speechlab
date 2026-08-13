"""/api/translate — machine translation for Dub + Transcribe."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.exceptions import BackendError
from .deps import get_translate_service
from .schemas import TranslateActivateBody, TranslateRequestBody

router = APIRouter(tags=["translate"])


@router.get("/api/translate/status")
def translate_status(svc=Depends(get_translate_service)) -> dict:
    return svc.status()


@router.post("/api/translate/activate")
def translate_activate(body: TranslateActivateBody, svc=Depends(get_translate_service)) -> dict:
    svc.set_active(body.name)
    return svc.status()


@router.post("/api/translate")
def translate(body: TranslateRequestBody, svc=Depends(get_translate_service)) -> dict:
    if body.segments is not None:
        texts = [s.text for s in body.segments]
    elif body.texts is not None:
        texts = body.texts
    else:
        raise BackendError("provide either 'segments' or 'texts'", http_status=422, code="invalid")
    res = svc.translate(texts=texts, source_lang=body.source_lang,
                        target_lang=body.target_lang, model=body.model)
    out: dict = {"source_lang": res.source_lang, "target_lang": res.target_lang,
                 "model": body.model or svc.active_name}
    if body.segments is not None:
        out["segments"] = [{"start": s.start, "end": s.end, "text": t}
                           for s, t in zip(body.segments, res.texts)]
    else:
        out["texts"] = res.texts
    return out
