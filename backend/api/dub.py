"""POST /api/dub — re-voice transcript segments in one chosen voice."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ..core.exceptions import BackendError
from ..services.dub import DubSegment
from .deps import get_dub_service
from .schemas import DubRequestBody

log = logging.getLogger(__name__)
router = APIRouter(tags=["dub"])


@router.post("/api/dub", responses={200: {"content": {"audio/wav": {}}}})
def dub(body: DubRequestBody, svc=Depends(get_dub_service)) -> Response:
    try:
        result = svc.dub(
            segments=[DubSegment(s.start, s.end, s.text) for s in body.segments],
            voice=body.voice,
            engine=body.engine,
            voice_mode=body.voice_mode, instruct=body.instruct,
            cfg_scale=body.cfg_scale, speed=body.speed, cfg_weight=body.cfg_weight,
            exaggeration=body.exaggeration, language_id=body.language_id,
            inference_steps=body.inference_steps, temperature=body.temperature,
            top_p=body.top_p, top_k=body.top_k, repetition_penalty=body.repetition_penalty,
            seed=body.seed, force_regenerate=body.force_regenerate,
            source_language=body.source_language,
            target_language=body.target_language,
            translator=body.translator,
        )
    except BackendError:
        raise
    return Response(
        content=result.wav_bytes, media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(result.sample_rate),
            "X-Audio-Duration-Sec": f"{result.duration_sec:.3f}",
            "X-Cache": "hit" if result.cache_hit else "miss",
            "X-Cache-Hash": result.cache_hash,
            "Content-Disposition": f'attachment; filename="dub-{result.cache_hash[:8]}.wav"',
        },
    )
