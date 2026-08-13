"""POST /api/youtube/transcript — YouTube scribe (captions + ASR fallback).

Ported from the user's KhmerScribe tool. Accepts a YouTube URL or video id,
returns caption segments (Khmer → English) or, when the video has no captions,
downloads the audio and transcribes it with the in-process ASR engine
(Whisper + Khmer spell-correction/segmentation).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..services.youtube_scribe import download_audio, transcribe_youtube
from .deps import get_asr_service, get_youtube_history

log = logging.getLogger(__name__)
router = APIRouter(tags=["youtube"])


class YoutubeTranscriptBody(BaseModel):
    url: str = Body(..., description="YouTube URL or bare video id")
    language: str | None = Body(None, description="Whisper language code (e.g. 'km')")
    timestamps: bool = Body(False, description="Return per-segment timestamps (ASR path)")
    force_asr: bool = Body(False, description="Skip captions and always transcribe audio with Whisper")


class YoutubeCorrectBody(BaseModel):
    video_id: str = Body(..., description="YouTube video id")
    text: str = Body(..., description="Corrected transcript text")
    language: str = Body("km", description="Language code")
    segments: list[dict] = Body(default_factory=list, description="Optional corrected segments")


@router.post("/api/youtube/transcript")
def youtube_transcript(
    body: YoutubeTranscriptBody,
    svc=Depends(get_asr_service),
    history=Depends(get_youtube_history),
) -> dict:
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="provide a YouTube URL or video id")
    try:
        result = transcribe_youtube(
            url,
            language=body.language,
            timestamps=body.timestamps,
            asr_service=svc,
            force_asr=body.force_asr,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("YouTube scribe failed")
        raise HTTPException(status_code=500, detail=f"YouTube scribe failed: {exc}") from exc
    try:
        history.save(result)
    except Exception:  # noqa: BLE001
        pass
    return result


@router.post("/api/youtube/transcript/correct")
def youtube_transcript_correct(
    body: YoutubeCorrectBody,
    history=Depends(get_youtube_history),
) -> dict:
    """Store a manually-corrected transcript for a video in its history entry.

    Lets users replace an unreliable auto-generated-CC / ASR transcript with the
    verified text (e.g. the real song lyrics). The ORIGINAL transcript from
    YouTube/ASR is preserved in `entry.original` so it is never lost.
    """
    video_id = body.video_id.strip()
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="invalid video id")

    existing = history.get(video_id)
    base = existing or {
        "video_id": video_id,
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "audio_url": f"/api/youtube/audio/{video_id}",
    }

    # Preserve the FIRST original we saw (don't re-capture our own correction).
    original = base.get("original")
    if original is None and existing:
        orig_text = (existing.get("text") or "").strip()
        if orig_text:
            original = {
                "text": orig_text,
                "segments": existing.get("segments", []),
                "source": existing.get("source"),
                "kind": existing.get("kind"),
                "is_generated": existing.get("is_generated", False),
                "created_at": existing.get("created_at"),
            }

    entry = {
        **base,
        "video_id": video_id,
        "text": body.text.strip(),
        "language": body.language,
        "source": "captions",
        "is_generated": False,
        "kind": "manual",
        "segments": body.segments or [],
        "language_name": existing.get("language_name"),
        "original": original,
    }
    history.save(entry)
    return entry


@router.get("/api/youtube/history")
def youtube_history(history=Depends(get_youtube_history)) -> dict:
    """Recent YouTube transcripts (most recent first)."""
    return {"entries": history.list()}


@router.delete("/api/youtube/history/{video_id}")
def youtube_history_delete(video_id: str, history=Depends(get_youtube_history)) -> dict:
    if not history.delete(video_id):
        raise HTTPException(status_code=404, detail=f"no history entry for {video_id}")
    return {"deleted": video_id}


@router.delete("/api/youtube/history")
def youtube_history_clear(history=Depends(get_youtube_history)) -> dict:
    return {"removed": history.clear()}


@router.get("/api/youtube/audio/{video_id}")
def youtube_audio(video_id: str) -> FileResponse:
    """Download (cached) WAV audio for a video, for playback / download."""
    try:
        path = download_audio(video_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("YouTube audio download failed")
        raise HTTPException(status_code=500, detail=f"audio download failed: {exc}") from exc
    return FileResponse(path, media_type="audio/wav", filename=f"{video_id}.wav")
