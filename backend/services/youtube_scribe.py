"""YouTube transcript extraction + ASR fallback — ported from the user's
KhmerScribe tool (km-yt-scribe-clean: app.py's fetch/extract_id/download_audio).

Flow:
  1. Extract the video id from a URL or bare id.
  2. Try YouTube captions first (Khmer → English) via youtube-transcript-api,
     applying the Khmer spell-correction + word-segmentation post-processing.
  3. If no captions exist, download the audio with yt-dlp and transcribe it
     with the studio's in-process ASR (WhisperEngine + Khmer post-processing).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_YOUTUBE_RE = re.compile(r"^/(?:embed|v|shorts|live)/([A-Za-z0-9_-]{11})")
_KH_CODES = ["km", "kh", "khm"]

#: Cache dir for downloaded video audio (survives across requests).
_YOUTUBE_AUDIO_CACHE = Path(tempfile.gettempdir()) / "speechlab_yt_audio"


def thumbnail_url(video_id: str) -> str:
    """YouTube thumbnail URL (predictable from the video id)."""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _audio_cache_path(video_id: str) -> Path:
    _YOUTUBE_AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    return _YOUTUBE_AUDIO_CACHE / f"{video_id}.wav"


def download_audio(video_id: str) -> Path:
    """Download (or return the cached) WAV audio for a video id."""
    cached = _audio_cache_path(video_id)
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    tmp_dir = tempfile.mkdtemp(prefix="speechlab_yt_dl_")
    out_tmpl = os.path.join(tmp_dir, f"{video_id}.%(ext)s")
    subprocess.run(
        [
            "yt-dlp", "-f", "bestaudio/best", "-o", out_tmpl,
            "--extract-audio", "--audio-format", "wav",
            "--no-playlist", "-q",
            video_url(video_id),
        ],
        capture_output=True, timeout=600,
    )
    try:
        for name in os.listdir(tmp_dir):
            if name.endswith(".wav"):
                src = os.path.join(tmp_dir, name)
                shutil.move(src, cached)
                return cached
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError("yt-dlp produced no audio file")

_YOUTUBE_TRANSCRIPT_ERRORS: tuple = ()


def extract_video_id(raw: str) -> str:
    raw = raw.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    parsed = urlparse(raw)
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        path_id = parsed.path.lstrip("/").split("/")[0].split("?")[0]
        if path_id:
            return path_id
    if "youtube" in parsed.netloc:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if qs.get("v") and qs["v"][0]:
                return qs["v"][0]
        m = _YOUTUBE_RE.match(parsed.path)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract a YouTube video id from: {raw!r}")


def _fetch_captions(video_id: str, languages: list[str] | None = None) -> dict[str, Any] | None:
    """Return caption segments or None when no captions are available."""
    import re as _re

    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (  # type: ignore[attr-defined]
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        VideoUnplayable,
        YouTubeTranscriptApiException,
    )

    if languages is None:
        languages = _KH_CODES + ["en"]
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, VideoUnplayable, YouTubeTranscriptApiException):
        return None

    segments = []
    full_parts = []
    for snippet in transcript.snippets:
        text = _re.sub(r"\s+", " ", snippet.text).strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "start": float(snippet.start),
                "end": float(snippet.start) + float(snippet.duration),
            }
        )
        full_parts.append(text)
    return {
        "video_id": video_id,
        "language": transcript.language_code,
        "language_name": transcript.language,
        "is_generated": bool(transcript.is_generated),
        "segments": segments,
        "text": " ".join(full_parts),
        "source": "captions",
    }


def transcribe_youtube(
    url_or_id: str,
    *,
    language: str | None,
    timestamps: bool,
    asr_service,
    force_asr: bool = False,
) -> dict[str, Any]:
    """Get a transcript for a YouTube video: captions first, then ASR.

    `source`/`is_generated`/`kind` tell where the transcript came from:
      * kind="captions"  -> manual YouTube captions
      * kind="auto_cc"   -> YouTube auto-generated captions
      * kind="asr"       -> our Whisper ASR fallback (audio downloaded + transcribed)

    `force_asr=True` skips captions entirely and always transcribes the audio.
    """
    video_id = extract_video_id(url_or_id)

    captions = None
    if not force_asr:
        try:
            captions = _fetch_captions(video_id)
        except Exception as exc:  # noqa: BLE001 — captions are best-effort
            captions = None

    if captions and captions["segments"]:
        # Khmer post-processing (spell correction + segmentation) on captions.
        try:
            from .khmer_postprocess import correct_segments, segment_transcript_segments

            captions["segments"] = segment_transcript_segments(
                correct_segments(captions["segments"])
            )
        except Exception:  # noqa: BLE001
            pass
        duration = captions["segments"][-1]["end"]
        is_generated = bool(captions.get("is_generated"))
        return {
            **captions,
            "kind": "auto_cc" if is_generated else "captions",
            "duration_sec": duration,
            "segments": _to_asr_segments(captions["segments"]),
            "inference_ms": 0,
            "thumbnail_url": thumbnail_url(video_id),
            "audio_url": f"/api/youtube/audio/{video_id}",
            "video_url": video_url(video_id),
        }

    # No captions — download audio and run the in-process ASR.
    wav_path = str(download_audio(video_id))
    result = asr_service.transcribe_file(wav_path, language=language, timestamps=timestamps)
    return {
        "video_id": video_id,
        "language": result.language,
        "text": result.text,
        "duration_sec": result.duration_sec,
        "inference_ms": result.inference_ms,
        "segments": result.segments,
        "source": "asr",
        "is_generated": True,
        "kind": "asr",
        "thumbnail_url": thumbnail_url(video_id),
        "audio_url": f"/api/youtube/audio/{video_id}",
        "video_url": video_url(video_id),
    }


def _to_asr_segments(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        out.append(
            {
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "text": s.get("text", ""),
                **({"segmented": s.get("segmented")} if s.get("segmented") else {}),
            }
        )
    return out
