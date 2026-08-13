"""FastAPI application entrypoint."""

from __future__ import annotations

# Configure the HuggingFace cache BEFORE importing anything that
# touches transformers / kokoro / huggingface_hub. Settings() reads
# models_dir from env / .env, so HF_HOME is set from the configured
# value before the engines load.
import os as _os
from pathlib import Path as _Path

_BACKEND_ROOT = _Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = _BACKEND_ROOT / "models"
_models_dir = _Path(
    _os.environ.get("MODELS_DIR", str(_DEFAULT_MODELS_DIR))
).expanduser().resolve()

from .core.hf_paths import configure_hf_cache as _configure_hf_cache
_configure_hf_cache(_models_dir)

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.asr import router as asr_router
from .api.dub import router as dub_router
from .api.translate import router as translate_router
from .api.youtube import router as youtube_router
from .api.cache import router as cache_router
from .api.download import router as download_router
from .api.engines import router as engines_router
from .api.health import router as health_router
from .api.stream import router as stream_router
from .api.synthesize import router as synthesize_router
from .api.system import router as system_router
from .api.voices import router as voices_router
from .config import Settings, get_settings
from .core.engine_manager import EngineManager
from .core.exceptions import BackendError
from .core.version import get_version
from .services.chatterbox_install import ChatterboxInstaller, EngineEnvInstaller
from .services.model_download import ModelDownloader
from .services.model_delete import ModelDeleter
from .services.engine_uninstall import EngineEnvUninstaller
from .services.join_cache import JoinCache
from .services.synth_cache import SynthCache
from .core.asr.whisper_engine import WhisperEngine
from .core.asr.whisper_engine import read_persisted_model as read_persisted_asr_model
from .core.translate.m2m100_translator import M2M100Translator
from .core.translate.argos_translator import ArgosTranslator
from .core.gpu_gate import GpuGate
from .services.asr_cache import AsrCache
from .services.synthesize import SynthService
from .services.transcribe import AsrService
from .services.voices import VoiceRegistry
from .services.youtube_history import YoutubeHistory

log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
    # phonemizer's espeak backend warns "words count mismatch …" on nearly every
    # Kokoro/Kitten synthesis — it's cosmetic (word-count alignment only matters
    # for per-word timestamps, which we don't use) and floods the log. Mute its
    # WARNINGs; real errors still surface.
    logging.getLogger("phonemizer").setLevel(logging.ERROR)


def _mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Serve the built frontend at / when a Vite build exists.

    No-op in dev mode (no dist). Must be called AFTER the API routers so
    the catch-all static mount never shadows /api/*.
    """
    if not (dist_dir / "index.html").is_file():
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")


def _warmup_active_engine(em: EngineManager) -> None:
    """Load the active engine; swallow + log any failure.

    Skips engines whose weights aren't downloaded yet so the user sees the
    Download button in the UI rather than a silent background download with
    no progress. Runs on a background thread so startup never blocks on it.
    """
    try:
        engine = em.active_engine
        if not engine.downloaded():
            log.info(
                "Skipping warm-up for %r: weights not in local cache. "
                "Use the Download button in the UI to pre-fetch them.",
                engine.name,
            )
            return
        em.ensure_active_loaded()
    except Exception:  # noqa: BLE001
        log.exception("Active engine failed to warm up; first use will retry.")


def _start_background_warmup(em: EngineManager) -> threading.Thread:
    """Start _warmup_active_engine on a daemon thread and return it."""
    t = threading.Thread(
        target=_warmup_active_engine,
        args=(em,),
        name="engine-warmup",
        daemon=True,
    )
    t.start()
    return t


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Re-configure HF cache from the resolved settings.models_dir so a
    # .env override (MODELS_DIR=...) or a test override actually wins.
    # Idempotent: safe to call twice.
    _configure_hf_cache(settings.models_dir)
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        em: EngineManager = app.state.engine_manager
        warmup = _start_background_warmup(em)
        try:
            yield
        finally:
            # Give an in-flight warm-up a moment to settle so we don't unload
            # mid-load; if it's hung, proceed anyway (daemon thread).
            warmup.join(timeout=2.0)
            for engine in em.list_engines():
                try:
                    engine.unload()
                except Exception:  # noqa: BLE001
                    log.exception("Engine unload failed for %s", engine.name)

    app = FastAPI(
        title="Multi-engine TTS API",
        version=get_version(),
        lifespan=lifespan,
    )

    # CORS — permissive for local dev. Tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- state singletons
    voice_registry = VoiceRegistry(
        voices_dir=settings.voices_dir,
        uploads_dir=settings.uploads_dir,
    )
    synth_cache = SynthCache(
        cache_dir=settings.cache_dir,
        enabled=settings.cache_enabled,
        max_entries=settings.cache_max_entries,
    )
    join_cache = JoinCache(synth_cache)

    engine_manager = EngineManager(
        default_engine=settings.default_engine,
        voices_dir=settings.voices_dir,
        uploads_dir=settings.uploads_dir,
        model_id=settings.model_id,
        device_request=settings.device if settings.device != "auto" else "cuda",
        max_text_chars=settings.max_text_chars,
        default_cfg_scale=settings.default_cfg_scale,
        kokoro_lang_code=settings.kokoro_lang_code,
        chatterbox_model_id=settings.chatterbox_model_id,
        chatterbox_default_language_id=settings.chatterbox_default_language_id,
        chatterbox_default_cfg_weight=settings.chatterbox_default_cfg_weight,
        chatterbox_default_exaggeration=settings.chatterbox_default_exaggeration,
        chatterbox_watermark=settings.chatterbox_watermark,
        omnivoice_model_id=settings.omnivoice_model_id,
        omnivoice_num_step=settings.omnivoice_num_step,
        voxcpm_model_id=settings.voxcpm_model_id,
        voxcpm_inference_timesteps=settings.voxcpm_inference_timesteps,
        voxcpm_model_path=settings.voxcpm_model_path,
        qwen_model_id=settings.qwen_model_id,
        nanovllm_model_id=settings.nanovllm_model_id,
        nanovllm_devices=settings.nanovllm_devices,
        nanovllm_inference_timesteps=settings.nanovllm_inference_timesteps,
        nanovllm_gpu_memory_utilization=settings.nanovllm_gpu_memory_utilization,
        nanovllm_max_num_seqs=settings.nanovllm_max_num_seqs,
        nanovllm_chunk_max_chars=settings.nanovllm_chunk_max_chars,
    )

    # Wire each engine's built-in voice catalog into the registry so
    # /api/voices returns a single merged list tagged with `engine`.
    for engine in engine_manager.list_engines():
        try:
            voices = engine.available_voices()
        except Exception:  # noqa: BLE001
            log.exception("Engine %s failed to list built-in voices", engine.name)
            voices = []
        if voices:
            voice_registry.register_engine_voices(engine.name, voices)

    # One gate for every GPU consumer: TTS synthesis and ASR transcription
    # serialize onto a single worker thread so they never contend for VRAM.
    gpu_gate = GpuGate(settings.synth_timeout_s)

    synth_service = SynthService(
        engine_manager=engine_manager,
        voice_registry=voice_registry,
        max_text_chars=settings.max_text_chars,
        synth_timeout_s=settings.synth_timeout_s,
        default_cfg_scale=settings.default_cfg_scale,
        cache=synth_cache,
        gate=gpu_gate,
    )

    # Speech-to-text. Constructed but NOT loaded (lazy on first transcription)
    # and deliberately absent from EngineManager: Whisper is audio->text, so it
    # must never appear in the TTS engine selector.
    asr_engine = WhisperEngine(
        model_id=read_persisted_asr_model(settings.asr_model_id),
        device_request=settings.device,
    )
    asr_service = AsrService(
        engine=asr_engine,
        gate=gpu_gate,
        cache=AsrCache(settings.cache_dir / "asr", enabled=settings.cache_enabled),
        max_upload_mb=settings.asr_max_upload_mb,
        max_duration_sec=settings.asr_max_duration_sec,
        timeout_s=settings.asr_timeout_s,
    )

    # Keep the ModelManager around for direct introspection (the
    # VibeVoiceEngine owns one). It's also exposed via /api/health
    # for the device/dtype fields.
    vibevoice_engine = engine_manager.get_engine("vibevoice")
    model_manager = getattr(vibevoice_engine, "_model_manager", None)

    app.state.settings = settings
    app.state.engine_manager = engine_manager
    app.state.model_manager = model_manager  # legacy field, may be None
    app.state.voice_registry = voice_registry
    app.state.synth_cache = synth_cache
    app.state.join_cache = join_cache
    # Machine translation (text -> text). In-process, main venv, GPU-gated, and
    # absent from EngineManager (like Whisper). Two selectable models.
    from .services.translate import TranslateService
    from .services.translate_cache import TranslateCache
    translate_service = TranslateService(
        translators={
            "m2m100": M2M100Translator(device_request=settings.device),
            "m2m100_large": M2M100Translator(
                model_id="facebook/m2m100_1.2B", device_request=settings.device,
                name="m2m100_large", display_name="M2M-100 (1.2B)",
                description="Meta's 100-language translator. Larger, higher quality (MIT).",
                model_url="https://huggingface.co/facebook/m2m100_1.2B",
            ),
            "argos": ArgosTranslator(
                packages_dir=settings.argos_packages_dir, device_request=settings.device),
        },
        default="m2m100",
        gate=gpu_gate,
        cache=TranslateCache(settings.cache_dir / "translate", enabled=settings.cache_enabled),
        active_marker=_BACKEND_ROOT / ".last_translator",
    )

    # Dubbing re-voices transcript segments via the synth service; its results
    # cache in their own dub/ dir (DubCache round-trips reliably in-process).
    from .services.dub import DubService
    from .services.dub_cache import DubCache
    dub_service = DubService(
        synth=synth_service,
        cache=DubCache(settings.cache_dir / "dub", enabled=settings.cache_enabled),
        translate=translate_service,
    )

    app.state.synth_service = synth_service
    app.state.gpu_gate = gpu_gate
    app.state.asr_service = asr_service
    app.state.translate_service = translate_service
    app.state.dub_service = dub_service
    app.state.engine_installers = {
        "chatterbox": ChatterboxInstaller(),
        "omnivoice": EngineEnvInstaller("install-omnivoice"),
        "voxcpm": EngineEnvInstaller("install-voxcpm"),
        "qwen": EngineEnvInstaller("install-qwen"),
        "nanovllm_km": EngineEnvInstaller("install-nanovllm"),
    }
    app.state.model_downloader = ModelDownloader()
    # asr_service so Whisper's weights are deletable too — it's absent from
    # EngineManager, but it still must be unloaded before its files are removed.
    app.state.model_deleter = ModelDeleter(
        em=engine_manager, asr_service=asr_service, translate_service=translate_service)
    app.state.engine_uninstallers = {
        "chatterbox": EngineEnvUninstaller("chatterbox", em=engine_manager),
        "omnivoice": EngineEnvUninstaller("omnivoice", em=engine_manager),
        "voxcpm": EngineEnvUninstaller("voxcpm", em=engine_manager),
        "qwen": EngineEnvUninstaller("qwen", em=engine_manager),
        # venv dir is "venv-nanovllm" (not "venv-nanovllm_km") to match
        # nanovllm_engine.py's default worker paths, so it needs an explicit
        # venv_dir override here.
        "nanovllm_km": EngineEnvUninstaller(
            "nanovllm_km", em=engine_manager, venv_dir=_BACKEND_ROOT / "venv-nanovllm"
        ),
    }
    app.state.model_downloader = ModelDownloader()
    app.state.youtube_history = YoutubeHistory(settings.cache_dir / "youtube_history")

    # ---- routers
    app.include_router(health_router)
    app.include_router(engines_router)
    app.include_router(voices_router)
    app.include_router(synthesize_router)
    app.include_router(download_router)
    app.include_router(cache_router)
    app.include_router(stream_router)
    app.include_router(system_router)
    app.include_router(asr_router)
    app.include_router(dub_router)
    app.include_router(translate_router)
    app.include_router(youtube_router)

    # ---- static frontend (prod mode only; no-op if frontend/dist is absent)
    _frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    _mount_frontend(app, _frontend_dist)

    # ---- exception handlers
    @app.exception_handler(BackendError)
    async def backend_error_handler(_: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        import traceback
        log.error(
            "Unhandled exception in %s %s:\n%s",
            request.method,
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{type(exc).__name__}: {exc}",
                "code": "internal_error",
            },
        )

    return app


# Module-level instance for `uvicorn backend.app:app`
app = create_app()
