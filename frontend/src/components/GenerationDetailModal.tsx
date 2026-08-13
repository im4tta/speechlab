import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, Volume2, X } from "lucide-react";
import { focusRing } from "@/lib/theme";
import type { CacheEntryInfo } from "@/lib/api";
import { cacheAudioUrl } from "@/lib/api";
import { isRtlText } from "@/lib/textStats";
import { Waveform } from "./Waveform";

// Highlight words this many seconds ahead of the audio cursor. Compensates for
// the even-distribution approximation + perception (reading runs slightly
// ahead of speech), so the highlight feels in sync rather than lagging.
const HIGHLIGHT_LEAD_SEC = 0.18;

interface Props {
  isDark: boolean;
  entry: CacheEntryInfo;
  onClose: () => void;
}

function formatTime(sec: number): string {
  if (!isFinite(sec) || isNaN(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDate(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

function Detail({ isDark, label, value }: { isDark: boolean; label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div
        className={`text-[10px] uppercase tracking-wide ${
          isDark ? "text-zinc-500" : "text-gray-500"
        }`}
      >
        {label}
      </div>
      <div
        className={`mt-0.5 truncate ${isDark ? "text-zinc-200" : "text-gray-800"}`}
        title={value}
      >
        {value || "—"}
      </div>
    </div>
  );
}

export function GenerationDetailModal({ isDark, entry, onClose }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);

  const progress = duration > 0 ? currentTime / duration : 0;

  // Pause on unmount
  useEffect(() => {
    return () => {
      audioRef.current?.pause();
    };
  }, []);

  // While playing, sample currentTime every animation frame (~60fps) instead
  // of relying on the `timeupdate` event (which fires only ~4x/sec and makes
  // the word highlight visibly lag/step).
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const a = audioRef.current;
      if (a) setCurrentTime(a.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      void audio.play();
    } else {
      audio.pause();
    }
  };

  const handleSeek = (fraction: number) => {
    const audio = audioRef.current;
    if (!audio || !isFinite(audio.duration)) return;
    audio.currentTime = fraction * audio.duration;
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setVolume(v);
    if (audioRef.current) audioRef.current.volume = v;
  };

  const audioUrl = cacheAudioUrl(entry.hash);

  // Split the text into tokens, tagging each non-whitespace token with its
  // word index so we can light words up left-to-right as playback advances.
  // (TTS engines don't give per-word timestamps, so we approximate with an
  // even distribution across the clip duration.)
  const tokens = useMemo(() => {
    const parts = (entry.text ?? "").split(/(\s+)/);
    let wi = 0;
    return parts.map((part) => {
      const isSpace = part === "" || /^\s+$/.test(part);
      return { part, idx: isSpace ? -1 : wi++ };
    });
  }, [entry.text]);
  const wordCount = tokens.reduce((n, t) => (t.idx >= 0 ? n + 1 : n), 0);
  const leadProgress =
    duration > 0 ? Math.min(1, (currentTime + HIGHLIGHT_LEAD_SEC) / duration) : 0;
  const spokenWords = leadProgress * wordCount;
  const rtl = isRtlText(entry.text);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Detail: ${entry.name}`}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* Modal card */}
      <div
        className={`relative w-full max-w-6xl max-h-[94vh] flex flex-col rounded-2xl shadow-2xl border ${
          isDark ? "bg-zinc-900 border-zinc-800" : "bg-white border-gray-200"
        }`}
      >
        {/* Header */}
        <div
          className={`px-5 py-4 border-b flex items-start justify-between gap-3 shrink-0 ${
            isDark ? "border-zinc-800" : "border-gray-200"
          }`}
        >
          <div className="min-w-0">
            <div
              className={`text-sm font-semibold truncate ${
                isDark ? "text-white" : "text-gray-900"
              }`}
            >
              {(entry.name ?? "").trim() || `Generation ${entry.hash.slice(0, 8)}`}
            </div>
            <div
              className={`text-xs mt-0.5 ${isDark ? "text-zinc-400" : "text-gray-600"}`}
            >
              {entry.duration_sec.toFixed(1)}s
              {entry.engine ? ` · ${entry.engine}` : ""}
              {entry.voice ? ` · ${entry.voice}` : ""}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className={`p-1 rounded transition-colors shrink-0 ${
              isDark
                ? "text-zinc-400 hover:text-zinc-300"
                : "text-gray-600 hover:text-gray-600"
            } ${focusRing}`}
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-8">
          {/* Full text — words brighten as the audio plays (RTL-aware) */}
          <div
            dir={rtl ? "rtl" : "ltr"}
            className={`text-2xl font-medium leading-relaxed max-h-72 overflow-y-auto rounded-lg p-5 whitespace-pre-wrap ${
              rtl ? "text-right" : "text-left"
            } ${isDark ? "bg-zinc-800/40" : "bg-gray-50"}`}
          >
            {entry.text ? (
              tokens.map((t, i) => {
                if (t.idx < 0) return <span key={i}>{t.part}</span>;
                const spoken = t.idx < spokenWords;
                return (
                  <span
                    key={i}
                    className={`transition-colors duration-700 ease-out ${
                      spoken
                        ? isDark ? "text-white" : "text-gray-900"
                        : isDark ? "text-zinc-600" : "text-gray-600"
                    }`}
                  >
                    {t.part}
                  </span>
                );
              })
            ) : (
              <span className={`text-base ${isDark ? "text-zinc-600" : "text-gray-600"}`}>
                No text stored for this clip.
              </span>
            )}
          </div>

          {/* Waveform */}
          <div className="px-1 py-2">
            <Waveform
              url={audioUrl}
              progress={progress}
              isDark={isDark}
              onSeek={handleSeek}
              height={160}
            />
          </div>

          {/* Player bar */}
          <div className="flex items-center gap-3">
            {/* Play / pause */}
            <button
              type="button"
              onClick={togglePlay}
              className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-full transition-colors ${
                isDark
                  ? "bg-indigo-700/40 hover:bg-indigo-700/60 text-indigo-200"
                  : "bg-indigo-50 hover:bg-indigo-100 text-indigo-700"
              } ${focusRing}`}
              title={playing ? "Pause" : "Play"}
            >
              {playing ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </button>

            {/* Seek range */}
            <input
              type="range"
              min={0}
              max={1}
              step={0.001}
              value={progress}
              onChange={(e) => handleSeek(parseFloat(e.target.value))}
              className={`flex-1 accent-indigo-500 ${focusRing}`}
            />

            {/* Time display */}
            <span
              className={`text-xs tabular-nums shrink-0 ${
                isDark ? "text-zinc-400" : "text-gray-600"
              }`}
            >
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>

            {/* Volume */}
            <Volume2
              className={`w-4 h-4 shrink-0 ${isDark ? "text-zinc-400" : "text-gray-600"}`}
            />
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={handleVolumeChange}
              className={`w-20 accent-indigo-500 ${focusRing}`}
            />
          </div>

          {/* Generation details */}
          <div
            className={`rounded-lg border p-4 ${
              isDark ? "bg-zinc-800/40 border-zinc-800" : "bg-gray-50 border-gray-200"
            }`}
          >
            <h4
              className={`text-xs font-semibold uppercase tracking-wide mb-3 ${
                isDark ? "text-zinc-400" : "text-gray-600"
              }`}
            >
              Generation details
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3 text-xs">
              <Detail isDark={isDark} label="Generated" value={formatDate(entry.created_at)} />
              <Detail isDark={isDark} label="Engine" value={entry.engine ?? "—"} />
              <Detail isDark={isDark} label="Duration" value={`${entry.duration_sec.toFixed(1)}s`} />
              <Detail isDark={isDark} label="Sample rate" value={`${(entry.sample_rate / 1000).toFixed(0)} kHz`} />
              <Detail isDark={isDark} label="Words" value={entry.word_count != null ? String(entry.word_count) : "—"} />
              <Detail isDark={isDark} label="Characters" value={entry.char_count != null ? String(entry.char_count) : "—"} />
              <Detail
                isDark={isDark}
                label="Inference time"
                value={entry.inference_ms ? formatTime(entry.inference_ms / 1000) : "—"}
              />
              <Detail isDark={isDark} label="GPU" value={entry.machine?.gpu ?? "—"} />
              <Detail isDark={isDark} label="CPU" value={entry.machine?.cpu ?? "—"} />
              <Detail
                isDark={isDark}
                label="RAM"
                value={entry.machine?.ram_gb != null ? `${entry.machine.ram_gb} GB` : "—"}
              />
              <Detail isDark={isDark} label="OS" value={entry.machine?.os ?? "—"} />
            </div>
          </div>
        </div>
      </div>

      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={audioUrl}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime ?? 0)}
        onLoadedMetadata={() => setDuration(audioRef.current?.duration ?? 0)}
        preload="metadata"
      />
    </div>
  );
}
