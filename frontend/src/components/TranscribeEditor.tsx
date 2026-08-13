import { useCallback, useEffect, useRef, useState } from "react";
import { Copy, Download, FileAudio, Languages, Link2, Loader2, Pencil, Send, Trash2, Upload, Youtube } from "lucide-react";
import { ApiError, clearYoutubeHistory, correctYoutubeTranscript, deleteYoutubeHistory, getYoutubeHistory, transcribe, transcribeYoutube, translateSegments, type YoutubeHistoryEntry } from "@/lib/api";
import { segmentsToSrt, segmentsToVtt } from "@/lib/subtitles";
import { isRtlText, textDirection } from "@/lib/textStats";
import { focusRing } from "@/lib/theme";
import type { AsrSegment, AsrStatus, EngineLanguage, TranscribeBuffer } from "@/types/models";

interface Props {
  isDark: boolean;
  buffer: TranscribeBuffer;
  onChange: (partial: Partial<TranscribeBuffer>) => void;
  asr: AsrStatus | null;
  /** Translation: the active model's languages, name, and whether it's downloaded. */
  translateLanguages: EngineLanguage[];
  activeTranslator: string;
  translatorDownloaded: boolean;
  /** Opens the download dialog for a model name (whisper or a translator). */
  onDownloadWeights: (name: string) => void;
  onSendToTts: (text: string) => void;
}

const ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.webm";

function saveText(name: string, body: string, mime: string) {
  const url = URL.createObjectURL(new Blob([body], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function TranscribeEditor({
  isDark,
  buffer,
  onChange,
  asr,
  translateLanguages,
  activeTranslator,
  translatorDownloaded,
  onDownloadWeights,
  onSendToTts,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [translated, setTranslated] = useState<AsrSegment[] | null>(null);
  const [translating, setTranslating] = useState(false);
  const [ytUrl, setYtUrl] = useState("");
  const [ytSource, setYtSource] = useState<"captions" | "asr" | null>(null);
  const [ytMeta, setYtMeta] = useState<{
    videoId: string;
    thumbnailUrl: string;
    audioUrl: string;
    videoUrl: string;
    originalText?: string | null;
    originalKind?: string | null;
  } | null>(null);
  const [ytHistory, setYtHistory] = useState<YoutubeHistoryEntry[]>([]);
  const [correctOpen, setCorrectOpen] = useState(false);
  const [correctText, setCorrectText] = useState("");
  const [showOriginal, setShowOriginal] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadYtHistory = useCallback(async () => {
    try {
      const res = await getYoutubeHistory();
      setYtHistory(res.entries);
    } catch {
      // history endpoint unavailable; ignore
    }
  }, []);

  useEffect(() => {
    void loadYtHistory();
  }, [loadYtHistory]);

  const runTranslate = useCallback(async () => {
    if (!buffer.targetLanguage || buffer.segments.length === 0) return;
    setTranslating(true);
    setError(null);
    try {
      const res = await translateSegments({
        segments: buffer.segments.map((s) => ({ start: s.start, end: s.end, text: s.text })),
        source_lang: buffer.detectedLanguage || null,
        target_lang: buffer.targetLanguage,
        model: activeTranslator,
      });
      setTranslated(res.segments);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Translation failed");
    } finally {
      setTranslating(false);
    }
  }, [buffer.targetLanguage, buffer.segments, buffer.detectedLanguage, activeTranslator]);

  const weightsMissing = asr != null && !asr.downloaded;
  const canRun = !!file && !busy && !!asr && asr.downloaded;

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed((Date.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [busy]);

  const run = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await transcribe({
        file,
        language: buffer.language,
        timestamps: buffer.timestamps,
      });
      onChange({
        text: res.text,
        segments: res.segments,
        detectedLanguage: res.language,
        fileName: file.name,
      });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setBusy(false);
    }
  }, [file, buffer.language, buffer.timestamps, onChange]);

  const pick = (f: File | null | undefined) => {
    if (!f) return;
    setFile(f);
    setError(null);
    setYtSource(null);
  };

  const runYoutube = useCallback(async () => {
    const url = ytUrl.trim();
    if (!url || busy) return;
    setBusy(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await transcribeYoutube(url, {
        language: buffer.language,
        timestamps: buffer.timestamps,
      });
      onChange({
        text: res.text,
        segments: res.segments,
        detectedLanguage: res.language,
        fileName: res.source === "asr" ? `youtube-${res.video_id}.wav` : `youtube-${res.video_id}`,
      });
      setYtSource(res.source);
      setYtMeta({
        videoId: res.video_id,
        thumbnailUrl: res.thumbnail_url,
        audioUrl: res.audio_url,
        videoUrl: res.video_url,
        originalText: null,
      });
      await loadYtHistory();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "YouTube fetch failed");
    } finally {
      setBusy(false);
    }
  }, [ytUrl, busy, buffer.language, buffer.timestamps, onChange, loadYtHistory]);

  const applyYtHistoryEntry = useCallback(
    (entry: YoutubeHistoryEntry) => {
      setYtUrl(entry.video_url);
      onChange({
        text: entry.text,
        segments: entry.segments,
        detectedLanguage: entry.language,
        fileName: entry.source === "asr" ? `youtube-${entry.video_id}.wav` : `youtube-${entry.video_id}`,
      });
      setYtSource(entry.source as "captions" | "asr");
      setYtMeta({
        videoId: entry.video_id,
        thumbnailUrl: entry.thumbnail_url,
        audioUrl: entry.audio_url,
        videoUrl: entry.video_url,
        originalText: entry.original?.text ?? null,
        originalKind: entry.original?.kind ?? null,
      });
    },
    [onChange],
  );

  const removeYtHistoryEntry = useCallback(async (videoId: string) => {
    try {
      await deleteYoutubeHistory(videoId);
      setYtHistory((h) => h.filter((e) => e.video_id !== videoId));
    } catch {
      // ignore
    }
  }, []);

  const runYoutubeAsr = useCallback(async () => {
    const url = ytMeta?.videoUrl ?? ytUrl.trim();
    if (!url || busy) return;
    setBusy(true);
    setError(null);
    setElapsed(0);
    try {
      const res = await transcribeYoutube(url, {
        language: buffer.language,
        timestamps: buffer.timestamps,
        forceAsr: true,
      });
      onChange({
        text: res.text,
        segments: res.segments,
        detectedLanguage: res.language,
        fileName: `youtube-${res.video_id}.wav`,
      });
      setYtSource("asr");
      setYtMeta({
        videoId: res.video_id,
        thumbnailUrl: res.thumbnail_url,
        audioUrl: res.audio_url,
        videoUrl: res.video_url,
        originalText: null,
      });
      await loadYtHistory();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "ASR re-transcription failed");
    } finally {
      setBusy(false);
    }
  }, [ytMeta, ytUrl, busy, buffer.language, buffer.timestamps, onChange, loadYtHistory]);

  const saveCorrect = useCallback(async () => {
    if (!ytMeta) return;
    const text = correctText.trim();
    if (!text) return;
    try {
      const saved = await correctYoutubeTranscript({
        video_id: ytMeta.videoId,
        text,
        language: buffer.language ?? "km",
      });
      onChange({
        text: saved.text,
        segments: saved.segments ?? [],
        detectedLanguage: saved.language,
        fileName: `youtube-${saved.video_id}`,
      });
      setYtSource("captions");
      setYtMeta({
        videoId: saved.video_id,
        thumbnailUrl: saved.thumbnail_url,
        audioUrl: saved.audio_url,
        videoUrl: saved.video_url,
        originalText: saved.original?.text ?? null,
        originalKind: saved.original?.kind ?? null,
      });
      setCorrectOpen(false);
      await loadYtHistory();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Save failed");
    }
  }, [ytMeta, correctText, buffer.language, onChange, loadYtHistory]);

  const restoreOriginal = useCallback(() => {
    if (!ytMeta?.originalText) return;
    onChange({
      text: ytMeta.originalText,
      segments: [],
      detectedLanguage: buffer.language ?? "km",
      fileName: `youtube-${ytMeta.videoId}`,
    });
    setCorrectText(ytMeta.originalText);
    setShowOriginal(false);
  }, [ytMeta, buffer.language, onChange]);

  const panel = isDark ? "bg-zinc-900 border-zinc-800" : "bg-white border-gray-200";
  const text = isDark ? "text-white" : "text-gray-900";
  const subtle = isDark ? "text-zinc-400" : "text-gray-600";
  const inputBg = isDark
    ? "bg-zinc-900 border-zinc-800 text-white"
    : "bg-white border-gray-200 text-gray-900";
  const btn = `px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
    isDark
      ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
      : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
  } ${focusRing}`;

  if (weightsMissing) {
    return (
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className={`max-w-2xl mx-auto p-6 rounded-xl border ${panel}`}>
          <h2 className={`text-lg font-semibold ${text}`}>Speech-to-text needs a one-time download</h2>
          <p className={`text-sm mt-2 ${subtle}`}>
            Whisper large-v3-turbo is about 1.6 GB. It runs fully offline once downloaded, and
            transcribes 99 languages.
          </p>
          <button
            type="button"
            onClick={() => onDownloadWeights("whisper")}
            className={`mt-4 px-4 py-2 rounded-lg accent-gradient hover:opacity-90 text-black text-sm font-medium ${focusRing}`}
          >
            Download Whisper (1.6 GB)
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4">
      <div className="max-w-4xl mx-auto space-y-4">
        {error && (
          <div
            className={`p-3 rounded-lg border text-sm ${
              isDark
                ? "bg-red-900/30 border-red-600/40 text-red-200"
                : "bg-red-50 border-red-200 text-red-700"
            }`}
          >
            {error}
          </div>
        )}

        {/* YouTube scribe */}
        <div className={`p-4 rounded-xl border ${panel}`}>
          <div className="flex items-center gap-2 flex-wrap">
            <Youtube className={`w-4 h-4 shrink-0 ${isDark ? "text-red-400" : "text-red-600"}`} />
            <span className={`text-sm font-medium ${text}`}>YouTube transcript</span>
            {ytSource && (
              <span
                className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                  ytSource === "captions"
                    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                    : "bg-indigo-500/15 text-indigo-300 border-indigo-500/30"
                }`}
              >
                {ytSource === "captions" ? "captions" : "ASR"}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-2">
            <div className="relative flex-1">
              <Link2
                className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${
                  isDark ? "text-zinc-500" : "text-gray-400"
                }`}
              />
              <input
                value={ytUrl}
                onChange={(e) => setYtUrl(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void runYoutube();
                }}
                placeholder="Paste a YouTube URL or video id (Khmer captions → ASR fallback)"
                className={`w-full pl-9 pr-3 py-2 rounded-lg border text-sm ${inputBg} ${focusRing}`}
              />
            </div>
            <button
              type="button"
              onClick={() => void runYoutube()}
              disabled={busy || !ytUrl.trim()}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                busy || !ytUrl.trim()
                  ? "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
                  : "accent-gradient hover:opacity-90 text-black"
              }`}
            >
              {busy ? (
                <span className="flex items-center gap-1.5">
                  <Loader2 className="w-4 h-4 animate-spin" /> {elapsed.toFixed(0)}s
                </span>
              ) : (
                "Fetch"
              )}
            </button>
          </div>
          <p className={`text-xs mt-1.5 ${subtle}`}>
            Uses YouTube captions when available; otherwise downloads the audio and transcribes it
            with your selected Khmer ASR model.
          </p>

          {ytMeta && (
            <div className="flex items-center gap-3 mt-3">
              <a href={ytMeta.videoUrl} target="_blank" rel="noreferrer noopener" title="Open on YouTube">
                <img
                  src={ytMeta.thumbnailUrl}
                  alt="Video thumbnail"
                  className="w-28 h-16 object-cover rounded-lg border shrink-0"
                  style={isDark ? { borderColor: "#3f3f46" } : { borderColor: "#e5e7eb" }}
                />
              </a>
              <div className="min-w-0 flex-1">
                <div className={`text-xs font-medium truncate ${text}`}>
                  Video {ytMeta.videoId.slice(0, 11)}
                </div>
                <div className={`text-[11px] ${subtle}`}>
                  {ytSource === "captions"
                    ? "Transcript from YouTube captions"
                    : "Transcript generated by ASR"}
                </div>
                <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                  <a
                    href={ytMeta.audioUrl}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                      isDark
                        ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
                        : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                    } ${focusRing}`}
                    title="Download the audio as WAV"
                  >
                    <Download className="w-3.5 h-3.5" /> Audio
                  </a>
                  <button
                    type="button"
                    onClick={() => void runYoutubeAsr()}
                    disabled={busy}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                      isDark
                        ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
                        : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                    } ${focusRing} ${busy ? "opacity-50 cursor-wait" : ""}`}
                    title="Ignore captions and re-transcribe the audio with Whisper"
                  >
                    <Loader2 className={`w-3.5 h-3.5 ${busy ? "animate-spin" : ""}`} /> Re-transcribe (ASR)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setCorrectText(buffer.text);
                      setCorrectOpen((o) => !o);
                    }}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                      isDark
                        ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
                        : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                    } ${focusRing}`}
                    title="Paste the correct lyrics / text and save it"
                  >
                    <Pencil className="w-3.5 h-3.5" /> Correct
                  </button>
                  <a
                    href={ytMeta.videoUrl}
                    target="_blank"
                    rel="noreferrer noopener"
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                      isDark
                        ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
                        : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                    } ${focusRing}`}
                    title="Open on YouTube"
                  >
                    <Youtube className="w-3.5 h-3.5" /> Watch
                  </a>
                </div>
                {correctOpen && (
                  <div className="mt-2">
                    <textarea
                      value={correctText}
                      onChange={(e) => setCorrectText(e.target.value)}
                      rows={6}
                      placeholder="Paste the correct lyrics / transcript here…"
                      className={`w-full rounded-lg border px-3 py-2 text-sm resize-y ${inputBg} ${focusRing}`}
                    />
                    <button
                      type="button"
                      onClick={() => void saveCorrect()}
                      disabled={!correctText.trim()}
                      className={`mt-2 px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                        correctText.trim()
                          ? "accent-gradient hover:opacity-90 text-black"
                          : "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
                      }`}
                    >
                      Save corrected transcript
                    </button>
                  </div>
                )}
                {ytMeta.originalText && (
                  <div className="mt-2">
                    <button
                      type="button"
                      onClick={() => setShowOriginal((o) => !o)}
                      className={`text-xs underline ${
                        isDark ? "text-zinc-400 hover:text-zinc-200" : "text-gray-500 hover:text-gray-700"
                      } ${focusRing}`}
                    >
                      {showOriginal ? "Hide" : "Show"} original
                      {ytMeta.originalKind ? ` (${ytMeta.originalKind})` : ""}
                    </button>
                    {showOriginal && (
                      <div className={`mt-1.5 rounded-lg border p-3 text-xs leading-relaxed whitespace-pre-wrap max-h-40 overflow-y-auto ${inputBg}`}>
                        {ytMeta.originalText}
                        <div className="mt-2">
                          <button
                            type="button"
                            onClick={restoreOriginal}
                            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                              isDark
                                ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
                                : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                            } ${focusRing}`}
                          >
                            Restore original text
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* YouTube history */}
        {ytHistory.length > 0 && (
          <div className={`p-4 rounded-xl border ${panel}`}>
            <div className="flex items-center justify-between mb-2 gap-2">
              <h3 className={`text-sm font-semibold ${text}`}>
                YouTube history
                <span className={`ml-2 font-normal ${subtle}`}>{ytHistory.length}</span>
              </h3>
              <button
                type="button"
                className={btn}
                onClick={async () => {
                  try {
                    await clearYoutubeHistory();
                    setYtHistory([]);
                  } catch {
                    // ignore
                  }
                }}
              >
                Clear
              </button>
            </div>
            <ul className="space-y-1.5">
              {ytHistory.map((entry) => (
                <li
                  key={entry.video_id}
                  className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-colors ${
                    isDark
                      ? "border-zinc-800 hover:border-indigo-500/50 bg-zinc-950/40"
                      : "border-gray-200 hover:border-indigo-500/50 bg-gray-50"
                  } ${focusRing}`}
                  onClick={() => applyYtHistoryEntry(entry)}
                  title="Load this transcript"
                >
                  <img
                    src={entry.thumbnail_url}
                    alt=""
                    className="w-16 h-10 object-cover rounded shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <div className={`text-xs font-medium truncate ${text}`}>
                      {entry.video_id.slice(0, 11)} · {entry.language || "?"}
                    </div>
                    <div className={`text-[11px] truncate ${subtle}`}>
                      {entry.created_at
                        ? new Date(entry.created_at * 1000).toLocaleString()
                        : ""}
                    </div>
                  </div>
                  <span
                    className={`shrink-0 text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                      entry.kind === "captions"
                        ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                        : entry.kind === "auto_cc"
                          ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
                          : "bg-indigo-500/15 text-indigo-300 border-indigo-500/30"
                    }`}
                  >
                    {entry.kind === "captions"
                      ? "captions"
                      : entry.kind === "auto_cc"
                        ? "auto CC"
                        : "ASR"}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      void removeYtHistoryEntry(entry.video_id);
                    }}
                    className={`p-1 rounded ${
                      isDark ? "text-zinc-500 hover:text-red-400" : "text-gray-500 hover:text-red-600"
                    } ${focusRing}`}
                    title="Remove from history"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </li>
              ))}
            </ul>
            <p className={`text-[11px] mt-2 ${subtle}`}>
              Badges: <span className="text-emerald-400">captions</span> = manual CC ·{" "}
              <span className="text-amber-400">auto CC</span> = YouTube auto-generated ·{" "}
              <span className="text-indigo-300">ASR</span> = transcribed by our Whisper model.
            </p>
          </div>
        )}

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            pick(e.dataTransfer.files?.[0]);
          }}
          className={`p-6 rounded-xl border-2 border-dashed text-center transition-colors ${
            dragOver ? "border-indigo-500" : isDark ? "border-zinc-700" : "border-gray-300"
          } ${panel}`}
        >
          <FileAudio className="w-8 h-8 mx-auto mb-2 text-indigo-400" />
          <p className={`text-sm ${text}`}>
            {file ? file.name : "Drop an audio file here, or choose one"}
          </p>
          <p className={`text-xs mt-1 ${subtle}`}>WAV, MP3, FLAC, OGG, M4A, WebM · up to 100 MB</p>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0])}
          />
          <div className="flex items-center justify-center gap-2 mt-4">
            <button type="button" onClick={() => inputRef.current?.click()} className={btn}>
              <span className="flex items-center gap-1.5">
                <Upload className="w-4 h-4" /> Choose file
              </span>
            </button>
            <button
              type="button"
              onClick={() => void run()}
              disabled={!canRun}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                canRun ? "accent-gradient hover:opacity-90 text-black" : "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
              }`}
            >
              {busy ? (
                <span className="flex items-center gap-1.5">
                  <Loader2 className="w-4 h-4 animate-spin" /> Transcribing… {elapsed.toFixed(1)}s
                </span>
              ) : (
                "Transcribe"
              )}
            </button>
          </div>
        </div>

        {(buffer.text || buffer.segments.length > 0) && (
          <div className={`p-4 rounded-xl border ${panel}`}>
            <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
              <h3 className={`text-sm font-semibold ${text}`}>
                Transcript
                {buffer.detectedLanguage && (
                  <span className={`ml-2 font-normal ${subtle}`}>
                    detected: {buffer.detectedLanguage}
                  </span>
                )}
              </h3>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className={btn}
                  onClick={() => void navigator.clipboard.writeText(buffer.text)}
                >
                  <span className="flex items-center gap-1.5">
                    <Copy className="w-4 h-4" /> Copy
                  </span>
                </button>
                <button
                  type="button"
                  className={btn}
                  disabled={!buffer.text.trim()}
                  onClick={() => onSendToTts(buffer.text)}
                >
                  <span className="flex items-center gap-1.5">
                    <Send className="w-4 h-4" /> Send to Text-to-Voice
                  </span>
                </button>
                <button
                  type="button"
                  className={btn}
                  disabled={buffer.segments.length === 0}
                  title={
                    buffer.segments.length === 0
                      ? "Enable Timestamps and transcribe again"
                      : "Download SubRip subtitles"
                  }
                  onClick={() =>
                    saveText(
                      `${buffer.fileName || "transcript"}.srt`,
                      segmentsToSrt(buffer.segments),
                      "text/plain",
                    )
                  }
                >
                  <span className="flex items-center gap-1.5">
                    <Download className="w-4 h-4" /> .srt
                  </span>
                </button>
                <button
                  type="button"
                  className={btn}
                  disabled={buffer.segments.length === 0}
                  onClick={() =>
                    saveText(
                      `${buffer.fileName || "transcript"}.vtt`,
                      segmentsToVtt(buffer.segments),
                      "text/vtt",
                    )
                  }
                >
                  <span className="flex items-center gap-1.5">
                    <Download className="w-4 h-4" /> .vtt
                  </span>
                </button>
              </div>
            </div>

            <textarea
              value={buffer.text}
              onChange={(e) => onChange({ text: e.target.value })}
              rows={10}
              spellCheck={false}
              dir={textDirection(buffer.text)}
              className={`w-full rounded-lg border px-3 py-2 text-sm resize-y ${
                isRtlText(buffer.text) ? "text-right" : "text-left"
              } ${
                isDark
                  ? "bg-zinc-950 border-zinc-800 text-zinc-100"
                  : "bg-white border-gray-200 text-gray-900"
              } ${focusRing}`}
            />

            {buffer.segments.length > 0 && (
              <details className="mt-3">
                <summary className={`text-xs cursor-pointer ${subtle}`}>
                  {buffer.segments.length} timestamped segment
                  {buffer.segments.length !== 1 ? "s" : ""}
                </summary>
                <ul className={`mt-2 text-xs space-y-1 ${subtle}`}>
                  {buffer.segments.map((s, i) => (
                    <li key={i}>
                      <span className="tabular-nums">
                        [{s.start.toFixed(2)}–{s.end.toFixed(2)}]
                      </span>{" "}
                      {s.text}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}

        {buffer.segments.length > 0 && (
          <div className={`p-4 rounded-xl border ${panel}`}>
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <Languages className="w-4 h-4 text-indigo-400" />
              <label className={`text-sm font-semibold ${text}`} htmlFor="tr-target-lang">Translate to</label>
              <select
                id="tr-target-lang"
                value={buffer.targetLanguage ?? ""}
                onChange={(e) => { onChange({ targetLanguage: e.target.value || null }); setTranslated(null); }}
                className={`rounded-md border px-2 py-1.5 text-sm ${
                  isDark ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-gray-200 text-gray-900"
                } ${focusRing}`}
              >
                <option value="">Choose a language…</option>
                {translateLanguages.map((l) => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
              {buffer.targetLanguage && !translatorDownloaded ? (
                <button
                  type="button"
                  onClick={() => onDownloadWeights(activeTranslator)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium text-black accent-gradient hover:opacity-90 ${focusRing}`}
                >
                  Download translation model
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void runTranslate()}
                  disabled={!buffer.targetLanguage || translating}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                    buffer.targetLanguage && !translating ? "accent-gradient hover:opacity-90 text-black" : "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
                  }`}
                >
                  {translating ? (
                    <span className="flex items-center gap-1.5"><Loader2 className="w-4 h-4 animate-spin" /> Translating…</span>
                  ) : (
                    "Translate"
                  )}
                </button>
              )}
              <span className={`text-xs ${subtle}`}>with {activeTranslator}</span>
            </div>

            {translated && (
              <>
                <div className="flex items-center justify-end gap-2 mb-2">
                  <button
                    type="button"
                    className={btn}
                    onClick={() => void navigator.clipboard.writeText(translated.map((s) => s.text).join(" "))}
                  >
                    <span className="flex items-center gap-1.5"><Copy className="w-4 h-4" /> Copy</span>
                  </button>
                  <button
                    type="button"
                    className={btn}
                    onClick={() => saveText(`${buffer.fileName || "transcript"}.${buffer.targetLanguage}.srt`, segmentsToSrt(translated), "text/plain")}
                  >
                    <span className="flex items-center gap-1.5"><Download className="w-4 h-4" /> .srt</span>
                  </button>
                  <button
                    type="button"
                    className={btn}
                    onClick={() => saveText(`${buffer.fileName || "transcript"}.${buffer.targetLanguage}.vtt`, segmentsToVtt(translated), "text/vtt")}
                  >
                    <span className="flex items-center gap-1.5"><Download className="w-4 h-4" /> .vtt</span>
                  </button>
                </div>
                <ul className="space-y-1 text-sm">
                  {translated.map((s, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className={`tabular-nums text-xs pt-0.5 shrink-0 ${subtle}`}>[{s.start.toFixed(2)}–{s.end.toFixed(2)}]</span>
                      <span dir={textDirection(s.text)} className={`${isRtlText(s.text) ? "text-right" : "text-left"} ${text}`}>{s.text}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
