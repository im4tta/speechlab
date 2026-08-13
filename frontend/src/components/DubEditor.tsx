import { useCallback, useEffect, useRef, useState } from "react";
import { AudioLines, Download, FileAudio, Loader2, Upload } from "lucide-react";
import { ApiError, dub, transcribe } from "@/lib/api";
import { isRtlText, textDirection } from "@/lib/textStats";
import { focusRing } from "@/lib/theme";
import { DESIGN_CHIPS, appendDesignChip, effectiveMode, type VoiceMode } from "@/lib/voiceModes";
import type { AsrSegment, AsrStatus, DubBuffer, EngineLanguage, Voice } from "@/types/models";

/** Create an object URL for a Blob/File and revoke it when it changes or unmounts. */
function useObjectUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);
  return url;
}

interface Props {
  isDark: boolean;
  buffer: DubBuffer;
  onChange: (partial: Partial<DubBuffer>) => void;
  asr: AsrStatus | null;
  /** Target voice picked in the library; null until chosen. */
  activeVoice: Voice | null;
  activeEngine: string | null;
  /** OmniVoice/VoxCPM expose Clone/Design/Auto; VoxCPM adds style-on-clone. */
  supportsVoiceModes: boolean;
  supportsStyleClone: boolean;
  /** Qwen: always-available free-text style prompt (no mode toggle). */
  supportsStylePrompt: boolean;
  /** Translation: the active model's languages, name, and whether it's downloaded. */
  translateLanguages: EngineLanguage[];
  activeTranslator: string;
  translatorDownloaded: boolean;
  /** Opens the download dialog for a model name (whisper or a translator). */
  onDownloadWeights: (name: string) => void;
}

const ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.webm";

// Popular dubbing targets, shown as one-click chips (filtered to what the active
// translator actually supports). The source language is auto-detected from audio.
const POPULAR_TARGET_CODES = ["es", "fr", "de", "pt", "it", "zh", "ja", "ko", "hi", "ur", "ar", "ru"];

export function DubEditor({
  isDark,
  buffer,
  onChange,
  asr,
  activeVoice,
  activeEngine,
  supportsVoiceModes,
  supportsStyleClone,
  supportsStylePrompt,
  translateLanguages,
  activeTranslator,
  translatorDownloaded,
  onDownloadWeights,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState<null | "transcribe" | "dub">(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [dubBlob, setDubBlob] = useState<Blob | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dubAudioRef = useRef<HTMLAudioElement>(null);

  // Native <audio> sources for the original clip and the generated dub, so the
  // user can A/B them with real scrubbers.
  const originalUrl = useObjectUrl(file);
  const dubbedUrl = useObjectUrl(dubBlob);

  // Auto-play the dub once it's ready (best-effort; browsers may block autoplay).
  useEffect(() => {
    if (dubbedUrl && dubAudioRef.current) {
      dubAudioRef.current.play().catch(() => {});
    }
  }, [dubbedUrl]);

  const weightsMissing = asr != null && !asr.downloaded;
  const canTranscribe = !!file && !busy && !!asr && asr.downloaded;
  // Effective voice mode: engines with modes derive it (explicit > clone-if-voice
  // > auto); every other engine is always clone (voice required).
  const mode: VoiceMode = supportsVoiceModes
    ? effectiveMode({ voice: buffer.voiceId ?? "", omnivoiceMode: buffer.voiceMode })
    : "clone";
  const design = (buffer.voiceDesign ?? "").trim();
  // clone needs a reference voice; design/auto don't.
  const needsVoice = mode === "clone";
  const canDub = !busy && buffer.segments.length > 0 && (!needsVoice || !!activeVoice);

  // Popular target-language chips, filtered to what the active translator supports.
  const popularTargets = POPULAR_TARGET_CODES
    .map((code) => translateLanguages.find((l) => l.code === code))
    .filter((l): l is (typeof translateLanguages)[number] => !!l);

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed((Date.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [busy]);

  const runTranscribe = useCallback(async () => {
    if (!file) return;
    setBusy("transcribe");
    setError(null);
    setElapsed(0);
    setDubBlob(null); // a new source invalidates any prior dub
    try {
      const res = await transcribe({ file, timestamps: true });
      onChange({
        segments: res.segments,
        detectedLanguage: res.language,
        fileName: file.name,
      });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setBusy(null);
    }
  }, [file, onChange]);

  const runDub = useCallback(async () => {
    if (buffer.segments.length === 0) return;
    if (needsVoice && !activeVoice) return;
    // instruct carries the design/style/clone-style prompt (empty = omitted).
    // Mirrors TTS: design and clone (with style) forward it; auto never does.
    const instruct =
      mode === "design" || mode === "clone" ? (design || undefined) : undefined;
    setBusy("dub");
    setError(null);
    setElapsed(0);
    try {
      const res = await dub({
        segments: buffer.segments.map((s) => ({ start: s.start, end: s.end, text: s.text })),
        // design/auto carry no reference voice — sending a stale library voice
        // would only muddy the cache key.
        voice: needsVoice ? (activeVoice?.id ?? "") : "",
        engine: activeEngine ?? undefined,
        // Only mode-aware engines carry an explicit voice_mode; others stay clone.
        ...(supportsVoiceModes ? { voice_mode: mode } : {}),
        ...(instruct ? { instruct } : {}),
        // Cross-language dubbing: translate to the chosen target first.
        ...(buffer.targetLanguage ? {
          source_language: buffer.detectedLanguage || undefined,
          target_language: buffer.targetLanguage,
          translator: activeTranslator,
        } : {}),
      });
      // A native <audio> reads the WAV header, so no sample rate needed here.
      setDubBlob(new Blob([res.audio], { type: "audio/wav" }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Dubbing failed");
    } finally {
      setBusy(null);
    }
  }, [activeVoice, activeEngine, buffer.segments, buffer.targetLanguage, buffer.detectedLanguage,
      mode, design, needsVoice, supportsVoiceModes, activeTranslator]);

  const download = useCallback(() => {
    if (!dubBlob) return;
    const url = URL.createObjectURL(dubBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dub-${(buffer.fileName || "audio").replace(/\.[^.]+$/, "")}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }, [dubBlob, buffer.fileName]);

  const editSegment = (i: number, text: string) => {
    const next: AsrSegment[] = buffer.segments.map((s, idx) => (idx === i ? { ...s, text } : s));
    onChange({ segments: next });
    setDubBlob(null); // editing text invalidates the prior dub
  };

  const pick = (f: File | null | undefined) => {
    if (!f) return;
    setFile(f);
    setError(null);
    setDubBlob(null);
  };

  const panel = isDark ? "bg-zinc-900 border-zinc-800" : "bg-white border-gray-200";
  const text = isDark ? "text-white" : "text-gray-900";
  const subtle = isDark ? "text-zinc-400" : "text-gray-600";
  const btn = `px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
    isDark
      ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
      : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
  } ${focusRing}`;

  if (weightsMissing) {
    return (
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className={`max-w-2xl mx-auto p-6 rounded-xl border ${panel}`}>
          <h2 className={`text-lg font-semibold ${text}`}>Dubbing needs speech-to-text</h2>
          <p className={`text-sm mt-2 ${subtle}`}>
            Dubbing transcribes your clip with Whisper large-v3-turbo (about 1.6 GB), then re-voices
            it. It runs fully offline once downloaded.
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
            {file ? file.name : "Drop an audio clip to re-voice, or choose one"}
          </p>
          <p className={`text-xs mt-1 ${subtle}`}>Same-language re-voicing · WAV, MP3, FLAC, OGG, M4A, WebM</p>

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
              onClick={() => void runTranscribe()}
              disabled={!canTranscribe}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                canTranscribe ? "accent-gradient hover:opacity-90 text-black" : "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
              }`}
            >
              {busy === "transcribe" ? (
                <span className="flex items-center gap-1.5">
                  <Loader2 className="w-4 h-4 animate-spin" /> Transcribing… {elapsed.toFixed(1)}s
                </span>
              ) : (
                "Transcribe"
              )}
            </button>
          </div>
        </div>

        {buffer.segments.length > 0 && (
          <>
          {/* Dub controls: language, voice mode, generate, players */}
          <div className={`p-4 rounded-xl border ${panel}`}>
            <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
              <h3 className={`text-sm font-semibold ${text}`}>Dub</h3>
              <div className="flex items-center gap-2">
                <span className={`text-xs ${subtle}`}>
                  {mode === "design" ? (
                    "designed voice"
                  ) : mode === "auto" ? (
                    "auto voice"
                  ) : activeVoice ? (
                    <>
                      target voice: <span className={text}>{activeVoice.name}</span>
                    </>
                  ) : (
                    "pick a target voice in the library →"
                  )}
                </span>
                <button
                  type="button"
                  onClick={() => void runDub()}
                  disabled={!canDub}
                  title={needsVoice && !activeVoice ? "Pick a target voice in the library first" : "Re-voice the clip"}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${focusRing} ${
                canDub ? "accent-gradient hover:opacity-90 text-black" : "bg-indigo-600/40 text-zinc-300 cursor-not-allowed"
              }`}
                >
                  {busy === "dub" ? (
                    <span className="flex items-center gap-1.5">
                      <Loader2 className="w-4 h-4 animate-spin" /> Dubbing… {elapsed.toFixed(1)}s
                    </span>
                  ) : (
                    <span className="flex items-center gap-1.5">
                      <AudioLines className="w-4 h-4" /> Generate Dub
                    </span>
                  )}
                </button>
              </div>
            </div>

            {/* Target language: same-language by default, or translate before synth.
                Source is auto-detected from the audio. */}
            <div className="mb-3 space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <label className={`text-xs ${subtle}`} htmlFor="dub-target-lang">Language</label>
                <select
                  id="dub-target-lang"
                  value={buffer.targetLanguage ?? ""}
                  onChange={(e) => onChange({ targetLanguage: e.target.value || null })}
                  className={`rounded-md border px-2 py-1.5 text-sm ${
                    isDark ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-gray-200 text-gray-900"
                  } ${focusRing}`}
                >
                  <option value="">Same language (no translation)</option>
                  {translateLanguages.map((l) => (
                    <option key={l.code} value={l.code}>{l.label}</option>
                  ))}
                </select>
                {buffer.targetLanguage && (
                  translatorDownloaded ? (
                    <span className={`text-xs ${subtle}`}>translating with {activeTranslator}</span>
                  ) : (
                    <button
                      type="button"
                      onClick={() => onDownloadWeights(activeTranslator)}
                      className={`px-2 py-1 rounded text-xs font-medium accent-gradient hover:opacity-90 text-black ${focusRing}`}
                    >
                      Download translation model
                    </button>
                  )
                )}
              </div>
              {popularTargets.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {popularTargets.map((l) => {
                    const on = buffer.targetLanguage === l.code;
                    return (
                      <button
                        key={l.code}
                        type="button"
                        onClick={() => onChange({ targetLanguage: on ? null : l.code })}
                        className={`px-2 py-0.5 text-[11px] rounded border transition-colors ${
                          on
                            ? "accent-gradient text-black border-transparent"
                            : isDark
                              ? "border-zinc-700 text-zinc-400 hover:border-indigo-500 hover:text-indigo-300"
                              : "border-gray-300 text-gray-600 hover:border-indigo-500 hover:text-indigo-600"
                        } ${focusRing}`}
                      >
                        {l.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Qwen always-available style prompt (built-in voice + optional style) */}
            {supportsStylePrompt && (
              <input
                type="text"
                value={buffer.voiceDesign ?? ""}
                onChange={(e) => onChange({ voiceDesign: e.target.value })}
                placeholder="Style (optional) — e.g. cheerful, slightly faster, whispering"
                className={`mb-3 w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${
                  isDark ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-gray-200 text-gray-900"
                } ${focusRing}`}
              />
            )}

            {/* OmniVoice / VoxCPM voice mode: Clone / Design / Auto */}
            {supportsVoiceModes && (
              <div className={`mb-3 rounded-lg border p-3 space-y-2 ${isDark ? "border-zinc-800 bg-zinc-950/40" : "border-gray-200 bg-gray-50"}`}>
                <div className="flex gap-1.5">
                  {(["clone", "design", "auto"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => onChange({ voiceMode: m })}
                      className={`flex-1 px-3 py-1.5 text-sm font-medium rounded capitalize transition-colors ${
                        mode === m
                          ? "accent-gradient text-black"
                          : isDark
                            ? "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                      } ${focusRing}`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                {mode === "clone" && (
                  <div className="space-y-1.5">
                    <p className={`text-xs ${subtle}`}>
                      Re-voices the clip in the library voice:{" "}
                      <span className="text-indigo-300">{activeVoice ? activeVoice.name : "none selected"}</span>
                    </p>
                    {supportsStyleClone && (
                      <input
                        type="text"
                        value={buffer.voiceDesign ?? ""}
                        onChange={(e) => onChange({ voiceDesign: e.target.value })}
                        placeholder="Style (optional) — e.g. cheerful, slightly faster"
                        className={`w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${
                          isDark ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-gray-200 text-gray-900"
                        } ${focusRing}`}
                      />
                    )}
                  </div>
                )}
                {mode === "design" && (
                  <div className="space-y-1.5">
                    <input
                      type="text"
                      value={buffer.voiceDesign ?? ""}
                      onChange={(e) => onChange({ voiceDesign: e.target.value })}
                      placeholder={activeEngine === "voxcpm" ? "e.g. a young woman, gentle and sweet" : "e.g. female, low pitch, british accent"}
                      className={`w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${
                        isDark ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-gray-200 text-gray-900"
                      } ${focusRing}`}
                    />
                    {activeEngine === "omnivoice" && (
                      <div className="flex flex-wrap gap-1">
                        {DESIGN_CHIPS.map((chip) => (
                          <button
                            key={chip}
                            type="button"
                            onClick={() => onChange({ voiceDesign: appendDesignChip(buffer.voiceDesign ?? "", chip) })}
                            className={`px-1.5 py-0.5 text-[11px] rounded border transition-colors ${
                              isDark
                                ? "border-zinc-700 text-zinc-400 hover:border-indigo-500 hover:text-indigo-300"
                                : "border-gray-300 text-gray-600 hover:border-indigo-500 hover:text-indigo-600"
                            } ${focusRing}`}
                          >
                            {chip}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {mode === "auto" && (
                  <p className={`text-xs italic ${subtle}`}>
                    {activeEngine === "voxcpm"
                      ? "VoxCPM will design a fresh voice for the clip."
                      : "OmniVoice will invent a voice for the clip."}
                  </p>
                )}
              </div>
            )}

            {dubbedUrl && (
              <div className={`mt-4 rounded-lg border p-3 space-y-3 ${isDark ? "border-zinc-800 bg-zinc-950/40" : "border-gray-200 bg-gray-50"}`}>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-1">
                    <div className={`text-xs font-medium ${subtle}`}>Original</div>
                    {originalUrl ? (
                      <audio controls src={originalUrl} className="w-full h-9" />
                    ) : (
                      <p className={`text-xs italic ${subtle}`}>
                        Re-choose the file to compare the original.
                      </p>
                    )}
                  </div>
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-indigo-400">Dubbed</div>
                    <audio ref={dubAudioRef} controls src={dubbedUrl} className="w-full h-9" />
                  </div>
                </div>
                <button type="button" onClick={download} className={btn}>
                  <span className="flex items-center gap-1.5">
                    <Download className="w-4 h-4" /> Download dubbed WAV
                  </span>
                </button>
              </div>
            )}
          </div>

          {/* Transcript: editable segment text, kept separate from the controls */}
          <div className={`p-4 rounded-xl border ${panel}`}>
            <h3 className={`text-sm font-semibold mb-3 ${text}`}>
              Transcript
              {buffer.detectedLanguage && (
                <span className={`ml-2 font-normal ${subtle}`}>detected: {buffer.detectedLanguage}</span>
              )}
            </h3>
            <ul className="space-y-2">
              {buffer.segments.map((s, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className={`tabular-nums text-xs pt-2.5 shrink-0 ${subtle}`}>
                    [{s.start.toFixed(2)}–{s.end.toFixed(2)}]
                  </span>
                  <input
                    value={s.text}
                    onChange={(e) => editSegment(i, e.target.value)}
                    spellCheck={false}
                    dir={textDirection(s.text)}
                    className={`flex-1 rounded-lg border px-3 py-2 text-sm ${
                      isRtlText(s.text) ? "text-right" : "text-left"
                    } ${
                      isDark
                        ? "bg-zinc-950 border-zinc-800 text-zinc-100"
                        : "bg-white border-gray-200 text-gray-900"
                    } ${focusRing}`}
                  />
                </li>
              ))}
            </ul>
          </div>
          </>
        )}
      </div>
    </div>
  );
}
