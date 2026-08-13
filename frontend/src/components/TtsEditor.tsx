import { useEffect, useRef } from "react";
import { Copy, Download, History, Loader2, Play, RefreshCw, Square } from "lucide-react";
import { focusRing } from "@/lib/theme";
import type { EngineLanguage, Voice } from "@/types/models";
import { textStats, fmtDuration, isRtlText, textDirection } from "@/lib/textStats";
import { DESIGN_CHIPS, NONVERBAL_TAGS, appendDesignChip, type OmniMode } from "@/lib/voiceModes";
import type { TtsHistoryItem } from "@/lib/ttsHistory";
import { useI18n } from "@/lib/i18n";
import { LanguageSelect } from "./LanguageSelect";

interface Props {
  isDark: boolean;
  text: string;
  onTextChange: (t: string) => void;
  activeVoice: Voice | null;
  languages: EngineLanguage[];
  showLanguage: boolean;          // false for built-in-voice engines (filter handled in library)
  language: string | null;
  onLanguageChange: (code: string) => void;
  // Engines with Clone/Design/Auto voice modes (OmniVoice, VoxCPM). `activeEngine`
  // drives engine-specific content (OmniVoice chips/tags vs VoxCPM placeholders).
  supportsVoiceModes: boolean;
  supportsStyleClone: boolean;
  supportsStylePrompt?: boolean;
  activeEngine: string | null;
  omniMode: OmniMode;
  onOmniModeChange: (m: OmniMode) => void;
  voiceDesign: string;
  onVoiceDesignChange: (v: string) => void;
  busy: boolean;
  isGenerating: boolean;
  isPlaying: boolean;
  /** True once a take exists for the current buffer. Gates Download. */
  hasAudio: boolean;
  onGenerate: () => void;
  onPlay: () => void;
  onDownload: () => void;
  /** Recently synthesized texts — click one to load it and re-generate with another voice. */
  history: TtsHistoryItem[];
  onUseText: (text: string) => void;
  onClearHistory: () => void;
}

export function TtsEditor(props: Props) {
  const {
    isDark, text, onTextChange, activeVoice, languages, showLanguage,
    language, onLanguageChange, supportsVoiceModes, supportsStyleClone, supportsStylePrompt = false, activeEngine, omniMode, onOmniModeChange,
    voiceDesign, onVoiceDesignChange, busy, isGenerating, isPlaying, hasAudio,
    onGenerate, onPlay, onDownload, history, onUseText, onClearHistory,
  } = props;
  const stats = textStats(text);
  const { t } = useI18n();
  const inputBg = isDark ? "bg-zinc-900 border-zinc-800 text-white" : "bg-white border-gray-200 text-gray-900";
  const selectBg = isDark ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-gray-300 text-gray-900";
  const sub = isDark ? "text-zinc-400" : "text-gray-600";

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pendingCaret = useRef<number | null>(null);

  // After a tag insertion changes the text, restore the caret just past the
  // inserted tag (the textarea is controlled, so we do this post-render).
  useEffect(() => {
    if (pendingCaret.current != null && textareaRef.current) {
      const pos = pendingCaret.current;
      textareaRef.current.focus();
      textareaRef.current.setSelectionRange(pos, pos);
      pendingCaret.current = null;
    }
  }, [text]);

  // Insert a non-verbal tag at the cursor (or replace the selection), padding
  // with spaces so it never glues to adjacent words.
  const insertTag = (tag: string) => {
    const el = textareaRef.current;
    const start = el?.selectionStart ?? text.length;
    const end = el?.selectionEnd ?? text.length;
    const before = text.slice(0, start);
    const after = text.slice(end);
    const lead = before.length > 0 && !/\s$/.test(before) ? " " : "";
    const trail = after.length === 0 || !/^\s/.test(after) ? " " : "";
    const chunk = `${lead}${tag}${trail}`;
    pendingCaret.current = before.length + chunk.length;
    onTextChange(before + chunk + after);
  };

  // Show the "Voice: X" note for any non-OmniVoice engine, and for OmniVoice
  // only in clone mode (design/auto carry no reference voice).
  const showVoiceNote = !supportsVoiceModes || omniMode === "clone";

  const segBtn = (m: OmniMode, label: string) => (
    <button
      type="button"
      onClick={() => onOmniModeChange(m)}
      className={`flex-1 px-3 py-1.5 text-sm font-medium rounded transition-colors ${
        omniMode === m
          ? "accent-gradient text-black"
          : isDark
            ? "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
            : "bg-gray-100 text-gray-600 hover:bg-gray-200"
      } ${focusRing}`}
    >
      {label}
    </button>
  );

  return (
    <div className="max-w-3xl mx-auto w-full space-y-3">
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        placeholder={t("tts.placeholder")}
        dir={textDirection(text)}
        className={`w-full min-h-[260px] rounded-xl border p-4 text-sm leading-relaxed focus:outline-none focus:border-indigo-500 ${inputBg} ${
          isRtlText(text) ? "text-right" : "text-left"
        }`}
      />

      {/* Text history — click to load, then pick another voice and Generate */}
      {history.length > 0 && (
        <div className={`rounded-xl border p-3 ${isDark ? "border-zinc-800 bg-zinc-900/50" : "border-gray-200 bg-gray-50"}`}>
          <div className="flex items-center justify-between mb-2">
            <span className={`text-xs font-semibold uppercase tracking-wide flex items-center gap-1.5 ${sub}`}>
              <History className="w-3.5 h-3.5" /> Text history
            </span>
            <button
              type="button"
              onClick={onClearHistory}
              className={`text-[11px] ${isDark ? "text-zinc-400 hover:text-zinc-200" : "text-gray-500 hover:text-gray-700"} ${focusRing}`}
            >
              Clear
            </button>
          </div>
          <ul className="space-y-1 max-h-48 overflow-y-auto">
            {history.map((h) => (
              <li
                key={`${h.at}-${h.text.slice(0, 16)}`}
                className="flex items-center gap-2"
              >
                <button
                  type="button"
                  onClick={() => onUseText(h.text)}
                  title={h.text}
                  className={`flex-1 min-w-0 flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs transition-colors ${
                    isDark
                      ? "text-zinc-300 hover:bg-zinc-800 hover:text-white"
                      : "text-gray-700 hover:bg-gray-200 hover:text-gray-900"
                  } ${focusRing}`}
                >
                  <span className="truncate flex-1">{h.text}</span>
                  <span className={`text-[10px] shrink-0 ${sub}`}>
                    {new Date(h.at).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => void navigator.clipboard.writeText(h.text)}
                  className={`p-1.5 rounded shrink-0 ${
                    isDark ? "text-zinc-400 hover:text-indigo-300" : "text-gray-500 hover:text-indigo-600"
                  } ${focusRing}`}
                  title="Copy text"
                >
                  <Copy className="w-3.5 h-3.5" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* OmniVoice inline non-verbal sounds — insert a tag at the cursor */}
      {activeEngine === "omnivoice" && (
        <div className="space-y-1.5">
          <div className={`text-xs ${sub}`}>
            Non-verbal sounds — click to insert at the cursor
          </div>
          <div className="flex flex-wrap gap-1">
            {NONVERBAL_TAGS.map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => insertTag(tag)}
                className={`px-1.5 py-0.5 text-[11px] font-mono rounded border transition-colors ${
                  isDark
                    ? "border-zinc-700 text-zinc-400 hover:border-indigo-500 hover:text-indigo-300"
                    : "border-gray-300 text-gray-600 hover:border-indigo-500 hover:text-indigo-600"
                } ${focusRing}`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Qwen always-available free-text style prompt (built-in voice + optional style) */}
      {supportsStylePrompt && (
        <div className={`rounded-xl border p-3 ${isDark ? "border-zinc-800 bg-zinc-900/50" : "border-gray-200 bg-gray-50"}`}>
          <input
            type="text"
            value={voiceDesign}
            onChange={(e) => onVoiceDesignChange(e.target.value)}
            placeholder={t("tts.style")} // Qwen style
            className={`w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${selectBg}`}
          />
        </div>
      )}

      {/* OmniVoice voice mode: Clone / Design / Auto */}
      {supportsVoiceModes && (
        <div className={`rounded-xl border p-3 space-y-2 ${isDark ? "border-zinc-800 bg-zinc-900/50" : "border-gray-200 bg-gray-50"}`}>
          <div className="flex gap-1.5">
            {segBtn("clone", "Clone")}
            {segBtn("design", "Design")}
            {segBtn("auto", "Auto")}
          </div>
          {omniMode === "clone" && (
            <div className="space-y-1.5">
              <p className={`text-xs ${sub}`}>
                Clones the voice selected in the library:{" "}
                <span className="text-indigo-300">{activeVoice ? activeVoice.name : t("tts.none")}</span>
              </p>
              {supportsStyleClone && (
                <input
                  type="text"
                  value={voiceDesign}
                  onChange={(e) => onVoiceDesignChange(e.target.value)}
                  placeholder={t("tts.style")}
                  className={`w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${selectBg}`}
                />
              )}
            </div>
          )}
          {omniMode === "design" && (
            <div className="space-y-1.5">
              <input
                type="text"
                value={voiceDesign}
                onChange={(e) => onVoiceDesignChange(e.target.value)}
                placeholder={activeEngine === "voxcpm" ? "e.g. a young woman, gentle and sweet" : "e.g. female, low pitch, british accent"}
                className={`w-full border rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-indigo-500 ${selectBg}`}
              />
              {activeEngine === "omnivoice" && (
                <div className="flex flex-wrap gap-1">
                  {DESIGN_CHIPS.map((chip) => (
                    <button
                      key={chip}
                      type="button"
                      onClick={() => onVoiceDesignChange(appendDesignChip(voiceDesign, chip))}
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
          {omniMode === "auto" && (
            <p className={`text-xs italic ${sub}`}>
              {activeEngine === "voxcpm"
                ? "VoxCPM will design a fresh voice for this text."
                : "OmniVoice will invent a voice for this text."}
            </p>
          )}
        </div>
      )}

      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className={`text-xs ${sub}`}>
          {t("tts.stats", { c: stats.chars, w: stats.words, s: fmtDuration(stats.seconds) })}
        </div>
        <div className="flex items-center gap-2">
          {showLanguage && (
            <LanguageSelect isDark={isDark} languages={languages} value={language} onChange={onLanguageChange} />
          )}
          {showVoiceNote && (
            <span className={`text-xs ${sub}`}>
              {t("tts.voice")}: <span className="text-indigo-300">{activeVoice ? activeVoice.name : t("tts.none")}</span>
            </span>
          )}
          <button type="button" onClick={onGenerate} disabled={busy || !text.trim()}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium accent-gradient hover:opacity-90 disabled:bg-zinc-700 disabled:bg-none disabled:text-zinc-400 text-black transition-opacity ${focusRing}`}>
            {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />} Generate
          </button>
          <button type="button" onClick={onPlay} disabled={busy && !isPlaying}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isPlaying
                ? "accent-gradient hover:opacity-90 text-black"
                : isDark ? "bg-zinc-800 hover:bg-zinc-700 text-white" : "bg-gray-100 hover:bg-gray-200 text-gray-900"} ${focusRing}`}>
            {isPlaying ? <><Square className="w-4 h-4" /> Stop</> : <><Play className="w-4 h-4" /> Play</>}
          </button>
          {/* Saves the take the browser already holds — no server round-trip, so
              it works even when the synthesis wasn't cached server-side. */}
          <button type="button" onClick={onDownload} disabled={!hasAudio}
            title={hasAudio ? "Download this take as WAV" : "Generate the audio first"}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              isDark ? "bg-zinc-800 hover:bg-zinc-700 text-white" : "bg-gray-100 hover:bg-gray-200 text-gray-900"} ${focusRing}`}>
            <Download className="w-4 h-4" /> Download
          </button>
        </div>
      </div>
    </div>
  );
}
