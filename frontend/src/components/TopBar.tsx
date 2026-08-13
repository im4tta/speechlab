import { useState } from "react";
import { Activity, CircleHelp, LayoutGrid, Maximize2, Mic2, Moon, Settings2, Sun } from "lucide-react";
import { focusRing } from "@/lib/theme";
import type {
  AsrStatus,
  ProjectMode,
  TranscribeBuffer,
  Voice,
  VoiceMetadata,
} from "@/types/models";
import { ModeToggle } from "./ModeToggle";
import { StudioLogo } from "./StudioLogo";
import { TranscribeSettingsPopup } from "./TranscribeSettingsPopup";
import { VoicePicker } from "./VoicePicker";
import { useI18n } from "@/lib/i18n";

interface Props {
  isDark: boolean;
  onThemeToggle: () => void;
  mode: ProjectMode | null;
  onModeChange: (m: ProjectMode) => void;
  /** Active engine id (e.g. "voxcpm"). */
  engine: string | null;
  /** Active engine display name (e.g. "VoxCPM2"). */
  engineLabel: string | null;
  /** Reported inference device, e.g. "cpu". */
  device: string | null;
  /** True when the active engine's model is loaded. */
  loaded: boolean;
  onOpenControls: () => void;
  onOpenRecent: () => void;
  monitorOpen: boolean;
  onToggleMonitor: (open: boolean) => void;
  onOpenCredits: () => void;
  version?: string | null;

  // Voice picker (top-right popover)
  voices: Voice[];
  onUploadVoice: (file: File, meta: VoiceMetadata) => Promise<unknown>;
  onRemoveVoice: (id: string) => Promise<void>;
  onUpdateVoiceMeta: (voiceId: string, meta: VoiceMetadata) => Promise<unknown>;
  supportsVoiceCloning: boolean;
  selectedVoiceId?: string | null;
  onSelectVoice?: (voiceId: string) => void;

  // Transcribe settings popover
  asr: AsrStatus | null;
  transcribeBuffer?: TranscribeBuffer | null;
  onTranscribeChange?: (partial: Partial<TranscribeBuffer>) => void;
  onActivateAsrModel?: (id: string) => Promise<{ downloading: boolean }>;
  onRefreshAsr?: () => Promise<void>;
}

function iconBtn(isDark: boolean, active = false): string {
  return active
    ? "accent-gradient text-black"
    : isDark
      ? "text-zinc-400 hover:text-white hover:bg-zinc-800"
      : "text-gray-600 hover:text-gray-900 hover:bg-gray-100";
}

/**
 * Top header bar — the single chrome for navigation (mode), Appearance/theme,
 * backend status, and the widget/overlay toggles. Mirrors the voxkhtts
 * SpeechLab header layout: logo left, mode nav in the middle, controls right.
 */
export function TopBar({
  isDark,
  onThemeToggle,
  mode,
  onModeChange,
  engine,
  engineLabel,
  device,
  loaded,
  onOpenControls,
  onOpenRecent,
  monitorOpen,
  onToggleMonitor,
  onOpenCredits,
  version,
  voices,
  onUploadVoice,
  onRemoveVoice,
  onUpdateVoiceMeta,
  supportsVoiceCloning,
  selectedVoiceId,
  onSelectVoice,
  asr,
  transcribeBuffer,
  onTranscribeChange,
  onActivateAsrModel,
  onRefreshAsr,
}: Props) {
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [transcribeOpen, setTranscribeOpen] = useState(false);
  const { t, lang, setLang } = useI18n();

  const selectedVoice = voices.find((v) => v.id === selectedVoiceId) ?? null;
  const isTranscribe = mode === "transcribe";

  return (
    <header
      className={`shrink-0 border-b flex items-center gap-3 px-4 py-2 ${
        isDark ? "bg-zinc-950/70 backdrop-blur-xl border-zinc-800" : "bg-white/70 backdrop-blur-xl border-gray-200"
      }`}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 shrink-0">
        <StudioLogo size={34} />
        <div className="leading-tight">
          <div className={`text-sm font-bold ${isDark ? "text-white" : "text-gray-900"}`}>
            SpeechLab
          </div>
          <div className={`text-[10px] tabular-nums ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
            {version ? `v${version}` : ""} · {t("brand.tagline")}
          </div>
        </div>
      </div>

      {/* Mode nav */}
      <div className="flex-1 flex justify-center min-w-0">
        {mode !== null && (
          <ModeToggle isDark={isDark} mode={mode} onChange={onModeChange} />
        )}
      </div>

      {/* Right cluster: backend status + toggles */}
      <div className="flex items-center gap-1.5 shrink-0">
        {/* Voice picker */}
        {(mode === "tts" || mode === "dub") && (
          <button
            type="button"
            onClick={() => setVoiceOpen((o) => !o)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${iconBtn(
              isDark,
              voiceOpen,
            )} ${focusRing}`}
            title={t("header.voices")}
          >
            <Mic2 className="w-4 h-4" />
            <span className="hidden xl:inline max-w-28 truncate">
              {selectedVoice?.name ?? "Voices"}
            </span>
          </button>
        )}

        {/* Transcribe settings */}
        {isTranscribe && (
          <button
            type="button"
            onClick={() => setTranscribeOpen((o) => !o)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${iconBtn(
              isDark,
              transcribeOpen,
            )} ${focusRing}`}
            title={t("header.settings")}
          >
            <Settings2 className="w-4 h-4" />
            <span className="hidden lg:inline">{t("header.settings")}</span>
          </button>
        )}

        {/* Backend status */}
        <button
          type="button"
          onClick={onOpenControls}
          className={`hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors ${
            isDark
              ? "bg-zinc-900/80 border-zinc-800 text-zinc-300 hover:text-white"
              : "bg-white/80 border-gray-200 text-gray-700 hover:text-gray-900"
          } ${focusRing}`}
          title="Backend status — click to open controls"
        >
          <span
            className={`w-2 h-2 rounded-full ${loaded ? "bg-emerald-500" : "bg-amber-500"}`}
            style={loaded ? {} : { animation: "pulse 1.4s ease-in-out infinite" }}
          />
          <span className="text-xs font-medium">{engineLabel || engine || "backend"}</span>
          {device && (
            <span className={`text-[10px] uppercase ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
              {device}
            </span>
          )}
        </button>

        {/* System monitor */}
        <button
          type="button"
          onClick={() => onToggleMonitor(!monitorOpen)}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${iconBtn(
            isDark,
            monitorOpen,
          )} ${focusRing}`}
          title={t("header.monitor")}
        >
          <Activity className="w-4 h-4" />
          <span className="hidden lg:inline">{t("header.monitor")}</span>
        </button>

        {/* Controls full page */}
        <button
          type="button"
          onClick={onOpenControls}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${iconBtn(
            isDark,
          )} ${focusRing}`}
          title={t("header.controls")}
        >
          <Maximize2 className="w-4 h-4" />
          <span className="hidden lg:inline">{t("header.controls")}</span>
        </button>

        {/* Recent generations */}
        <button
          type="button"
          onClick={onOpenRecent}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${iconBtn(
            isDark,
          )} ${focusRing}`}
          title={t("header.recent")}
        >
          <LayoutGrid className="w-4 h-4" />
          <span className="hidden lg:inline">{t("header.recent")}</span>
        </button>

        {/* Credits / about */}
        <button
          type="button"
          onClick={onOpenCredits}
          className={`p-2 rounded-lg transition-colors ${iconBtn(isDark)} ${focusRing}`}
          title="Credits & acknowledgements"
          aria-label="Credits"
        >
          <CircleHelp className="w-4 h-4" />
        </button>

        {/* Language toggle */}
        <button
          type="button"
          onClick={() => setLang(lang === "km" ? "en" : "km")}
          className={`flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs font-bold transition-colors ${iconBtn(isDark)} ${focusRing}`}
          title={lang === "km" ? "Switch to English" : "ប្តូរទៅជាភាសាខ្មែរ"}
          aria-label="Toggle language"
        >
          {lang === "km" ? "EN" : "ខ្មែរ"}
        </button>

        {/* Appearance / theme */}
        <button
          type="button"
          onClick={onThemeToggle}
          className={`p-2 rounded-lg transition-colors ${iconBtn(isDark)} ${focusRing}`}
          title={t("header.theme")}
          aria-label="Toggle appearance"
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>

      {/* Top-right popovers */}
      <VoicePicker
        isDark={isDark}
        voices={voices}
        onUploadVoice={onUploadVoice}
        onRemoveVoice={onRemoveVoice}
        onUpdateVoiceMeta={onUpdateVoiceMeta}
        supportsVoiceCloning={supportsVoiceCloning}
        selectedVoiceId={selectedVoiceId}
        onSelectVoice={onSelectVoice}
        open={voiceOpen}
        onClose={() => setVoiceOpen(false)}
      />
      {isTranscribe && transcribeBuffer && onTranscribeChange && onActivateAsrModel && onRefreshAsr && (
        <TranscribeSettingsPopup
          isDark={isDark}
          asr={asr}
          buffer={transcribeBuffer}
          onChange={onTranscribeChange}
          open={transcribeOpen}
          onClose={() => setTranscribeOpen(false)}
          onActivateModel={onActivateAsrModel}
          onRefreshAsr={onRefreshAsr}
        />
      )}
    </header>
  );
}
