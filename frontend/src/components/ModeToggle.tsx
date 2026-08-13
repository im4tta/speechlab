import type { LucideIcon } from "lucide-react";
import { AudioLines, FileAudio, FileText, Mic2 } from "lucide-react";
import type { ProjectMode } from "@/types/models";
import { focusRing } from "@/lib/theme";
import { useI18n } from "@/lib/i18n";

interface Props {
  isDark: boolean;
  mode: ProjectMode;
  onChange: (m: ProjectMode) => void;
}

export function ModeToggle({ isDark, mode, onChange }: Props) {
  const { t } = useI18n();
  const wrap = isDark ? "bg-zinc-800" : "bg-gray-100";
  // Same icons as the ModeChooser cards so the toggle and chooser stay in sync.
  const seg = (m: ProjectMode, labelKey: string, Icon: LucideIcon) => (
    <button type="button" onClick={() => onChange(m)} title={t(labelKey)} aria-label={t(labelKey)}
      className={`flex items-center gap-1.5 px-2 py-1.5 text-xs lg:text-sm font-medium rounded-md transition-colors ${
        mode === m ? "accent-gradient text-black"
        : isDark ? "text-zinc-400 hover:text-zinc-200" : "text-gray-600 hover:text-gray-700"
      } ${focusRing}`}>
      <Icon className="w-4 h-4 shrink-0" />
      {t(labelKey)}
    </button>
  );
  return (
    <div className={`inline-flex gap-1 p-1 rounded-lg ${wrap}`}>
      {seg("tts", "mode.tts", FileText)}
      {seg("podcast", "mode.podcast", Mic2)}
      {seg("transcribe", "mode.transcribe", FileAudio)}
      {seg("dub", "mode.dub", AudioLines)}
    </div>
  );
}
