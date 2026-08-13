import { PanelLeftClose, PanelLeftOpen, Moon, Sun } from "lucide-react";
import { focusRing } from "@/lib/theme";
import { StudioLogo } from "./StudioLogo";

// The left column's chrome - logo, product name + version, and the collapse
// control. Shared by every left panel (VoiceLibrary, TranscribeControls)
// so they can't drift apart.

function iconBtnCls(isDark: boolean): string {
  return isDark
    ? "text-zinc-400 hover:text-white hover:bg-zinc-800"
    : "text-gray-600 hover:text-gray-900 hover:bg-gray-100";
}

interface HeaderProps {
  isDark: boolean;
  /** Rendered as `v{version}`; em dash while config is still loading. */
  version: string | null | undefined;
  onCollapse: () => void;
  collapseTitle: string;
}

export function SidebarHeader({ isDark, version, onCollapse, collapseTitle }: HeaderProps) {
  const border = isDark ? "border-zinc-800" : "border-gray-200";
  const heading = isDark ? "text-zinc-400" : "text-gray-600";
  const iconBtn = iconBtnCls(isDark);

  return (
    <div className={`p-3 xxl:p-4 border-b flex items-center gap-3 ${border}`}>
      <StudioLogo size={36} />
      <div className="min-w-0 flex-1">
        <h1 className={`font-semibold text-sm truncate ${isDark ? "text-white" : "text-gray-900"}`}>
          SpeechLab
        </h1>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className={`text-xs tabular-nums ${heading}`}>v{version ?? "-"}</span>
        </div>
      </div>
      <button
        type="button"
        onClick={onCollapse}
        className={`p-1 rounded transition-colors ${iconBtn} ${focusRing}`}
        title={collapseTitle}
      >
        <PanelLeftClose className="w-4 h-4" />
      </button>
    </div>
  );
}

interface StripProps {
  isDark: boolean;
  onOpen: () => void;
  openTitle: string;
  onThemeToggle: () => void;
}

/** The collapsed left column: logo, re-open control, theme toggle. */
export function SidebarStrip({ isDark, onOpen, openTitle, onThemeToggle }: StripProps) {
  const surface = isDark ? "bg-zinc-950" : "bg-white";
  const border = isDark ? "border-zinc-800" : "border-gray-200";
  const iconBtn = iconBtnCls(isDark);

  return (
    <aside
      className={`w-12 shrink-0 z-10 border-r flex flex-col items-center pt-4 gap-3 transition-colors ${surface} ${border}`}
    >
      <StudioLogo size={36} />
      <button
        type="button"
        onClick={onOpen}
        className={`p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
        title={openTitle}
      >
        <PanelLeftOpen className="w-5 h-5" />
      </button>
      <button
        type="button"
        onClick={onThemeToggle}
        className={`p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
        title="Toggle theme"
      >
        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </button>
    </aside>
  );
}
