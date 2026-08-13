import { useEffect, useState } from "react";
import { LayoutGrid, Maximize2, PanelRightClose, PanelRightOpen } from "lucide-react";
import { focusRing } from "@/lib/theme";
import { defaultControlPanelOpen } from "@/lib/layout";
import type { AsrStatus, EngineInfo, TranslateStatus } from "@/types/models";
import { useCacheData } from "./CachePanel";
import { FullScreenPage } from "./FullScreenPage";

const LS_KEY = "vs.controlPanel.open";

export interface ControlPanelProps {
  isDark: boolean;
  engines: EngineInfo[];
  /** Speech-to-text model, shown in the engine popup so its weights are
   *  manageable. Null while /api/asr/status is loading. */
  asr: AsrStatus | null;
  activeEngine: string | null;
  onSelectEngine: (name: string) => Promise<void>;
  onLoadEngine: (name: string) => Promise<void>;
  onInstallEngine: (name: string) => void;
  onDownloadEngine: (name: string) => void;
  onDeleteWeights: (name: string) => void;
  onUninstallEngine: (name: string) => void;
  /** Translation models (m2m100/madlad). Null while /api/translate/status loads. */
  translate: TranslateStatus | null;
  onActivateTranslator: (name: string) => void;
  cfgScale: number;
  onCfgScaleChange: (v: number) => void;
  exaggeration: number;
  onExaggerationChange: (v: number) => void;
  quality?: "fast" | "balanced" | "high";
  onQualityChange?: (q: "fast" | "balanced" | "high") => void;
  qwenParams?: { temperature: number; topP: number; topK: number; repetitionPenalty: number; seed: number | null };
  onQwenParamsChange?: (p: { temperature: number; topP: number; topK: number; repetitionPenalty: number; seed: number | null }) => void;
  qwenDefaults?: { temperature: number; topP: number; topK: number; repetitionPenalty: number; seed: number | null };
  /** Configured local VoxCPM2 model folder (null = HuggingFace weights). */
  voxcpmModelPath?: string | null;
  onRefreshConfig?: () => Promise<void>;
}

/**
 * Right-hand chrome: a slim, never-scrolling rail of shortcuts. All the actual
 * controls live on the full-page view (FullScreenPage), which this rail opens.
 * The old narrow scrollable panel was too cramped — the full page is the home
 * for the controls now.
 */
export function ControlPanel(props: ControlPanelProps) {
  const { isDark } = props;
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    const stored = localStorage.getItem(LS_KEY);
    if (stored !== null) return stored === "false";
    return typeof window !== "undefined"
      ? !defaultControlPanelOpen(window.innerWidth)
      : false;
  });

  const { data: cacheData, busy: cacheBusy, refresh: cacheRefresh, onClear: onCacheClear, onDelete: onCacheDelete } = useCacheData();
  const [fullscreenTab, setFullscreenTab] = useState<"control" | "recent" | null>(null);

  useEffect(() => {
    localStorage.setItem(LS_KEY, collapsed ? "false" : "true");
  }, [collapsed]);

  const surface = isDark
    ? "bg-zinc-950/60 backdrop-blur-xl"
    : "bg-white/70 backdrop-blur-xl";
  const border = isDark ? "border-zinc-800" : "border-gray-200";
  const iconBtn = isDark
    ? "text-zinc-400 hover:text-indigo-400"
    : "text-gray-600 hover:text-indigo-600";

  const railCls = `w-12 shrink-0 border-l flex flex-col items-center pt-3 gap-2 transition-colors ${surface} ${border}`;

  const shared = {
    cacheData,
    cacheBusy,
    onCacheClear: onCacheClear,
    onCacheDelete: onCacheDelete,
    onCacheRefresh: () => void cacheRefresh(),
  };

  return (
    <>
      <aside className={railCls}>
        {collapsed ? (
          <button
            type="button"
            onClick={() => setCollapsed(false)}
            className={`p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
            title="Open control rail"
            aria-label="Open control rail"
          >
            <PanelRightOpen className="w-4 h-4" />
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={() => setFullscreenTab("control")}
              className={`p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
              title="Open controls full page"
              aria-label="Open controls full page"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setFullscreenTab("recent")}
              className={`relative p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
              title="Recent generations"
              aria-label="Recent generations"
            >
              <LayoutGrid className="w-4 h-4" />
              {!!cacheData && cacheData.entry_count > 0 && (
                <span
                  className="absolute -top-0.5 -right-0.5 min-w-3.5 h-3.5 px-0.5 flex items-center justify-center rounded-full accent-gradient text-black text-[9px] font-bold"
                >
                  {cacheData.entry_count > 99 ? "99+" : cacheData.entry_count}
                </span>
              )}
            </button>
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className={`p-2 rounded-lg transition-colors ${iconBtn} ${focusRing}`}
              title="Hide control rail"
              aria-label="Hide control rail"
            >
              <PanelRightClose className="w-4 h-4" />
            </button>
          </>
        )}
      </aside>

      {fullscreenTab && (
        <FullScreenPage
          {...props}
          {...shared}
          tab={fullscreenTab}
          onTabChange={setFullscreenTab}
          onClose={() => setFullscreenTab(null)}
        />
      )}
    </>
  );
}
