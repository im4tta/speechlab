import { getCfgHints } from "@/lib/engineHints";
import { focusRing } from "@/lib/theme";
import type { CacheListResponse } from "@/lib/api";
import type { ControlPanelProps } from "./ControlPanel";
import { CacheBody } from "./CachePanel";
import { CfgScaleBody } from "./CfgScaleControl";
import { EngineSelector } from "./EngineSelector";
import { ExaggerationBody } from "./ExaggerationControl";
import { RecommendationCard } from "./RecommendationCard";
import { TranslationCard } from "./TranslationCard";
import { VoxCpmModelPathCard } from "./VoxCpmModelPathCard";

export interface ControlSectionsProps extends ControlPanelProps {
  /** Full-width grid layout (fullscreen page) vs the narrow column stack. */
  wide?: boolean;
  cacheData: CacheListResponse | null;
  cacheBusy: boolean;
  onCacheClear: () => void;
  onCacheDelete: (hash: string) => void;
  onCacheRefresh: () => void;
  /** When set, the Recent generations list gains an "open full page" button. */
  onOpenRecent?: () => void;
}

/**
 * Every control section (recommendations, engine, translation, settings,
 * local VoxCPM folder, recent generations). Shared by the narrow right-hand
 * ControlPanel and the full-screen Controls page so they can't drift apart.
 */
export function ControlSections({
  isDark,
  wide,
  engines,
  asr,
  activeEngine,
  onSelectEngine,
  onLoadEngine,
  onInstallEngine,
  onDownloadEngine,
  onDeleteWeights,
  onUninstallEngine,
  translate,
  onActivateTranslator,
  cfgScale,
  onCfgScaleChange,
  exaggeration,
  onExaggerationChange,
  quality,
  onQualityChange,
  qwenParams,
  onQwenParamsChange,
  qwenDefaults,
  voxcpmModelPath,
  onRefreshConfig,
  cacheData,
  cacheBusy,
  onCacheClear,
  onCacheDelete,
  onCacheRefresh,
  onOpenRecent,
}: ControlSectionsProps) {
  const heading = isDark ? "text-zinc-400" : "text-gray-600";
  const cfgHints = getCfgHints(activeEngine);
  const isChatterbox = activeEngine === "chatterbox";
  const isVoxCpm2 = activeEngine === "voxcpm" || activeEngine === "nanovllm_km";
  const wrap = wide ? "grid gap-4 xl:grid-cols-2" : "space-y-4";
  const span2 = wide ? "xl:col-span-2" : "";
  const sectionCls =
    "p-3 dark:bg-zinc-900 dark:border-zinc-800 bg-gray-100/80 border border-gray-200 rounded-lg";

  return (
    <div className={wrap}>
      {/* Hardware-aware recommendations */}
      <div className={span2}>
        <RecommendationCard isDark={isDark} engines={engines} />
      </div>

      {/* Engine section */}
      <section className={sectionCls}>
        <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
          Engine
        </h3>
        <EngineSelector
          isDark={isDark}
          engines={engines}
          asr={asr}
          activeName={activeEngine}
          onSelect={onSelectEngine}
          onLoad={onLoadEngine}
          onInstall={onInstallEngine}
          onDownload={onDownloadEngine}
          onDeleteWeights={onDeleteWeights}
          onUninstall={onUninstallEngine}
        />
      </section>

      {/* Translation section */}
      <section className={sectionCls}>
        <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
          Translation
        </h3>
        <TranslationCard
          isDark={isDark}
          translate={translate}
          onActivate={onActivateTranslator}
          onDownload={onDownloadEngine}
          onDelete={onDeleteWeights}
        />
      </section>

      {/* Settings section */}
      <section className={`${sectionCls} ${span2}`}>
        <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
          {isChatterbox ? "CFG weight (voice fidelity)" : "Voice fidelity (CFG)"}
        </h3>
        <CfgScaleBody
          isDark={isDark}
          value={cfgScale}
          onChange={onCfgScaleChange}
          hints={cfgHints}
        />

        {isChatterbox && (
          <>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 mt-4 ${heading}`}>
              Voice expressiveness (Chatterbox)
            </h3>
            <ExaggerationBody
              isDark={isDark}
              value={exaggeration}
              onChange={onExaggerationChange}
            />
          </>
        )}

        {activeEngine === "voxcpm" && onQualityChange && (
          <div className="space-y-1.5 mt-4">
            <div className={`text-xs font-medium ${isDark ? "text-zinc-300" : "text-gray-700"}`}>
              Quality
            </div>
            <div className="flex gap-1">
              {(["fast", "balanced", "high"] as const).map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => onQualityChange(q)}
                  className={`flex-1 px-2 py-1.5 text-xs font-medium rounded border transition-colors ${
                    (quality ?? "balanced") === q
                      ? "accent-gradient text-black border-indigo-500 hover:opacity-90"
                      : isDark
                        ? "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border-zinc-700"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200 border-gray-300"
                  } ${focusRing}`}
                >
                  {q[0].toUpperCase() + q.slice(1)}
                </button>
              ))}
            </div>
            <p className={`text-[11px] ${isDark ? "text-zinc-400" : "text-gray-600"}`}>
              Diffusion steps: Fast 5 · Balanced 10 · High 25. Higher = better quality, slower.
            </p>
          </div>
        )}

        {activeEngine === "qwen" && qwenParams && onQwenParamsChange && (
          <div className="space-y-2 mt-4">
            <div className={`text-xs font-medium ${isDark ? "text-zinc-300" : "text-gray-700"}`}>
              Advanced generation
            </div>
            {([
              { key: "temperature", label: "Temperature", min: 0.1, max: 2.0, step: 0.05 },
              { key: "topP", label: "Top-p", min: 0.0, max: 1.0, step: 0.05 },
              { key: "topK", label: "Top-k", min: 0, max: 200, step: 1 },
              { key: "repetitionPenalty", label: "Repetition penalty", min: 1.0, max: 2.0, step: 0.05 },
            ] as const).map((f) => (
              <label key={f.key} className={`block text-[11px] ${isDark ? "text-zinc-400" : "text-gray-600"}`}>
                <span className="flex justify-between"><span>{f.label}</span><span>{qwenParams[f.key]}</span></span>
                <input
                  type="range" min={f.min} max={f.max} step={f.step}
                  value={qwenParams[f.key]}
                  onChange={(e) => onQwenParamsChange({ ...qwenParams, [f.key]: Number(e.target.value) })}
                  className="w-full accent-indigo-600"
                />
              </label>
            ))}
            <label className={`block text-[11px] ${isDark ? "text-zinc-400" : "text-gray-600"}`}>
              Seed (optional)
              <input
                type="number"
                value={qwenParams.seed ?? ""}
                onChange={(e) => onQwenParamsChange({ ...qwenParams, seed: e.target.value === "" ? null : Number(e.target.value) })}
                placeholder="random"
                className={`mt-1 w-full border rounded-md px-2 py-1 text-xs focus:outline-none focus:border-indigo-500 ${
                  isDark ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-gray-300 text-gray-900"
                }`}
              />
            </label>
            {qwenDefaults && (
              <button
                type="button"
                onClick={() => onQwenParamsChange(qwenDefaults)}
                className={`text-[11px] underline ${isDark ? "text-zinc-400 hover:text-indigo-400" : "text-gray-600 hover:text-indigo-600"} ${focusRing}`}
              >
                Reset to defaults
              </button>
            )}
          </div>
        )}

        {isVoxCpm2 && (
          <VoxCpmModelPathCard
            isDark={isDark}
            modelPath={voxcpmModelPath ?? null}
            onRefreshConfig={async () => {
              await onRefreshConfig?.();
            }}
          />
        )}
      </section>

      {/* Recent generations section (CacheBody renders its own heading + actions) */}
      <section className={`${sectionCls} ${span2}`}>
        <CacheBody
          isDark={isDark}
          data={cacheData}
          busy={cacheBusy}
          onClear={onCacheClear}
          onDelete={onCacheDelete}
          onExpand={onOpenRecent}
        />
        <button
          type="button"
          onClick={onCacheRefresh}
          disabled={cacheBusy}
          className={`w-full text-xs font-medium py-1 rounded mt-1 ${
            isDark
              ? "text-zinc-400 hover:text-zinc-200"
              : "text-gray-600 hover:text-gray-700"
          } ${focusRing}`}
        >
          Refresh list
        </button>
      </section>
    </div>
  );
}
