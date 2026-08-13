import { Loader2 } from "lucide-react";
import { useSynthProgress } from "@/hooks/useSynthProgress";

interface Props {
  isDark: boolean;
  /** True while a synthesis request is in flight. */
  enabled: boolean;
  /** Fallback engine label shown before the first progress snapshot lands. */
  engine: string | null;
}

function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toString().padStart(2, "0")}`;
}

/**
 * Floating live-progress card shown while generating: engine, a ticking
 * elapsed timer, a real progress bar (nanovllm chunk counts) or an
 * indeterminate shimmer, ETA, stage and a text preview.
 */
export function SynthProgressPanel({ isDark, enabled, engine }: Props) {
  const { snapshot, elapsed } = useSynthProgress(enabled);

  if (!enabled) return null;

  const pct = snapshot.percent;
  const hasPct = pct != null;
  const label = snapshot.engine || engine || "model";

  return (
    <div
      className={`fixed bottom-24 left-1/2 -translate-x-1/2 z-40 w-[min(92vw,30rem)] rounded-2xl border shadow-2xl px-5 py-4 backdrop-blur-xl ${
        isDark ? "bg-zinc-900/90 border-zinc-700" : "bg-white/95 border-gray-200"
      }`}
    >
      <div className="flex items-center gap-3">
        <Loader2 className={`w-4 h-4 animate-spin shrink-0 ${isDark ? "text-indigo-400" : "text-indigo-600"}`} />
        <div className="min-w-0 flex-1">
          <div className={`text-sm font-semibold truncate ${isDark ? "text-white" : "text-gray-900"}`}>
            Synthesizing with {label}
          </div>
          <div className={`text-xs mt-0.5 flex items-center gap-2 ${isDark ? "text-zinc-400" : "text-gray-600"}`}>
            <span className="tabular-nums shrink-0">{fmtTime(elapsed)} elapsed</span>
            {snapshot.stage && (
              <span className="min-w-0 flex-1 break-all" title={snapshot.stage}>
                {snapshot.stage}
              </span>
            )}
            {hasPct && snapshot.eta_sec != null && (
              <>
                <span>·</span>
                <span className="tabular-nums shrink-0">~{fmtTime(snapshot.eta_sec)} left</span>
              </>
            )}
          </div>
        </div>
        {hasPct && (
          <span className={`text-sm font-bold tabular-nums shrink-0 ${isDark ? "text-indigo-300" : "text-indigo-700"}`}>
            {Math.round(pct!)}%
          </span>
        )}
      </div>

      {/* Progress bar — real % when known, otherwise an indeterminate shimmer */}
      <div className="mt-3 h-2 rounded-full overflow-hidden relative" style={{ background: isDark ? "#26262a" : "#e6e6ea" }}>
        {hasPct ? (
          <div
            className="h-full accent-gradient transition-all duration-500"
            style={{ width: `${Math.min(100, pct!)}%` }}
          />
        ) : (
          <div
            className="absolute inset-y-0 w-1/3 accent-gradient"
            style={{ animation: "synth-indeterminate 1.2s ease-in-out infinite" }}
          />
        )}
      </div>

      {snapshot.text && (
        <p className={`mt-2.5 text-[11px] leading-relaxed truncate ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
          {snapshot.text}
        </p>
      )}

      <style>{`
        @keyframes synth-indeterminate {
          0% { left: -35%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}
