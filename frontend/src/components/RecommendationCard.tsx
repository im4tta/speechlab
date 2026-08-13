import { useI18n } from "@/lib/i18n";
import { Sparkles } from "lucide-react";
import { useSystemStats } from "@/hooks/useSystemStats";
import type { EngineInfo } from "@/types/models";

// Curated per-engine hardware fit so the user can pick the right software for
// their GPU. `vramGb` is null when there's no NVIDIA GPU.
type Tone = "best" | "ok" | "slow" | "no";
interface Fit {
  badge: string;
  tone: Tone;
  reason: string;
}

const TONE_CLS: Record<Tone, { badge: string; row: string }> = {
  best: {
    badge: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    row: "border-emerald-500/25",
  },
  ok: {
    badge: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
    row: "border-zinc-800",
  },
  slow: {
    badge: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    row: "border-zinc-800",
  },
  no: {
    badge: "bg-red-500/10 text-red-400/80 border-red-500/20",
    row: "border-zinc-800 opacity-70",
  },
};

function fitFor(name: string, vramGb: number | null): Fit | null {
  const gpu = vramGb != null && vramGb > 0;
  const v = vramGb ?? 0;
  switch (name) {
    case "kokoro":
      return { badge: "Recommended", tone: "best", reason: "82M · fastest quality/speed, fits any GPU" };
    case "kitten":
      return { badge: "Recommended", tone: "best", reason: "~80M · tiny & instant, runs in-process" };
    case "vibevoice":
      return gpu && v < 4
        ? { badge: "Tight", tone: "ok", reason: "1.5B · needs ~3 GB VRAM — may just fit" }
        : { badge: "Fits", tone: "ok", reason: "1.5B · runs on your GPU" };
    case "chatterbox":
      return gpu && v >= 6
        ? { badge: "Fits", tone: "ok", reason: "multilingual · needs ~6 GB VRAM" }
        : { badge: "Needs 6 GB+", tone: "no", reason: gpu ? "too much VRAM for this GPU" : "no NVIDIA GPU" };
    case "omnivoice":
      return gpu && v >= 8
        ? { badge: "Fits", tone: "ok", reason: "voice cloning · needs ~8 GB VRAM" }
        : { badge: "Needs 8 GB+", tone: "no", reason: gpu ? "too much VRAM for this GPU" : "no NVIDIA GPU" };
    case "qwen":
      return gpu && v >= 8
        ? { badge: "Fits", tone: "ok", reason: "1.7B · needs ~8 GB VRAM" }
        : { badge: "Needs 8 GB+", tone: "no", reason: gpu ? "too much VRAM for this GPU" : "no NVIDIA GPU" };
    case "voxcpm":
      return gpu && v >= 8
        ? { badge: "Best for Khmer", tone: "best", reason: "2B · full GPU speed with your local model" }
        : gpu
          ? { badge: "CPU offload", tone: "slow", reason: "2B · runs on your GPU via CPU-offload — slower" }
          : { badge: "CPU only", tone: "slow", reason: "no CUDA — very slow" };
    case "nanovllm_km":
      return gpu && v >= 8
        ? { badge: "Best for Khmer", tone: "best", reason: "2B · batched, fastest on ≥8 GB GPUs" }
        : { badge: "Needs 8 GB+", tone: "no", reason: "2B model + KV cache don't fit your GPU" };
    default:
      return null;
  }
}

interface Props {
  isDark: boolean;
  engines: EngineInfo[];
}

export function RecommendationCard({ isDark, engines }: Props) {
  const { t } = useI18n();
  const stats = useSystemStats(true);
  const vramGb = stats?.vram?.total_bytes ? stats.vram.total_bytes / 1e9 : null;

  const rows = engines
    .map((e) => ({ engine: e, fit: fitFor(e.name, vramGb) }))
    .filter((r): r is { engine: EngineInfo; fit: Fit } => r.fit !== null)
    .sort((a, b) => {
      const order: Record<Tone, number> = { best: 0, ok: 1, slow: 2, no: 3 };
      return order[a.fit.tone] - order[b.fit.tone];
    });

  const title = vramGb
    ? t("rec.title.gpu", { n: Math.round(vramGb) })
    : t("rec.title.noGpu");

  return (
    <section
      className={`p-3 dark:bg-zinc-900 dark:border-zinc-800 bg-gray-100/80 border border-gray-200 rounded-lg ${
        isDark ? "" : "border-gray-200"
      }`}
    >
      <h3
        className={`text-xs font-semibold uppercase tracking-wide mb-2 flex items-center gap-1.5 ${
          isDark ? "text-zinc-400" : "text-gray-600"
        }`}
      >
        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
        {title}
      </h3>
      {!stats ? (
        <p className={`text-[11px] ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
          {t("rec.loading")}
        </p>
      ) : (
        <ul className="space-y-1.5">
          {rows.map(({ engine, fit }) => (
            <li
              key={engine.name}
              className={`p-2 rounded-lg border ${TONE_CLS[fit.tone].row} ${
                isDark ? "bg-zinc-950/40" : "bg-white"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className={`text-xs font-medium ${isDark ? "text-zinc-200" : "text-gray-800"}`}>
                  {engine.display_name}
                </span>
                <span
                  className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border whitespace-nowrap ${TONE_CLS[fit.tone].badge}`}
                >
                  {fit.badge}
                </span>
              </div>
              <p className={`text-[11px] mt-0.5 ${isDark ? "text-zinc-400" : "text-gray-600"}`}>
                {fit.reason}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
