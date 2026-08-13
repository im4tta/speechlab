import { Check, Download, Trash2 } from "lucide-react";
import { focusRing } from "@/lib/theme";
import type { TranslateStatus } from "@/types/models";

interface Props {
  isDark: boolean;
  translate: TranslateStatus | null;
  onActivate: (name: string) => void;
  onDownload: (name: string) => void;
  onDelete: (name: string) => void;
}

/**
 * Translation-model manager, mirroring the Whisper card: pick the active
 * translator, download its weights, or delete them. Translators aren't TTS
 * engines, so this is its own small section rather than part of EngineSelector.
 */
export function TranslationCard({ isDark, translate, onActivate, onDownload, onDelete }: Props) {
  const sub = isDark ? "text-zinc-400" : "text-gray-600";
  const rowBg = isDark ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-gray-200";
  const btn = `px-2 py-1 rounded text-xs font-medium border transition-colors ${
    isDark
      ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border-zinc-700"
      : "bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
  } ${focusRing}`;

  if (!translate) {
    return <p className={`text-xs ${sub}`}>Loading translation models…</p>;
  }

  return (
    <div className="space-y-2">
      <p className={`text-xs ${sub}`}>Used for cross-language dubbing and translating transcripts.</p>
      {translate.models.map((m) => {
        const active = m.name === translate.active;
        return (
          <div key={m.name} className={`p-2 rounded-lg border ${rowBg}`}>
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className={`text-sm font-medium flex items-center gap-1.5 ${isDark ? "text-white" : "text-gray-900"}`}>
                  {m.display_name}
                  {active && <Check className="w-3.5 h-3.5 text-indigo-400" />}
                </div>
                <div className={`text-[11px] ${sub}`}>{m.license} · {m.languages.length} languages</div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {m.on_demand ? (
                  <>
                    <span className={`text-[11px] ${sub}`}>offline · installs on demand</span>
                    {!active && (
                      <button type="button" className={btn} onClick={() => onActivate(m.name)}>Use</button>
                    )}
                  </>
                ) : !m.downloaded ? (
                  <button type="button" className={btn} onClick={() => onDownload(m.name)}>
                    <span className="flex items-center gap-1"><Download className="w-3.5 h-3.5" /> Download</span>
                  </button>
                ) : (
                  <>
                    {!active && (
                      <button type="button" className={btn} onClick={() => onActivate(m.name)}>Use</button>
                    )}
                    <button type="button" className={btn} title="Delete weights" onClick={() => onDelete(m.name)}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
