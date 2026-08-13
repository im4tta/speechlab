import { useEffect, useState } from "react";
import { Check, Download, Loader2, X } from "lucide-react";
import type { AsrStatus, TranscribeBuffer } from "@/types/models";
import { focusRing } from "@/lib/theme";
import { useI18n } from "@/lib/i18n";

interface Props {
  isDark: boolean;
  asr: AsrStatus | null;
  buffer: TranscribeBuffer;
  onChange: (partial: Partial<TranscribeBuffer>) => void;
  open: boolean;
  onClose: () => void;
  /** Activate a speech-to-text model (downloads it first if needed). */
  onActivateModel: (id: string) => Promise<{ downloading: boolean }>;
  /** Refetch /api/asr/status (used while polling a model download). */
  onRefreshAsr: () => Promise<void>;
}

/** Top-right popover with transcription settings + the ASR model picker. */
export function TranscribeSettingsPopup({
  isDark,
  asr,
  buffer,
  onChange,
  open,
  onClose,
  onActivateModel,
  onRefreshAsr,
}: Props) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const { t } = useI18n();

  const pick = async (id: string) => {
    if (busyId) return;
    setBusyId(id);
    try {
      const res = await onActivateModel(id);
      if (res.downloading) setDownloadingId(id);
      else await onRefreshAsr();
    } finally {
      setBusyId(null);
    }
  };

  // Poll while a model downloads, then activate it automatically.
  useEffect(() => {
    if (!downloadingId) return;
    let stopped = false;
    const t = setInterval(async () => {
      await onRefreshAsr();
      const m = asr?.models.find((x) => x.id === downloadingId);
      if (m?.downloaded) {
        clearInterval(t);
        if (!stopped) {
          setDownloadingId(null);
          await pick(downloadingId);
        }
      }
    }, 1500);
    return () => {
      stopped = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [downloadingId, asr, onRefreshAsr]);

  if (!open) return null;

  const heading = isDark ? "text-zinc-400" : "text-gray-600";
  const subtle = isDark ? "text-zinc-400" : "text-gray-600";
  const bodyText = isDark ? "text-zinc-300" : "text-gray-700";
  const label = isDark ? "text-zinc-300" : "text-gray-700";
  const inputBg = isDark
    ? "bg-zinc-900 border-zinc-800 text-white"
    : "bg-white border-gray-200 text-gray-900";
  const rowCls = (active: boolean) =>
    `w-full flex items-center gap-2 px-2.5 py-2 rounded-lg border text-left transition-colors ${
      active
        ? "border-indigo-500 bg-indigo-500/10"
        : isDark
          ? "border-zinc-800 bg-zinc-950/40 hover:border-indigo-500/50"
          : "border-gray-200 bg-gray-50 hover:border-indigo-500/50"
    }`;

  return (
    <>
      <div className="absolute inset-0 z-30" onClick={onClose} />
      <div
        className={`fixed top-14 right-4 z-40 w-80 max-h-[75vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden ${
          isDark ? "bg-zinc-900 border-zinc-700" : "bg-white border-gray-200"
        }`}
      >
        <div
          className={`flex items-center justify-between px-4 py-3 border-b ${
            isDark ? "border-zinc-800" : "border-gray-200"
          }`}
        >
          <h2 className={`text-sm font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
            {t("tr.title")}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className={`p-1 rounded ${isDark ? "text-zinc-400 hover:text-white" : "text-gray-600 hover:text-gray-900"} ${focusRing}`}
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* ASR model picker */}
          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              {t("tr.model")}
            </h3>
            {!asr ? (
              <p className={`text-xs ${subtle}`}>status unavailable</p>
            ) : (
              <ul className="space-y-1.5">
                {asr.models.map((m) => {
                  const active = m.active;
                  const isKhmer = m.best_for.includes("khmer");
                  const isDownloading = downloadingId === m.id;
                  const isBusy = busyId === m.id;
                  return (
                    <li key={m.id}>
                      <button
                        type="button"
                        onClick={() => void pick(m.id)}
                        disabled={active || !!busyId || isDownloading}
                        className={`${rowCls(active)} ${
                          active || !!busyId ? "cursor-default" : "cursor-pointer"
                        } ${focusRing}`}
                        title={m.description}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span
                              className={`text-xs font-medium truncate ${
                                isDark ? "text-zinc-200" : "text-gray-800"
                              }`}
                            >
                              {m.label}
                            </span>
                            {isKhmer && (
                              <span className="shrink-0 text-[9px] font-bold px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                                KH
                              </span>
                            )}
                          </div>
                          <div className={`text-[10px] mt-0.5 ${subtle}`}>
                            {m.size}
                            {isKhmer ? " · best for Khmer" : ""}
                          </div>
                        </div>
                        {active ? (
                          <Check className={`w-3.5 h-3.5 shrink-0 ${isDark ? "text-emerald-400" : "text-emerald-600"}`} />
                        ) : isDownloading || isBusy ? (
                          <Loader2 className={`w-3.5 h-3.5 shrink-0 animate-spin ${isDark ? "text-indigo-400" : "text-indigo-600"}`} />
                        ) : m.downloaded ? (
                          <span
                            className={`shrink-0 text-[10px] font-semibold ${
                              isDark ? "text-indigo-300" : "text-indigo-700"
                            }`}
                          >
                            Use
                          </span>
                        ) : (
                          <Download
                            className={`w-3.5 h-3.5 shrink-0 ${isDark ? "text-zinc-400" : "text-gray-500"}`}
                          />
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
            <p className={`text-[11px] mt-2 ${subtle}`}>
              Whisper is weak on Khmer. The <span className="font-semibold">KH</span> models are
              Khmer fine-tunes — far more accurate. "unspaced" emits raw Khmer without inserted
              word spaces.
            </p>
          </div>

          {/* Language */}
          <div>
            <label htmlFor="asr-language-popup" className={`block text-sm font-medium mb-1 ${label}`}>
              {t("tr.language")}
            </label>
            <select
              id="asr-language-popup"
              value={buffer.language ?? ""}
              onChange={(e) => onChange({ language: e.target.value || null })}
              className={`w-full rounded-lg border px-3 py-2 text-sm ${inputBg} ${focusRing}`}
            >
              <option value="">{t("tr.autoDetect")}</option>
              {(asr?.languages ?? []).map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <p className={`text-xs mt-1 ${subtle}`}>
              For Khmer, pick Khmer when the clip is short or accented.
            </p>
          </div>

          <div>
            <label className={`flex items-center gap-2 text-sm font-medium ${label}`}>
              <input
                type="checkbox"
                checked={buffer.timestamps}
                onChange={(e) => onChange({ timestamps: e.target.checked })}
                className={`accent-indigo-600 ${focusRing}`}
              />
              {t("tr.timestamps")}
            </label>
            <p className={`text-xs mt-1 ${subtle}`}>Required to export .srt / .vtt subtitles.</p>
          </div>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-1 ${heading}`}>
              {t("tr.modelStatus")}
            </h3>
            <div className={`text-xs ${subtle} space-y-1`}>
              <div className={bodyText}>{asr?.model_id ?? "—"}</div>
              <div>
                {asr == null
                  ? "status unavailable"
                  : !asr.downloaded
                    ? "weights not downloaded"
                    : asr.loaded
                      ? "loaded"
                      : "ready (loads on first use)"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
