import { useState } from "react";
import { FolderOpen } from "lucide-react";
import { focusRing } from "@/lib/theme";
import { useI18n } from "@/lib/i18n";
import { ApiError, chooseVoxcpmFolder, setVoxcpmModelPath } from "@/lib/api";

interface Props {
  isDark: boolean;
  /** Currently configured local VoxCPM2 folder (null = HuggingFace weights). */
  modelPath: string | null;
  /** Refetch /api/config after saving so the UI reflects the new value. */
  onRefreshConfig: () => Promise<void>;
}

export function VoxCpmModelPathCard({ isDark, modelPath, onRefreshConfig }: Props) {
  const { t } = useI18n();
  const [draft, setDraft] = useState(modelPath ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  const text = isDark ? "text-zinc-300" : "text-gray-700";
  const sub = isDark ? "text-zinc-400" : "text-gray-600";
  const inputCls = isDark
    ? "bg-zinc-800 border-zinc-700 text-white"
    : "bg-white border-gray-300 text-gray-900";

  const save = async (path: string | null) => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const res = await setVoxcpmModelPath(path);
      setSaved(res.path ?? "(cleared) — will use HuggingFace weights");
      setDraft(res.path ?? "");
      await onRefreshConfig();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const browse = async () => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const res = await chooseVoxcpmFolder();
      if (res.path) {
        setSaved(res.path);
        setDraft(res.path);
        await onRefreshConfig();
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-1.5 mt-4">
      <div className={`text-xs font-medium ${text}`}>{t("controls.localModel")}</div>
      <div className="flex gap-1">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="C:\models\VoxCPM2"
          className={`flex-1 min-w-0 border rounded-md px-2 py-1 text-xs focus:outline-none focus:border-indigo-500 ${inputCls}`}
        />
        <button
          type="button"
          onClick={() => void browse()}
          disabled={busy}
          title="Choose a folder on this computer"
          className={`px-2 rounded border transition-colors ${focusRing} ${
            isDark
              ? "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border-zinc-700"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200 border-gray-300"
          } ${busy ? "opacity-50 cursor-wait" : ""}`}
        >
          <FolderOpen className="w-4 h-4" />
        </button>
      </div>
      <div className="flex gap-1">
        <button
          type="button"
          onClick={() => void save(draft.trim() || null)}
          disabled={busy}
          className={`flex-1 text-xs font-medium px-2 py-1 rounded border transition-colors ${focusRing} ${
            isDark
              ? "bg-indigo-700/40 hover:bg-indigo-700/60 text-indigo-200 border-indigo-800"
              : "bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border-indigo-200"
          } ${busy ? "opacity-50 cursor-wait" : ""}`}
        >
          {t("vox.useThis")}
        </button>
        {modelPath && (
          <button
            type="button"
            onClick={() => void save(null)}
            disabled={busy}
            className={`px-2 text-xs rounded border transition-colors ${focusRing} ${
              isDark
                ? "bg-zinc-800 text-zinc-400 hover:text-zinc-200 border-zinc-700"
                : "bg-gray-100 text-gray-600 hover:text-gray-800 border-gray-300"
            } ${busy ? "opacity-50 cursor-wait" : ""}`}
          >
            Clear
          </button>
        )}
      </div>
      {modelPath && (
        <p className={`text-[11px] break-all ${sub}`}>Folder: {modelPath}</p>
      )}
      {saved && <p className={`text-[11px] ${isDark ? "text-emerald-400" : "text-emerald-700"}`}>{saved}</p>}
      {error && <p className={`text-[11px] ${isDark ? "text-red-400" : "text-red-700"}`}>{error}</p>}
      <p className={`text-[11px] ${sub}`}>
        Loads VoxCPM2 from this folder instead of HuggingFace. Applies after a
        backend restart; an empty folder shows a Download button that fetches
        the weights into it.
      </p>
    </div>
  );
}
