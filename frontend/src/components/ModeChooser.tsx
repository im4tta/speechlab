import { Mic2, FileText, FileAudio, AudioLines } from "lucide-react";
import type { ProjectMode } from "@/types/models";
import { focusRing } from "@/lib/theme";
import { useI18n } from "@/lib/i18n";

interface Props {
  isDark: boolean;
  onPick: (m: ProjectMode) => void;
}

export function ModeChooser({ isDark, onPick }: Props) {
  const { t } = useI18n();
  const card = isDark
    ? "bg-zinc-900 border-zinc-800 hover:border-indigo-500"
    : "bg-white border-gray-200 hover:border-indigo-500";
  const title = isDark ? "text-white" : "text-gray-900";
  const sub = isDark ? "text-zinc-400" : "text-gray-600";
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl w-full">
        <button type="button" onClick={() => onPick("tts")}
          className={`text-left p-6 rounded-xl border transition-colors ${card} ${focusRing}`}>
          <FileText className="w-8 h-8 text-indigo-400 mb-3" />
          <div className={`font-semibold ${title}`}>{t("mode.tts")}</div>
          <p className={`text-sm mt-1 ${sub}`}>{t("chooser.ttsDesc")}</p>
        </button>
        <button type="button" onClick={() => onPick("podcast")}
          className={`text-left p-6 rounded-xl border transition-colors ${card} ${focusRing}`}>
          <Mic2 className="w-8 h-8 text-indigo-400 mb-3" />
          <div className={`font-semibold ${title}`}>{t("mode.podcast")}</div>
          <p className={`text-sm mt-1 ${sub}`}>{t("chooser.podcastDesc")}</p>
        </button>
        <button type="button" onClick={() => onPick("transcribe")}
          className={`text-left p-6 rounded-xl border transition-colors ${card} ${focusRing}`}>
          <FileAudio className="w-8 h-8 text-indigo-400 mb-3" />
          <div className={`font-semibold ${title}`}>{t("mode.transcribe")}</div>
          <p className={`text-sm mt-1 ${sub}`}>{t("chooser.transcribeDesc")}</p>
        </button>
        <button type="button" onClick={() => onPick("dub")}
          className={`text-left p-6 rounded-xl border transition-colors ${card} ${focusRing}`}>
          <AudioLines className="w-8 h-8 text-indigo-400 mb-3" />
          <div className={`font-semibold ${title}`}>{t("mode.dub")}</div>
          <p className={`text-sm mt-1 ${sub}`}>{t("chooser.dubDesc")}</p>
        </button>
      </div>
    </div>
  );
}
