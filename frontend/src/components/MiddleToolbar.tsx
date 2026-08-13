import { Plus, RefreshCw } from "lucide-react";
import { SampleMenu } from "./SampleMenu";
import { ImportExportMenu } from "./ImportExportMenu";
import type { Sample, TtsSample } from "@/lib/samples";
import type { ProjectMode } from "@/types/models";
import { focusRing } from "@/lib/theme";

interface Props {
  validCount: number;
  cachedCount: number;
  busy: boolean;
  isDark: boolean;
  mode: ProjectMode | null;
  onAddSegment: () => void;
  onGenerateAll: () => void;
  onExportJson: () => void;
  onImportJson: (file: File) => void;
  onLoadPodcastSample: (sample: Sample) => void;
  onLoadTtsSample: (sample: TtsSample) => void;
  onExportSubtitles: () => void;
  subtitlesDisabled: boolean;
}

export function MiddleToolbar({
  validCount,
  cachedCount,
  busy,
  isDark,
  mode,
  onAddSegment,
  onGenerateAll,
  onExportJson,
  onImportJson,
  onLoadPodcastSample,
  onLoadTtsSample,
  onExportSubtitles,
  subtitlesDisabled,
}: Props) {
  const generateDisabled = busy || cachedCount === validCount;
  const isPodcast = mode === "podcast";

  return (
    <div
      className={`flex items-center justify-between gap-2 @[1200px]:gap-3 p-2.5 @[1200px]:p-2.5 border-b ${
        isDark
          ? "bg-zinc-950/60 backdrop-blur-xl border-zinc-800"
          : "bg-white/70 backdrop-blur-xl border-gray-200"
      }`}
    >
      <div className="flex items-center gap-3">
        {isPodcast && (
          <button
            type="button"
            onClick={onAddSegment}
            disabled={busy}
            title="Add a new segment"
            className={`flex items-center gap-1.5 px-3 py-2 accent-gradient text-black hover:opacity-90 disabled:bg-zinc-700 disabled:bg-none disabled:text-zinc-400 rounded-lg font-medium text-sm transition-opacity disabled:cursor-not-allowed ${focusRing}`}
          >
            <Plus className="w-4 h-4" />
            <span className="hidden @[1100px]:inline">Add Segment</span>
          </button>
        )}
      </div>

      <div className="flex items-center gap-2">
        {isPodcast && (
          <button
            type="button"
            onClick={onGenerateAll}
            disabled={generateDisabled}
            title="Generate all uncached segments"
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg font-medium text-sm transition-opacity disabled:cursor-not-allowed ${
              generateDisabled
                ? isDark
                  ? "bg-zinc-800 text-zinc-400"
                  : "bg-gray-100 text-gray-600"
                : "accent-gradient text-black hover:opacity-90"
            } ${focusRing}`}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span className="hidden @[1100px]:inline">Generate All</span>
            {validCount > 0 && (
              <span
                className={`text-xs ml-1 ${
                  cachedCount === validCount ? "text-indigo-100" : "text-white"
                }`}
              >
                {cachedCount}/{validCount}
              </span>
            )}
          </button>
        )}

        <ImportExportMenu
          isDark={isDark}
          busy={busy}
          onExportJson={onExportJson}
          onImportJson={onImportJson}
          // Transcribe/Dub modes have no project-cached audio to subtitle here
          // (Dub renders + downloads its WAV inside DubEditor).
          onExportSubtitles={mode === "transcribe" || mode === "dub" ? undefined : onExportSubtitles}
          subtitlesDisabled={subtitlesDisabled}
        />

        {/* Transcribe mode has no sample scripts — its input is an audio file. */}
        {(mode === "podcast" || mode === "tts") && (
          <SampleMenu
            isDark={isDark}
            mode={mode}
            onLoadPodcast={onLoadPodcastSample}
            onLoadTts={onLoadTtsSample}
          />
        )}
      </div>
    </div>
  );
}
