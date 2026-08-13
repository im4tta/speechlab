import { useState } from "react";
import { Mic2, Pencil, Plus, Trash2, Volume2, X } from "lucide-react";
import type { Voice, VoiceMetadata } from "@/types/models";
import { UploadVoiceDialog } from "./UploadVoiceDialog";
import { VoiceMetaDialog } from "./VoiceMetaDialog";
import { useConfirm } from "./ConfirmProvider";
import { focusRing } from "@/lib/theme";
import { useI18n } from "@/lib/i18n";

export interface VoicePickerProps {
  isDark: boolean;
  voices: Voice[];
  onUploadVoice: (file: File, meta: VoiceMetadata) => Promise<unknown>;
  onRemoveVoice: (id: string) => Promise<void>;
  onUpdateVoiceMeta: (voiceId: string, meta: VoiceMetadata) => Promise<unknown>;
  supportsVoiceCloning: boolean;
  selectedVoiceId?: string | null;
  onSelectVoice?: (voiceId: string) => void;
  open: boolean;
  onClose: () => void;
}

/** Top-right popover with the voice library (built-ins + your uploads). */
export function VoicePicker({
  isDark,
  voices,
  onUploadVoice,
  onRemoveVoice,
  onUpdateVoiceMeta,
  supportsVoiceCloning,
  selectedVoiceId,
  onSelectVoice,
  open,
  onClose,
}: VoicePickerProps) {
  const confirm = useConfirm();
  const { t } = useI18n();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [editingVoice, setEditingVoice] = useState<Voice | null>(null);

  if (!open) return null;

  const builtins = voices.filter((v) => v.source === "builtin");
  const uploads = voices.filter((v) => v.source === "upload");

  const heading = isDark ? "text-zinc-400" : "text-gray-600";
  const subtle = isDark ? "text-zinc-400" : "text-gray-600";
  const empty = isDark ? "text-zinc-600" : "text-gray-500";
  const iconBtn = isDark
    ? "text-zinc-400 hover:text-indigo-400"
    : "text-gray-600 hover:text-indigo-600";

  const rowState = (selected: boolean) =>
    selected
      ? "bg-indigo-100 text-indigo-900 ring-1 ring-indigo-300 dark:bg-indigo-500/15 dark:text-indigo-50 dark:ring-indigo-500/50"
      : "bg-white text-gray-700 hover:bg-indigo-50 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-indigo-500/10";

  const row = (v: Voice) => {
    const isSelected = onSelectVoice !== undefined && selectedVoiceId === v.id;
    const Icon = v.source === "builtin" ? Volume2 : Mic2;
    return (
      <li
        key={v.id}
        onClick={onSelectVoice ? () => onSelectVoice(v.id) : undefined}
        className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors ${rowState(isSelected)} ${
          onSelectVoice ? "cursor-pointer" : ""
        }`}
      >
        <Icon
          className={`w-4 h-4 shrink-0 ${
            v.source === "builtin"
              ? isSelected
                ? "text-indigo-600 dark:text-indigo-300"
                : "text-gray-400 dark:text-indigo-400/80"
              : isSelected
                ? "text-indigo-600 dark:text-indigo-300"
                : "text-indigo-500 dark:text-indigo-400"
          }`}
        />
        <span className="flex-1 truncate">{v.name}</span>
        {v.gender && (
          <span
            className={`text-xs ${
              isSelected ? "text-indigo-700/90 dark:text-indigo-200/80" : subtle
            }`}
          >
            {v.gender}
          </span>
        )}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setEditingVoice(v);
          }}
          className={`p-1 ${iconBtn} ${focusRing}`}
          title={t("voice.edit")}
        >
          <Pencil className="w-3.5 h-3.5" />
        </button>
        {v.source === "upload" && (
          <button
            type="button"
            onClick={async (e) => {
              e.stopPropagation();
              const ok = await confirm({
                title: `Delete "${v.name}"?`,
                message: "This permanently removes the uploaded voice.",
                confirmLabel: "Delete",
                danger: true,
              });
              if (ok) void onRemoveVoice(v.id);
            }}
            className={`p-1 ${isDark ? "dark:text-red-400" : ""} ${iconBtn} ${focusRing}`}
            title={t("voice.delete")}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </li>
    );
  };

  return (
    <>
      <div className="absolute inset-0 z-30" onClick={onClose} />
      <div
        className={`fixed top-14 right-4 z-40 w-80 max-h-[70vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden ${
          isDark ? "bg-zinc-900 border-zinc-700" : "bg-white border-gray-200"
        }`}
      >
        <div
          className={`flex items-center justify-between px-4 py-3 border-b ${
            isDark ? "border-zinc-800" : "border-gray-200"
          }`}
        >
          <h2 className={`text-sm font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
            Voices
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

        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              {t("voice.builtin")}
            </h3>
            <ul className="space-y-1">
              {builtins.map(row)}
              {builtins.length === 0 && (
                <li className={`text-xs italic px-2 py-1.5 ${empty}`}>{t("voice.none")}</li>
              )}
            </ul>
          </section>

          {supportsVoiceCloning && (
            <section>
              <div className="flex items-center justify-between mb-2">
                <h3 className={`text-xs font-semibold uppercase tracking-wide ${heading}`}>
                  {t("voice.mine")}
                </h3>
                <button
                  type="button"
                  onClick={() => setUploadOpen(true)}
                  className={`p-1 ${iconBtn} ${focusRing}`}
                  title={t("voice.upload")}
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
              <ul className="space-y-1">
                {uploads.map(row)}
                {uploads.length === 0 && (
                  <li className={`text-xs italic px-2 py-1.5 ${empty}`}>{t("voice.uploadHint")}</li>
                )}
              </ul>
            </section>
          )}
        </div>
      </div>

      <UploadVoiceDialog
        open={uploadOpen}
        theme={isDark ? "dark" : "light"}
        onClose={() => setUploadOpen(false)}
        onUpload={onUploadVoice}
      />

      <VoiceMetaDialog
        voice={editingVoice}
        theme={isDark ? "dark" : "light"}
        onClose={() => setEditingVoice(null)}
        onSave={async (meta) => {
          if (editingVoice) {
            await onUpdateVoiceMeta(editingVoice.id, meta);
            setEditingVoice(null);
          }
        }}
      />
    </>
  );
}
