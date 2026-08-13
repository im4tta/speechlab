import { useEffect } from "react";
import { LayoutGrid, SlidersHorizontal, X } from "lucide-react";
import { focusRing } from "@/lib/theme";
import { CacheBody } from "./CachePanel";
import { ControlSections, type ControlSectionsProps } from "./ControlSections";

interface Props extends ControlSectionsProps {
  tab: "control" | "recent";
  onTabChange: (t: "control" | "recent") => void;
  onClose: () => void;
}

const TABS = [
  { id: "control", label: "Controls", Icon: SlidersHorizontal },
  { id: "recent", label: "Recent generations", Icon: LayoutGrid },
] as const;

/**
 * Full-screen page for the control sections and the Recent generations list,
 * so every control and every cached take can be inspected at full size.
 */
export function FullScreenPage({ isDark, tab, onTabChange, onClose, ...sections }: Props) {
  const border = isDark ? "border-zinc-800" : "border-gray-200";
  const heading = isDark ? "text-zinc-400" : "text-gray-600";

  // Esc closes the full page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col" role="dialog" aria-modal="true">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* Page */}
      <div
        className={`relative flex-1 flex flex-col overflow-hidden ${
          isDark ? "bg-zinc-950/95 backdrop-blur-2xl" : "bg-gray-100/95 backdrop-blur-2xl"
        }`}
      >
        {/* Header */}
        <div
          className={`px-5 py-4 border-b flex items-center gap-4 shrink-0 ${border}`}
        >
          <h2 className={`text-sm font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
            {tab === "control" ? "Controls" : "Recent generations"}
          </h2>

          {/* Tab switcher */}
          <nav className="flex gap-1 p-1 rounded-lg">
            {TABS.map(({ id, label, Icon }) => {
              const active = tab === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => onTabChange(id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    active
                      ? "accent-gradient text-black"
                      : isDark
                        ? "text-zinc-400 hover:text-zinc-200"
                        : "text-gray-600 hover:text-gray-800"
                  } ${focusRing}`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              );
            })}
          </nav>

          <span className={`text-xs ml-auto ${heading}`}>Esc to close</span>

          <button
            type="button"
            onClick={onClose}
            className={`p-1.5 rounded transition-colors ${
              isDark ? "text-zinc-400 hover:text-white" : "text-gray-600 hover:text-gray-900"
            } ${focusRing}`}
            title="Close full page"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 @container">
          {tab === "control" ? (
            <div className="max-w-6xl mx-auto">
              <ControlSections
                {...sections}
                isDark={isDark}
                wide
                onOpenRecent={() => onTabChange("recent")}
              />
            </div>
          ) : (
            <div className="max-w-4xl mx-auto">
              <section
                className={`p-4 dark:bg-zinc-900 dark:border-zinc-800 bg-white border border-gray-200 rounded-xl`}
              >
                <CacheBody
                  isDark={isDark}
                  data={sections.cacheData}
                  busy={sections.cacheBusy}
                  onClear={sections.onCacheClear}
                  onDelete={sections.onCacheDelete}
                  fullHeight
                />
                <button
                  type="button"
                  onClick={sections.onCacheRefresh}
                  disabled={sections.cacheBusy}
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
          )}
        </div>
      </div>
    </div>
  );
}
