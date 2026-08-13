import { StudioLogo } from "./StudioLogo";

interface Props {
  isDark: boolean;
}

/**
 * Thin attribution strip pinned at the bottom of the middle column.
 */
export function BrandFooter({ isDark }: Props) {
  return (
    <footer
      className={`shrink-0 border-t px-6 py-2 flex items-center justify-center gap-2 ${
        isDark ? "border-zinc-800 text-zinc-500" : "border-gray-200 text-gray-500"
      }`}
    >
      <StudioLogo size={20} />
      <span className="text-xs">SpeechLab</span>
    </footer>
  );
}

