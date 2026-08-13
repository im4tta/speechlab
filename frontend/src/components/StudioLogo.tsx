import { AudioLines } from "lucide-react";

interface Props {
  size?: number;
}

/** SpeechLab-style brand badge: orange gradient tile with a waveform glyph. */
export function StudioLogo({ size = 36 }: Props) {
  return (
    <div
      className="accent-gradient-135 flex items-center justify-center rounded-lg shrink-0"
      style={{ width: size, height: size }}
      role="img"
      aria-label="SpeechLab logo"
    >
      <AudioLines style={{ width: size * 0.55, height: size * 0.55 }} className="text-black" />
    </div>
  );
}
