/**
 * WCAG 2.x relative-luminance contrast math + the subset of the Tailwind v3
 * palette this app uses. Pure and dependency-free so it can run under Vitest
 * and double as living documentation of our color decisions.
 */

export const PALETTE: Record<string, string> = {
  white: "#ffffff",
  black: "#000000",
  "gray-50": "#fafafb",
  "gray-100": "#f5f5f7",
  "gray-200": "#e6e6ea",
  "gray-300": "#d4d4da",
  "gray-400": "#a1a1ab",
  "gray-500": "#71717c",
  "gray-600": "#575760",
  "gray-700": "#3f3f46",
  "gray-900": "#18181b",
  "zinc-100": "#f4f4f5",
  "zinc-300": "#d4d4d8",
  "zinc-400": "#a1a1aa",
  "zinc-500": "#71717a",
  "zinc-700": "#26262a",
  "zinc-800": "#1a1a1e",
  "zinc-900": "#101013",
  "zinc-950": "#060606",
  "indigo-100": "#ffe3cd",
  "indigo-200": "#ffc59a",
  "indigo-300": "#ffab6b",
  "indigo-400": "#ff8f4d",
  "indigo-500": "#ff7a3d",
  "indigo-600": "#ff5e3a",
  "indigo-700": "#c2410c",
  "indigo-800": "#a0320c",
  "indigo-900": "#7c2709",
  "amber-600": "#d97706",
  "amber-700": "#b45309",
  "red-300": "#fca5a5",
  "red-600": "#dc2626",
  "red-700": "#b91c1c",
};

function channelToLinear(c8: number): number {
  const c = c8 / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Relative luminance per WCAG 2.x. */
export function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return (
    0.2126 * channelToLinear(r) +
    0.7152 * channelToLinear(g) +
    0.0722 * channelToLinear(b)
  );
}

/** Contrast ratio between two hex colors (order-independent, 1..21). */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  const lighter = Math.max(la, lb);
  const darker = Math.min(la, lb);
  return (lighter + 0.05) / (darker + 0.05);
}
