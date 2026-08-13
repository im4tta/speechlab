/** @type {import('tailwindcss').Config} */
import containerQueries from "@tailwindcss/container-queries";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── SpeechLab palette ─────────────────────────────────────────────
        // Dark theme (zinc): near-black page bg + subtle white-alpha surfaces.
        // Text tones stay at the previous AAA-tuned values for readability.
        zinc: {
          100: "#f4f4f5", // text-primary
          200: "#e8e8ea",
          300: "#d4d4d8", // text-secondary
          400: "#a1a1aa", // text-muted
          500: "#71717a",
          600: "#35353b",
          700: "#26262a",
          800: "#1a1a1e", // surface / border
          900: "#101013", // surface-solid (#111)
          950: "#060606", // page bg
        },
        // Accent: SpeechLab's orange gradient (start → end). Black text sits
        // on the gradient at ~7:1 (AAA for large, AA for body) — matching the
        // original SpeechLab buttons, which use black-on-orange.
        indigo: {
          50: "#fff2e8",
          100: "#ffe3cd",
          200: "#ffc59a",
          300: "#ffab6b",
          400: "#ff8f4d",
          500: "#ff7a3d",
          600: "#ff5e3a", // accent-start (buttons: use black text)
          700: "#c2410c", // accent text on light (AA on white)
          800: "#a0320c",
          900: "#7c2709",
        },
        // Light theme (gray): SpeechLab's warm-grey light surfaces.
        gray: {
          50: "#fafafb",
          100: "#f5f5f7", // light page bg
          200: "#e6e6ea",
          300: "#d4d4da",
          400: "#a1a1ab",
          500: "#71717c",
          600: "#575760",
          700: "#3f3f46",
          800: "#26262b",
          900: "#18181b",
          950: "#0d0d10",
        },
      },
    },
  },
  plugins: [containerQueries],
};
