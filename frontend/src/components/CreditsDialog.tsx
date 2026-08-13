import { X } from "lucide-react";
import { focusRing } from "@/lib/theme";

interface Props {
  isDark: boolean;
  onClose: () => void;
}

const ENGINES = [
  ["VibeVoice-1.5B", "Neural Collective / Microsoft"],
  ["Kokoro-82M", "hexgrad"],
  ["Kitten TTS Mini", "KittenML"],
  ["Chatterbox Multilingual V3", "Resemble AI"],
  ["OmniVoice", "K2-FSA"],
  ["VoxCPM2", "OpenBMB"],
  ["nano-vllm-voxcpm (Khmer batching)", "a710128 / nanovllm"],
  ["Qwen3-TTS CustomVoice", "Alibaba Qwen"],
  ["Whisper (ASR)", "OpenAI"],
  ["M2M-100 / Argos Translate", "Meta / Argos"],
] as const;

const KH_ASR = [
  "seanghay (whisper-small/medium-khmer, unspaced)",
  "1morecupofhottea (whisper-turbo-khmer v7/v9)",
  "ksoky (whisper-large-khmer-asr)",
  "steja (whisper-small/large-khmer)",
  "ken0997 (whisper-medium-khmer)",
] as const;

const DEVS = [
  ["im4tta", "Neural Collective — original Voice Studio author"],
  ["seanghay", "voxkhtts (SpeechLab) + voxcpm2-server + Khmer Whisper fine-tunes"],
  ["1morecupofhottea", "Khmer whisper-turbo fine-tunes (v7 / v9)"],
  ["OpenBMB", "VoxCPM2 model + reference implementation"],
  ["a710128", "nano-vllm-voxcpm runtime"],
  ["K2-FSA", "OmniVoice"],
  ["Resemble AI", "Chatterbox"],
  ["hexgrad", "Kokoro"],
  ["KittenML", "Kitten TTS Mini"],
  ["Alibaba Qwen", "Qwen3-TTS CustomVoice"],
  ["OpenAI", "Whisper large-v3 / turbo"],
  ["Meta", "M2M-100 translation"],
] as const;

const TOOLS = [
  "PyTorch",
  "Hugging Face Hub",
  "transformers",
  "FastAPI",
  "React",
  "Vite",
  "Tailwind CSS",
  "lucide",
  "Hanuman · Battambang · Moul · Inter (fonts)",
] as const;

export function CreditsDialog({ isDark, onClose }: Props) {
  const border = isDark ? "border-zinc-800" : "border-gray-200";
  const heading = isDark ? "text-zinc-400" : "text-gray-600";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div
        className={`relative w-full max-w-2xl max-h-[88vh] flex flex-col rounded-2xl shadow-2xl border ${
          isDark ? "bg-zinc-900 border-zinc-800" : "bg-white border-gray-200"
        }`}
      >
        <div className={`px-6 py-4 border-b flex items-center justify-between ${border}`}>
          <h2 className={`text-base font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
            Credits &amp; Acknowledgements
          </h2>
          <button
            type="button"
            onClick={onClose}
            className={`p-1 rounded ${isDark ? "text-zinc-400 hover:text-white" : "text-gray-600 hover:text-gray-900"} ${focusRing}`}
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6 text-sm">
          {/* AI improvements */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Built &amp; improved with AI
            </h3>
            <p className={isDark ? "text-zinc-300" : "text-gray-700"}>
              This app — its UI, SpeechLab restyle, live generation progress, per-take detail
              metadata, hardware recommendations, local VoxCPM2 loading with CPU-offload, and
              the full-page controls — was designed, coded and iterated on with assistance
              from an AI coding agent. Human review and local testing remain the final gate.
            </p>
          </section>

          {/* Developers */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Developers &amp; contributors
            </h3>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1.5">
              {DEVS.map(([name, role]) => (
                <li key={name} className="flex flex-col min-w-0">
                  <span className={`font-semibold ${isDark ? "text-zinc-100" : "text-gray-900"}`}>
                    {name}
                  </span>
                  <span className={`text-xs ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
                    {role}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          {/* Speech engines */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Speech &amp; language models
            </h3>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1.5">
              {ENGINES.map(([name, credit]) => (
                <li key={name} className="flex justify-between gap-3">
                  <span className={isDark ? "text-zinc-200" : "text-gray-800"}>{name}</span>
                  <span className={`text-xs ${isDark ? "text-zinc-500" : "text-gray-500"}`}>
                    {credit}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          {/* Khmer ASR contributors */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Khmer speech-to-text contributors
            </h3>
            <ul className="space-y-1">
              {KH_ASR.map((c) => (
                <li key={c} className={isDark ? "text-zinc-300" : "text-gray-700"}>
                  · {c}
                </li>
              ))}
            </ul>
          </section>

          {/* Frameworks */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Built on
            </h3>
            <p className={isDark ? "text-zinc-300" : "text-gray-700"}>{TOOLS.join(" · ")}</p>
          </section>

          {/* Origin */}
          <section>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-2 ${heading}`}>
              Origin
            </h3>
            <p className={isDark ? "text-zinc-300" : "text-gray-700"}>
              Inspired by and adapted from the{" "}
              <span className="font-semibold">voxkhtts</span> Khmer TTS Studio (SpeechLab) and
              the multi-engine{" "}
              <span className="font-semibold">Voice Studio by Neural Collective (im4tta)</span>.
              All model weights are open-source and run locally on this machine.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
