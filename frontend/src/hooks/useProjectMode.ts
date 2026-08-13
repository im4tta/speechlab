import { useCallback, useEffect, useState } from "react";
import type { DubBuffer, ProjectMode, TranscribeBuffer, TtsBuffer } from "@/types/models";

const MODE_KEY = "vs.mode";
const TTS_KEY = "vs.tts";
const TRANSCRIBE_KEY = "vs.transcribe";
const DUB_KEY = "vs.dub";
const EMPTY_TTS: TtsBuffer = { text: "", voiceId: null, language: null };
const EMPTY_TRANSCRIBE: TranscribeBuffer = {
  fileName: "",
  text: "",
  language: null, // auto-detect
  timestamps: true,
  segments: [],
  detectedLanguage: "",
  targetLanguage: null,
};
const EMPTY_DUB: DubBuffer = {
  fileName: "",
  segments: [],
  detectedLanguage: "",
  voiceId: null,
  targetLanguage: null,
};

function readMode(): ProjectMode | null {
  const v = localStorage.getItem(MODE_KEY);
  // A stored "music" (from the removed music mode) falls through to null, so
  // those users land on the ModeChooser and re-pick.
  return v === "tts" || v === "podcast" || v === "transcribe" || v === "dub" ? v : null;
}
function readDub(): DubBuffer {
  try {
    const raw = localStorage.getItem(DUB_KEY);
    if (!raw) return EMPTY_DUB;
    const p = JSON.parse(raw) as Partial<DubBuffer>;
    return { ...EMPTY_DUB, ...p };
  } catch {
    return EMPTY_DUB;
  }
}
function readTranscribe(): TranscribeBuffer {
  try {
    const raw = localStorage.getItem(TRANSCRIBE_KEY);
    if (!raw) return EMPTY_TRANSCRIBE;
    const p = JSON.parse(raw) as Partial<TranscribeBuffer>;
    return { ...EMPTY_TRANSCRIBE, ...p };
  } catch {
    return EMPTY_TRANSCRIBE;
  }
}
function readTts(): TtsBuffer {
  try {
    const raw = localStorage.getItem(TTS_KEY);
    if (!raw) return EMPTY_TTS;
    const p = JSON.parse(raw) as Partial<TtsBuffer>;
    return {
      text: p.text ?? "",
      voiceId: p.voiceId ?? null,
      language: p.language ?? null,
      omnivoiceMode: p.omnivoiceMode,
      voiceDesign: p.voiceDesign,
    };
  } catch {
    return EMPTY_TTS;
  }
}

export interface UseProjectModeApi {
  mode: ProjectMode | null;
  setMode: (m: ProjectMode) => void;
  tts: TtsBuffer;
  setTtsText: (text: string) => void;
  setTtsVoice: (voiceId: string | null) => void;
  setTtsLanguage: (language: string | null) => void;
  setTtsOmniMode: (mode: "clone" | "design" | "auto") => void;
  setTtsVoiceDesign: (voiceDesign: string) => void;
  transcribe: TranscribeBuffer;
  setTranscribe: (partial: Partial<TranscribeBuffer>) => void;
  dub: DubBuffer;
  setDub: (partial: Partial<DubBuffer>) => void;
  setDubVoice: (voiceId: string | null) => void;
}

export function useProjectMode(): UseProjectModeApi {
  const [mode, setModeState] = useState<ProjectMode | null>(readMode);
  const [tts, setTts] = useState<TtsBuffer>(readTts);
  const [transcribe, setTranscribeState] = useState<TranscribeBuffer>(readTranscribe);
  const [dub, setDubState] = useState<DubBuffer>(readDub);

  useEffect(() => {
    if (mode) localStorage.setItem(MODE_KEY, mode);
  }, [mode]);
  useEffect(() => {
    localStorage.setItem(TTS_KEY, JSON.stringify(tts));
  }, [tts]);
  useEffect(() => {
    localStorage.setItem(TRANSCRIBE_KEY, JSON.stringify(transcribe));
  }, [transcribe]);
  useEffect(() => {
    localStorage.setItem(DUB_KEY, JSON.stringify(dub));
  }, [dub]);

  const setMode = useCallback((m: ProjectMode) => setModeState(m), []);
  const setTtsText = useCallback((text: string) => setTts((t) => ({ ...t, text })), []);
  const setTtsVoice = useCallback((voiceId: string | null) => setTts((t) => ({ ...t, voiceId })), []);
  const setTtsLanguage = useCallback((language: string | null) => setTts((t) => ({ ...t, language })), []);
  const setTtsOmniMode = useCallback(
    (omnivoiceMode: "clone" | "design" | "auto") => setTts((t) => ({ ...t, omnivoiceMode })),
    [],
  );
  const setTtsVoiceDesign = useCallback(
    (voiceDesign: string) => setTts((t) => ({ ...t, voiceDesign })),
    [],
  );
  const setTranscribe = useCallback(
    (partial: Partial<TranscribeBuffer>) => setTranscribeState((t) => ({ ...t, ...partial })),
    [],
  );
  const setDub = useCallback(
    (partial: Partial<DubBuffer>) => setDubState((d) => ({ ...d, ...partial })),
    [],
  );
  const setDubVoice = useCallback((voiceId: string | null) => setDubState((d) => ({ ...d, voiceId })), []);

  return {
    mode,
    setMode,
    tts,
    setTtsText,
    setTtsVoice,
    setTtsLanguage,
    setTtsOmniMode,
    setTtsVoiceDesign,
    transcribe,
    setTranscribe,
    dub,
    setDub,
    setDubVoice,
  };
}
