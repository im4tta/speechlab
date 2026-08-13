// Lightweight i18n — English + Khmer (ភាសាខ្មែរ).
// `t(key)` falls back to the English text when a key is missing, so
// untranslated strings degrade gracefully instead of breaking.

import { createContext, useCallback, useContext, useMemo, useState } from "react";

export type Lang = "en" | "km";

const LS_KEY = "vs.lang";

const EN: Record<string, string> = {
  // header
  "brand.tagline": "AI Voice Studio",
  "header.monitor": "Monitor",
  "header.controls": "Controls",
  "header.recent": "Recent",
  "header.voices": "Voices",
  "header.settings": "Settings",
  "header.credits": "Credits & acknowledgements",
  "header.theme": "Toggle appearance",
  // modes
  "mode.tts": "Text-to-Voice",
  "mode.podcast": "Podcast",
  "mode.transcribe": "Transcribe",
  "mode.dub": "Dub",
  "chooser.ttsDesc": "Type or paste text and generate with a single voice.",
  "chooser.podcastDesc": "Build a multi-speaker conversation from segments.",
  "chooser.transcribeDesc": "Turn an audio file into text, with subtitles.",
  "chooser.dubDesc": "Re-voice an audio clip in a voice you choose.",
  // toolbar
  "toolbar.addSegment": "Add Segment",
  "toolbar.generateAll": "Generate All",
  // common buttons
  "btn.generate": "Generate",
  "btn.play": "Play",
  "btn.stop": "Stop",
  "btn.download": "Download",
  "btn.upload": "Upload",
  "btn.copy": "Copy",
  "btn.close": "Close",
  "btn.clear": "Clear",
  "btn.cancel": "Cancel",
  "btn.save": "Save",
  "btn.fetch": "Fetch",
  // controls
  "controls.title": "Controls",
  "controls.engine": "Engine",
  "controls.translation": "Translation",
  "controls.cfg": "Voice fidelity (CFG)",
  "controls.cfg.chatterbox": "CFG weight (voice fidelity)",
  "controls.exaggeration": "Voice expressiveness (Chatterbox)",
  "controls.quality": "Quality",
  "controls.quality.hint": "Diffusion steps: Fast 5 · Balanced 10 · High 25. Higher = better quality, slower.",
  "controls.advanced": "Advanced generation",
  "controls.seed": "Seed (optional)",
  "controls.reset": "Reset to defaults",
  "controls.localModel": "Local VoxCPM2 model folder",
  "vox.useThis": "Use this folder",
  "controls.recent": "Recent generations",
  "controls.refresh": "Refresh list",
  // recommendation card
  "rec.title.gpu": "Pick a model for your {n} GB GPU",
  "rec.title.noGpu": "Pick a model for your hardware",
  "rec.loading": "Reading your hardware…",
  // voice picker
  "voice.title": "Voices",
  "voice.builtin": "Built-in voices",
  "voice.mine": "My voices",
  "voice.none": "No built-in voices.",
  "voice.uploadHint": "Click + to upload a voice.",
  "voice.upload": "Upload voice",
  "voice.edit": "Edit name / gender / language",
  "voice.delete": "Delete",
  // tts editor
  "tts.placeholder": "Type or paste text to synthesize…",
  "tts.stats": "{c} chars · {w} words · {s}",
  "tts.voice": "Voice",
  "tts.none": "none selected",
  "tts.history": "Text history",
  "tts.copy": "Copy text",
  "tts.style": "Style (optional)",
  // transcribe
  "tr.title": "Transcription",
  "tr.model": "Speech-to-text model",
  "tr.language": "Language",
  "tr.autoDetect": "Auto-detect",
  "tr.timestamps": "Timestamps",
  "tr.modelStatus": "Model status",
  "tr.drop": "Drop an audio file here, or choose one",
  "tr.transcribe": "Transcribe",
  "tr.transcribing": "Transcribing…",
  "tr.yt": "YouTube transcript",
  "tr.ytPlaceholder": "Paste a YouTube URL or video id (Khmer captions → ASR fallback)",
  "tr.ytFetch": "Fetch",
  "tr.ytAudio": "Audio",
  "tr.ytWatch": "Watch",
  "tr.ytReasr": "Re-transcribe (ASR)",
  "tr.ytCorrect": "Correct",
  "tr.ytCorrectSave": "Save corrected transcript",
  "tr.ytHistory": "YouTube history",
  "tr.ytOriginal": "Show original",
  "tr.ytRestore": "Restore original text",
  "tr.sendToTts": "Send to Text-to-Voice",
  "tr.export": "Export",
  // dub
  "dub.title": "Dub",
  "dub.drop": "Drop an audio clip here",
  "dub.dub": "Generate Dub",
  // cache
  "cache.recent": "Recent generations",
  "cache.clear": "Clear all",
  "cache.delete": "Delete this entry",
  "cache.download": "Download WAV",
  "cache.empty": "No generations yet.",
  // footer
  "footer.brand": "SpeechLab",
};

const KM: Record<string, string> = {
  "brand.tagline": "ស្ទូឌីយោសំឡេង AI",
  "header.monitor": "តាមដាន",
  "header.controls": "ការកំណត់",
  "header.recent": "ថ្មីៗ",
  "header.voices": "សំឡេង",
  "header.settings": "ការកំណត់",
  "header.credits": "កិត្តិយស និងការអរគុណ",
  "header.theme": "ប្តូររូបរាង",
  "mode.tts": "អក្សរទៅជាសំឡេង",
  "mode.podcast": "ផតខាស",
  "mode.transcribe": "បំប្លែងសំឡេងទៅជាអក្សរ",
  "mode.dub": "បិទសំឡេង",
  "chooser.ttsDesc": "វាយឬបិទភ្ជាប់អក្សរ ហើយបង្កើតសំឡេងដោយសំឡេងតែមួយ។",
  "chooser.podcastDesc": "បង្កើតការសន្ទនាច្រើនអ្នកនិយាយពីផ្នែកនីមួយៗ។",
  "chooser.transcribeDesc": "បំប្លែងឯកសារសំឡេងទៅជាអក្សរ ជាមួយអក្សររត់។",
  "chooser.dubDesc": "បិទសំឡេងថ្មីលើវីដេអូ/សំឡេង តាមសំឡេងដែលអ្នកជ្រើស។",
  "toolbar.addSegment": "បន្ថែមផ្នែក",
  "toolbar.generateAll": "បង្កើតទាំងអស់",
  "btn.generate": "បង្កើត",
  "btn.play": "ចាក់",
  "btn.stop": "បញ្ឈប់",
  "btn.download": "ទាញយក",
  "btn.upload": "ផ្ទុកឡើង",
  "btn.copy": "ចម្លង",
  "btn.close": "បិទ",
  "btn.clear": "សម្អាត",
  "btn.cancel": "បោះបង់",
  "btn.save": "រក្សាទុក",
  "btn.fetch": "ទាញយក",
  "controls.title": "ការកំណត់",
  "controls.engine": "ម៉ាស៊ីន",
  "controls.translation": "ការបកប្រែ",
  "controls.cfg": "សំឡេងស្មោះត្រង់ (CFG)",
  "controls.cfg.chatterbox": "CFG weight (voice fidelity)",
  "controls.exaggeration": "Voice expressiveness (Chatterbox)",
  "controls.quality": "គុណភាព",
  "controls.quality.hint": "ជំហាន: លឿន 5 · មធ្យម 10 · ខ្ពស់ 25. ខ្ពស់ = គុណភាពល្អ យឺតជាង។",
  "controls.advanced": "ការបង្កើតកម្រិតខ្ពស់",
  "controls.seed": "គ្រាប់ (ស្រេចចិត្ត)",
  "controls.reset": "កំណត់ឡើងវិញ",
  "controls.localModel": "ថតគំរូ VoxCPM2 ក្នុងម៉ាស៊ីន",
  "vox.useThis": "ប្រើថតនេះ",
  "controls.recent": "ការបង្កើតថ្មីៗ",
  "controls.refresh": "ធ្វើបញ្ជីឡើងវិញ",
  "rec.title.gpu": "ជ្រើសគំរូសម្រាប់ GPU {n} GB របស់អ្នក",
  "rec.title.noGpu": "ជ្រើសគំរូសម្រាប់ម៉ាស៊ីនរបស់អ្នក",
  "rec.loading": "កំពុងអានផ្នែករឹង…",
  "voice.title": "សំឡេង",
  "voice.builtin": "សំឡេងស្រាប់",
  "voice.mine": "សំឡេងរបស់ខ្ញុំ",
  "voice.none": "គ្មានសំឡេងស្រាប់។",
  "voice.uploadHint": "ចុច + ដើម្បីផ្ទុកសំឡេង។",
  "voice.upload": "ផ្ទុកសំឡេង",
  "voice.edit": "កែឈ្មោះ / ភេទ / ភាសា",
  "voice.delete": "លុប",
  "tts.placeholder": "វាយឬបិទភ្ជាប់អក្សរដើម្បីបង្កើតសំឡេង…",
  "tts.stats": "{c} តួអក្សរ · {w} ពាក្យ · {s}",
  "tts.voice": "សំឡេង",
  "tts.none": "មិនបានជ្រើស",
  "tts.history": "ប្រវត្តិអក្សរ",
  "tts.copy": "ចម្លងអក្សរ",
  "tts.style": "រចនាបទ (ស្រេចចិត្ត)",
  "tr.title": "ការបំប្លែងអក្សរ",
  "tr.model": "គំរូសំឡេងទៅអក្សរ",
  "tr.language": "ភាសា",
  "tr.autoDetect": "ស្វែងរកដោយស្វ័យប្រវត្តិ",
  "tr.timestamps": "ពេលវេលា",
  "tr.modelStatus": "ស្ថានភាពគំរូ",
  "tr.drop": "ទម្លាក់ឯកសារសំឡេងនៅទីនេះ ឬជ្រើសរើស",
  "tr.transcribe": "បំប្លែង",
  "tr.transcribing": "កំពុងបំប្លែង…",
  "tr.yt": "អក្សរពី YouTube",
  "tr.ytPlaceholder": "បិទភ្ជាប់ URL YouTube ឬលេខវីដេអូ",
  "tr.ytFetch": "ទាញយក",
  "tr.ytAudio": "សំឡេង",
  "tr.ytWatch": "មើល",
  "tr.ytReasr": "បំប្លែងឡើងវិញ (ASR)",
  "tr.ytCorrect": "កែតម្រូវ",
  "tr.ytCorrectSave": "រក្សាទុកអក្សរដែលបានកែ",
  "tr.ytHistory": "ប្រវត្តិ YouTube",
  "tr.ytOriginal": "បង្ហាញអក្សរដើម",
  "tr.ytRestore": "ស្តារអក្សរដើម",
  "tr.sendToTts": "ផ្ញើទៅអក្សរទៅសំឡេង",
  "tr.export": "នាំចេញ",
  "dub.title": "បិទសំឡេង",
  "dub.drop": "ទម្លាក់វីដេអូ/សំឡេងនៅទីនេះ",
  "dub.dub": "បង្កើតសំឡេងថ្មី",
  "cache.recent": "ការបង្កើតថ្មីៗ",
  "cache.clear": "សម្អាតទាំងអស់",
  "cache.delete": "លុបធាតុនេះ",
  "cache.download": "ទាញយក WAV",
  "cache.empty": "មិនទាន់មានការបង្កើតទេ។",
  "footer.brand": "SpeechLab",
};

const DICTS: Record<Lang, Record<string, string>> = { en: EN, km: KM };

function loadLang(): Lang {
  try {
    const stored = localStorage.getItem(LS_KEY);
    return stored === "km" ? "km" : "en";
  } catch {
    return "en";
  }
}

export function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, k) =>
    vars[k] != null ? String(vars[k]) : `{${k}}`,
  );
}

interface I18nValue {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

export const I18nContext = createContext<I18nValue>({
  lang: "en",
  setLang: () => {},
  t: (k) => EN[k] ?? k,
});

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(loadLang);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(LS_KEY, l);
    } catch {
      // ignore
    }
  }, []);
  const value = useMemo<I18nValue>(() => {
    const dict = DICTS[lang];
    return {
      lang,
      setLang,
      t: (key, vars) => interpolate(dict[key] ?? EN[key] ?? key, vars),
    };
  }, [lang, setLang]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  return useContext(I18nContext);
}
