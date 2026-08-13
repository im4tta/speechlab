// TTS text history — persists previously synthesized texts locally so the
// user can copy / re-run any of them with a different voice.

const KEY = "vs.tts.history";
const MAX_ITEMS = 50;

export interface TtsHistoryItem {
  text: string;
  at: number;
}

export function loadTtsHistory(): TtsHistoryItem[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((i) => i && typeof i.text === "string")
      : [];
  } catch {
    return [];
  }
}

/** Add a text to the front of history (deduped, capped). Returns new list. */
export function addTtsText(text: string): TtsHistoryItem[] {
  const trimmed = text.trim();
  if (!trimmed) return loadTtsHistory();
  const items = loadTtsHistory().filter((i) => i.text !== trimmed);
  items.unshift({ text: trimmed, at: Date.now() });
  const capped = items.slice(0, MAX_ITEMS);
  try {
    localStorage.setItem(KEY, JSON.stringify(capped));
  } catch {
    // storage unavailable; history just won't persist
  }
  return capped;
}

export function clearTtsHistory(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
