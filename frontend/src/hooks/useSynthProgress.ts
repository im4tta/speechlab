import { useEffect, useState } from "react";
import { getSynthProgress } from "@/lib/api";
import type { SynthProgress } from "@/types/models";

const IDLE: SynthProgress = {
  state: "idle",
  engine: null,
  text: null,
  elapsed_sec: 0,
  done: 0,
  total: null,
  percent: null,
  eta_sec: null,
  stage: null,
};

/**
 * Polls GET /api/synthesis/progress every second while `enabled`. Also keeps a
 * local elapsed ticker so the timer advances smoothly even between polls.
 */
export function useSynthProgress(enabled: boolean) {
  const [snapshot, setSnapshot] = useState<SynthProgress>(IDLE);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setSnapshot(IDLE);
      setElapsed(0);
      return;
    }
    let cancelled = false;
    const refresh = async () => {
      try {
        const s = await getSynthProgress();
        if (cancelled) return;
        setSnapshot(s);
        setElapsed(s.elapsed_sec);
      } catch {
        // endpoint briefly unreachable; keep last snapshot
      }
    };
    void refresh();
    const t = setInterval(refresh, 1_000);
    const ticker = setInterval(() => setElapsed((e) => e + 1), 1_000);
    return () => {
      cancelled = true;
      clearInterval(t);
      clearInterval(ticker);
    };
  }, [enabled]);

  return { snapshot, elapsed };
}
