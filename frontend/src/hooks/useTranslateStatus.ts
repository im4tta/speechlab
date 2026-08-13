import { useCallback, useEffect, useState } from "react";
import { getTranslateStatus } from "@/lib/api";
import type { TranslateStatus } from "@/types/models";

/**
 * Translation-model status. Each model's language list comes back even before
 * its weights download (the backend serves a static list), so the target-language
 * picker is populated on first paint.
 */
export function useTranslateStatus() {
  const [status, setStatus] = useState<TranslateStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getTranslateStatus());
    } catch {
      /* leave prior status as-is */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { status, refresh };
}
