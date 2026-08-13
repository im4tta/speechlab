"""Khmer post-processing for ASR output — ported from the user's KhmerScribe
tool (km-yt-scribe-clean: spell_corrector.py + khmer_utils.py).

Applied to Whisper segments that contain Khmer script:
  * `clean_artifacts` — strips leaked caption markers like ">>"/"&gt;&gt;".
  * `correct_segments` — dictionary + phoneme (G2P) spell correction against a
    ~34k-word Khmer wordlist, with frequency tie-breaking for homophone pairs.
  * `segment_transcript_segments` — CRF word segmentation (khmer-nltk, with a
    Khmer/non-Khmer run-split fallback) exposed as the per-segment `segmented`
    field (words joined with zero-width spaces).

Dependencies: `khmer-nltk` (tokenizer) and `khmerpronounce` (G2P). Both degrade
gracefully to simpler fallbacks when not installed. The dictionary data lives in
`backend/services/data/`.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

KHMER_CHAR = re.compile(r"[\u1780-\u17FF]")

_DATA_DIR = Path(__file__).resolve().parent / "data"
WORD_LIST_PATH = _DATA_DIR / "khmer_words.json"
WORD_FREQ_PATH = _DATA_DIR / "khmer_word_freq.json"
PHONEME_MAP_PATH = _DATA_DIR / "phoneme_map.pkl"

PHONEME_CACHE_VERSION = 1
DEFAULT_CER_THRESHOLD = 0.4
TIE_BREAK_MARGIN = 0.05

_word_list: list[str] = []
_word_freq: dict[str, int] = {}
_phoneme_map: dict[str, str] = {}
_trigram_index: dict[str, list[str]] = {}
_correction_cache: dict[str, str] = {}
_loaded = False


# ---------------------------------------------------------------- helpers

def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(prev + cost, dp[j] + 1, dp[j - 1] + 1)
            prev = temp
    return dp[n]


def _normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b), 1)


def _phonemize(word: str) -> str:
    try:
        from khmerpronounce import pronounce

        return "".join(pronounce(word))
    except Exception:  # noqa: BLE001 — phoneme path is optional
        return word


def _ngrams(s: str, n: int = 3):
    return {s[i : i + n] for i in range(len(s) - n + 1)}


# ---------------------------------------------------------------- data load

def _load_cache() -> None:
    global _word_list, _word_freq, _loaded
    if _loaded:
        return
    try:
        _word_list = json.loads(WORD_LIST_PATH.read_text(encoding="utf-8")) if WORD_LIST_PATH.is_file() else []
    except Exception:  # noqa: BLE001
        _word_list = []
    try:
        _word_freq = json.loads(WORD_FREQ_PATH.read_text(encoding="utf-8")) if WORD_FREQ_PATH.is_file() else {}
    except Exception:  # noqa: BLE001
        _word_freq = {}
    _load_phoneme_map()
    _build_trigram_index()
    _loaded = True


def _load_phoneme_map() -> None:
    global _phoneme_map
    _phoneme_map = {}
    if not _word_list:
        return
    if PHONEME_MAP_PATH.is_file():
        try:
            cached = pickle.loads(PHONEME_MAP_PATH.read_bytes())
            if cached.get("version") == PHONEME_CACHE_VERSION and cached.get("words") == _word_list:
                _phoneme_map = cached["map"]
                return
        except Exception:  # noqa: BLE001
            pass
    # One-time G2P build over the dictionary (silences khmerpronounce's stderr
    # spam for rare symbols by redirecting fd 2 for this bulk pass).
    import os as _os

    stderr_fd = 2
    saved_fd = _os.dup(stderr_fd)
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        _os.dup2(devnull_fd, stderr_fd)
        for w in _word_list:
            _phoneme_map[w] = _phonemize(w)
    finally:
        _os.dup2(saved_fd, stderr_fd)
        _os.close(devnull_fd)
        _os.close(saved_fd)


def _build_trigram_index() -> None:
    global _trigram_index
    _trigram_index = {}
    for w in _word_list:
        for ng in _ngrams(w):
            _trigram_index.setdefault(ng, []).append(w)


# ---------------------------------------------------------------- correction

# Khmer character normalisation for robust edit-distance comparison.
_KHMER_NORM = str.maketrans({
    "\u17bb": "\u17bc",
    "\u17bd": "\u17bc",
    "\u17be": "\u17b6",
    "\u17c6": "\u1798",
    "\u17c7": "\u1784",
    "\u17c4": "\u17c1",
    "\u17c5": "\u17c2",
    "\u1780": "\u1781",
    "\u1782": "\u1783",
    "\u1784": "\u1785",
    "\u1786": "\u1787",
    "\u1788": "\u1789",
    "\u178a": "\u178b",
    "\u178c": "\u178d",
    "\u178e": "\u178f",
    "\u1790": "\u1791",
    "\u1792": "\u1793",
    "\u1794": "\u1795",
    "\u1796": "\u1797",
    "\u1798": "\u1799",
    "\u179a": "\u179b",
    "\u179c": "\u179d",
    "\u179e": "\u179f",
    "\u17a0": "\u17a1",
    "\u17a2": "\u17a3",
})
_SUB_RE = re.compile("\u17d2.")
_STRESS_MARKS_RE = re.compile("[\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0]")


def _norm_khmer(word: str) -> str:
    w = _STRESS_MARKS_RE.sub("", word)
    w = _SUB_RE.sub("", w)
    return w.translate(_KHMER_NORM)


def _levenshtein_norm(a: str, b: str) -> float:
    return _normalized_edit_distance(_norm_khmer(a), _norm_khmer(b))


def correct_word(word: str, threshold: float | None = None) -> str:
    if threshold is None:
        threshold = DEFAULT_CER_THRESHOLD
    word = word.strip()
    if not word or not KHMER_CHAR.match(word[0]) or len(word) <= 1:
        return word
    _load_cache()
    if word in _correction_cache:
        return _correction_cache[word]
    if word in _word_list:
        _correction_cache[word] = word
        return word

    word_trigrams = _ngrams(word)
    word_len = len(word)
    candidates: set[str] = set()
    for ng in word_trigrams:
        for w in _trigram_index.get(ng, []):
            if abs(len(w) - word_len) <= 3:
                candidates.add(w)
    if not candidates:
        candidates = set(_word_list)

    input_phon = _phonemize(word) if len(candidates) > 1 else word
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        cand_phon = _phoneme_map.get(candidate, candidate)
        raw_dist = _normalized_edit_distance(word, candidate)
        phon_dist = _normalized_edit_distance(input_phon, cand_phon)
        norm_dist = _levenshtein_norm(word, candidate)
        score = min(raw_dist, phon_dist, norm_dist)
        scored.append((score, candidate))

    scored.sort(key=lambda x: x[0])
    best_score = scored[0][0] if scored else 1.0
    tied = [c for s, c in scored if s <= best_score + TIE_BREAK_MARGIN]
    if len(tied) > 1 and _word_freq:
        best_match = max(tied, key=lambda w: _word_freq.get(w, 0))
    else:
        best_match = scored[0][1] if scored else word

    result = best_match if best_score <= threshold else word
    _correction_cache[word] = result
    return result


# ---------------------------------------------------------------- segmentation

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from khmernltk import word_tokenize

            _tokenizer = word_tokenize
        except Exception:  # noqa: BLE001
            _tokenizer = False
    return _tokenizer


def _fallback_tokenize(text: str):
    return [p for p in re.findall(r"[\u1780-\u17FF]+|[^\u1780-\u17FF]+", text) if p]


def tokenize(text: str):
    if not KHMER_CHAR.search(text):
        return text.split()
    tok = _get_tokenizer()
    if tok:
        try:
            return tok(text)
        except Exception:  # noqa: BLE001
            pass
    return _fallback_tokenize(text)


ARTIFACT_PREFIX_RE = re.compile(r"^(?:\s*(?:&gt;|>)+)+\s*")


def clean_artifacts(text: str) -> str:
    """Strip leaked caption artifacts like '>>' / '&gt;&gt;' speaker markers."""
    return ARTIFACT_PREFIX_RE.sub("", text).strip()


def correct_segments(segments: list[dict], threshold: float | None = None) -> list[dict]:
    """Correct each segment's Khmer text in place (dictionary + phonemes)."""
    for seg in segments:
        text = clean_artifacts(seg.get("text", ""))
        seg["text"] = text
        if not KHMER_CHAR.search(text):
            continue
        words = tokenize(text)
        corrected = []
        for word in words:
            if KHMER_CHAR.match(word[0]):
                corrected.append(correct_word(word, threshold))
            else:
                corrected.append(word)
        seg["text"] = " ".join(corrected)
    return segments


def segment_transcript_segments(segments: list[dict]) -> list[dict]:
    """Add a `segmented` field (words joined with zero-width spaces) per segment."""
    for seg in segments:
        text = seg.get("text", "")
        if KHMER_CHAR.search(text):
            seg["segmented"] = "\u200B".join(tokenize(text))
    return segments
