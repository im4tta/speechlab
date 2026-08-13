"""Khmer text normalization + chunking.

Khmer script has three properties that make naive (English-oriented) text
splitting actively harmful for TTS:

1.  No spaces between words within a sentence — you can only safely cut
    between orthographic *syllable clusters* (a base consonant/independent
    vowel, an optional COENG (U+17D2) + subscript consonant stack, dependent
    vowel signs, and diacritics). Cutting mid-cluster produces garbage input
    that either errors or mispronounces.
2.  Sentence-final punctuation is different from Latin scripts: khan (។),
    bariyoosan (៕), camnuc pii kuuh (៖) plus the borrowed !/?.
3.  Numerals are usually more natural read as words than digit-by-digit, and
    Khmer text frequently mixes Arabic digits (0-9) with native Khmer digits
    (០-៩) in the same document.

Nothing here is engine-specific: `normalize_khmer_text` improves what any
Khmer TTS engine receives, and `chunk_khmer_text` produces cluster-safe,
sentence-respecting chunks sized for a target model context window. The
nanovllm Khmer engine additionally uses the chunk boundaries to fan work out
across nanovllm's concurrent request batching (see nanovllm_engine.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ─── Khmer Unicode ranges (U+1780–U+17FF) ────────────────────────────────────
_KHMER_CONSONANT = "\u1780-\u17a2"
_KHMER_INDEP_VOWEL = "\u17a3-\u17b3"
_COENG = "\u17d2"
_KHMER_DEP_VOWEL = "\u17b6-\u17c5"
_KHMER_DIACRITIC = "\u17c6-\u17d1\u17dd"
_KHMER_DIGIT = "\u17e0-\u17e9"
_KHMER_SYMBOL = "\u17d4-\u17da\u17db-\u17df"  # punctuation + currency (lekto etc.)

# One Khmer orthographic cluster: base char, then zero or more
# (COENG + subscript-consonant) pairs, then dependent vowels/diacritics.
_KHMER_CLUSTER_RE = re.compile(
    rf"[{_KHMER_CONSONANT}{_KHMER_INDEP_VOWEL}]"
    rf"(?:[{_COENG}][{_KHMER_CONSONANT}])*"
    rf"[{_KHMER_DEP_VOWEL}{_KHMER_DIACRITIC}]*"
)

# Sentence-final punctuation: Khmer khan/bariyoosan/camnuc-pii-kuuh + borrowed
# Latin terminators. Comma/semicolon-ish marks are treated as soft breaks only.
_SENTENCE_END_RE = re.compile(r"([\u17d4\u17d5\u17d8\u17d9!?]+|\.{1,3}(?!\d))\s*")
_SOFT_BREAK_RE = re.compile(r"([\u17d6,;:\u17da]+)\s*")

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_NEWLINES_RE = re.compile(r"\n{2,}")

# ─── Number → Khmer words ────────────────────────────────────────────────────
_DIGIT_WORDS = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]
_TEEN_TENS = "ដប់"  # ten
_SCALE_WORDS = [
    (10 ** 12, "លានលាន"),
    (10 ** 9, "ប៊ីលាន"),
    (10 ** 6, "លាន"),
    (10 ** 3, "ពាន់"),
    (100, "រយ"),
]

_KHMER_DIGIT_MAP = str.maketrans("0123456789", "\u17e0\u17e1\u17e2\u17e3\u17e4\u17e5\u17e6\u17e7\u17e8\u17e9")
_ASCII_DIGIT_MAP = str.maketrans("\u17e0\u17e1\u17e2\u17e3\u17e4\u17e5\u17e6\u17e7\u17e8\u17e9", "0123456789")

# Negative lookaround on both sides avoids mangling alphanumeric tokens like
# "VoxCPM2" or "GPT4" that happen to end/start with a digit.
_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z\u1780-\u17ff])[0-9{_KHMER_DIGIT}]+(?:[.,][0-9{_KHMER_DIGIT}]+)*(?![A-Za-z])"
)

# Common symbols/abbreviations worth expanding for natural speech.
_SYMBOL_EXPANSIONS = [
    (re.compile(r"°C"), "អង្សាសេ"),
    (re.compile(r"°F"), "អង្សាហ្វារិនហៃត៍"),
    (re.compile(r"%|ភាគរយ​"), " ភាគរយ"),
    (re.compile(r"\$"), "ដុល្លារ "),
    (re.compile(r"៛"), " រៀល"),
    (re.compile(r"&"), " និង "),
    (re.compile(r"@"), " អាត "),
]


def _int_to_khmer_words(n: int) -> str:
    """Convert a non-negative integer to Khmer number words."""
    if n < 10:
        return _DIGIT_WORDS[n]
    if n < 20:
        rem = n - 10
        return _TEEN_TENS if rem == 0 else f"{_TEEN_TENS}{_DIGIT_WORDS[rem]}"
    if n < 100:
        tens, rem = divmod(n, 10)
        tens_word = {2: "ម្ភៃ", 3: "សាមសិប", 4: "សែសិប", 5: "ហាសិប",
                     6: "ហុកសិប", 7: "ចិតសិប", 8: "ប៉ែតសិប", 9: "កៅសិប"}.get(tens, f"{_DIGIT_WORDS[tens]}ដប់")
        return tens_word if rem == 0 else f"{tens_word}{_DIGIT_WORDS[rem]}"
    for scale_val, scale_word in _SCALE_WORDS:
        if n >= scale_val:
            head, rem = divmod(n, scale_val)
            head_word = _int_to_khmer_words(head) if head > 1 or scale_val == 100 else ""
            out = f"{head_word}{scale_word}"
            if rem:
                out += f"{_int_to_khmer_words(rem)}"
            return out
    return str(n)  # unreachable given _SCALE_WORDS covers everything above 100


def number_to_khmer_words(raw: str) -> str:
    """Convert an Arabic/Khmer numeral string (optionally with , / . separators)
    to spoken Khmer words. Falls back to reading digit-by-digit for anything
    that isn't a clean integer/decimal (phone numbers, IDs, etc.)."""
    ascii_str = raw.translate(_ASCII_DIGIT_MAP).replace(",", "")
    if re.fullmatch(r"\d+", ascii_str):
        try:
            return _int_to_khmer_words(int(ascii_str))
        except (ValueError, RecursionError):
            pass
    if re.fullmatch(r"\d+\.\d+", ascii_str):
        whole, frac = ascii_str.split(".")
        whole_words = _int_to_khmer_words(int(whole)) if whole else _DIGIT_WORDS[0]
        frac_words = "".join(_DIGIT_WORDS[int(d)] for d in frac)
        return f"{whole_words}ចុច{frac_words}"
    # Not a clean number (e.g. "012-345-6789") — read digit by digit so at
    # least it's pronounceable, rather than passing raw punctuation through.
    digits = re.sub(r"[^\d]", "", ascii_str)
    return "".join(_DIGIT_WORDS[int(d)] for d in digits) or raw


def normalize_khmer_text(text: str, *, expand_numbers: bool = True) -> str:
    """Clean + normalize Khmer (and mixed Khmer/Latin) text before synthesis.

    - Strips zero-width joiners/BOM that some Khmer input methods leave behind.
    - Collapses runs of horizontal whitespace; keeps paragraph breaks.
    - Expands common symbols (°C, %, $, ៛, &) to spoken words.
    - Expands standalone numbers to Khmer number words (toggleable — some
      voices/domains prefer digit-by-digit for IDs; leave those to the caller
      by disabling this and pre-formatting).
    """
    if not text:
        return text
    out = _ZERO_WIDTH_RE.sub("", text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _WHITESPACE_RE.sub(" ", out)
    out = _NEWLINES_RE.sub("\n", out)
    for pattern, repl in _SYMBOL_EXPANSIONS:
        out = pattern.sub(repl, out)
    if expand_numbers:
        out = _NUMBER_RE.sub(lambda m: number_to_khmer_words(m.group(0)), out)
    return out.strip()


# ─── Cluster-safe chunking ────────────────────────────────────────────────────
@dataclass
class TextChunk:
    text: str
    index: int


def _khmer_clusters(run: str) -> list[str]:
    """Decompose a whitespace-free run into Khmer orthographic clusters
    (falling back to one output element per character for non-Khmer runs,
    e.g. Latin words or bare digits)."""
    clusters: list[str] = []
    i = 0
    n = len(run)
    while i < n:
        m = _KHMER_CLUSTER_RE.match(run, i)
        if m and m.end() > i:
            clusters.append(m.group(0))
            i = m.end()
        else:
            clusters.append(run[i])
            i += 1
    return clusters


def _pack(units: list[str], max_chars: int, sep: str) -> list[str]:
    """Greedy bin-pack `units` into chunks <= max_chars, joined by `sep`. A
    single unit longer than max_chars passes through oversized (caller is
    expected to have already decomposed anything that can be decomposed)."""
    chunks: list[str] = []
    buf = ""
    for u in units:
        candidate = f"{buf}{sep}{u}" if buf else u
        if buf and len(candidate) > max_chars:
            chunks.append(buf)
            buf = u
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def _cluster_safe_hard_split(segment: str, max_chars: int) -> list[str]:
    """Split an over-long run with no usable punctuation breaks. Prefers
    whitespace boundaries (keeps Latin words / numbers intact); only decomposes
    down to individual Khmer orthographic clusters for a run that has no
    whitespace and still exceeds max_chars on its own."""
    words = segment.split(" ")
    if len(words) > 1:
        packed = _pack(words, max_chars, sep=" ")
        out: list[str] = []
        for p in packed:
            if len(p) > max_chars:
                out.extend(_pack(_khmer_clusters(p), max_chars, sep=""))
            else:
                out.append(p)
        return out
    return _pack(_khmer_clusters(segment), max_chars, sep="")


def _split_sentences(text: str) -> list[str]:
    """Split on Khmer/Latin sentence terminators, keeping the terminator
    attached to the sentence that precedes it."""
    sentences: list[str] = []
    last = 0
    for m in _SENTENCE_END_RE.finditer(text):
        sentences.append(text[last:m.end()].strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _split_soft(sentence: str, max_chars: int) -> list[str]:
    """A sentence that's still too long: try soft-break punctuation (commas,
    colons) before falling back to a cluster-safe hard split."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    last = 0
    for m in _SOFT_BREAK_RE.finditer(sentence):
        parts.append(sentence[last:m.end()].strip())
        last = m.end()
    tail = sentence[last:].strip()
    if tail:
        parts.append(tail)
    if len(parts) <= 1:
        return _cluster_safe_hard_split(sentence, max_chars)

    out: list[str] = []
    for p in parts:
        if len(p) > max_chars:
            out.extend(_cluster_safe_hard_split(p, max_chars))
        else:
            out.append(p)
    return out


def chunk_khmer_text(text: str, max_chars: int = 220) -> list[TextChunk]:
    """Split text into TTS-safe chunks: sentence-respecting, cluster-safe,
    and greedily bin-packed up to `max_chars` so short sentences merge into
    fewer requests (fewer join seams, better use of nanovllm's batching).

    Paragraph breaks (blank lines) always start a new chunk, since inserting
    a natural pause there sounds better than gluing paragraphs together.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences: list[str] = []
        for s in _split_sentences(paragraph):
            sentences.extend(_split_soft(s, max_chars))

        buf = ""
        for s in sentences:
            if buf and len(buf) + 1 + len(s) > max_chars:
                chunks.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip() if buf else s
        if buf:
            chunks.append(buf)

    return [TextChunk(text=c, index=i) for i, c in enumerate(chunks)]


def is_khmer_text(text: str, threshold: float = 0.15) -> bool:
    """Heuristic: True if a meaningful fraction of the letters are Khmer
    script. Used to decide whether to apply Khmer-specific normalization to
    mixed-language input rather than assuming based on the active engine."""
    letters = [c for c in text if c.isalpha() or ("\u1780" <= c <= "\u17ff")]
    if not letters:
        return False
    khmer = sum(1 for c in letters if "\u1780" <= c <= "\u17ff")
    return (khmer / len(letters)) >= threshold
