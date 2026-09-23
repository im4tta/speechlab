"""Khmer text normalization + chunking.

Khmer script has properties that make naive (English-oriented) text handling
actively harmful for TTS:

1.  No spaces between words within a sentence — you can only safely cut
    between orthographic *syllable clusters* (a base consonant/independent
    vowel, an optional COENG (U+17D2) + subscript consonant stack, dependent
    vowel signs, and diacritics). Cutting mid-cluster produces garbage input
    that either errors or mispronounces.
2.  Sentence-final punctuation is different from Latin scripts: khan (។),
    bariyoosan (៕), koomuut (៚) plus the borrowed !/?.
3.  Numerals are read as words, and Khmer text mixes Arabic digits (0-9) with
    native Khmer digits (០-៩) — usually glued straight onto the surrounding
    Khmer word ("ឆ្នាំ២០២៤"), since there are no word spaces to separate them.
4.  Written shorthand the model can't pronounce as-is: the repetition mark
    ៗ (lek too — "repeat the previous word"), ។ល។ / ៘ ("et cetera"), dates,
    clock times, currencies (៛, $), units and phone numbers.
5.  Input methods leave invisible debris (zero-width spaces, deprecated
    code points, split vowel sequences like េ+ា instead of ោ) that make the
    same word tokenize differently.

Nothing here is engine-specific: `normalize_khmer_text` improves what any
Khmer TTS engine receives, and `chunk_khmer_text` produces cluster-safe,
sentence-respecting chunks sized for a target model context window, tagged
with the kind of break that ends each one (so the joiner can pick a natural
pause). The nanovllm Khmer engine additionally uses the chunk boundaries to
fan work out across nanovllm's concurrent request batching.

This module is imported by the isolated worker venvs too, so it must stay
stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ─── Khmer Unicode ranges (U+1780–U+17FF) ────────────────────────────────────
_KHMER_CONSONANT = "\u1780-\u17a2"
_KHMER_INDEP_VOWEL = "\u17a3-\u17b3"
_COENG = "\u17d2"
_KHMER_DEP_VOWEL = "\u17b6-\u17c5"
_KHMER_DIACRITIC = "\u17c6-\u17d1\u17d3\u17dd"

# One Khmer orthographic cluster: base char, then zero or more
# (COENG + subscript-consonant) pairs, then dependent vowels/diacritics.
_KHMER_CLUSTER_RE = re.compile(
    rf"[{_KHMER_CONSONANT}{_KHMER_INDEP_VOWEL}]"
    rf"(?:[{_COENG}][{_KHMER_CONSONANT}])*"
    rf"[{_KHMER_DEP_VOWEL}{_KHMER_DIACRITIC}]*"
)

# Sentence-final punctuation: Khmer khan/bariyoosan/koomuut + borrowed Latin
# terminators. Comma/colon-ish marks are treated as soft breaks only.
_SENTENCE_END_RE = re.compile(r"([\u17d4\u17d5\u17da!?]+|\.{1,3}(?!\d))\s*")
_SOFT_BREAK_RE = re.compile(r"([\u17d6,;:]+)\s*")

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"[ \t\f\v\u00a0]+")
_NEWLINES_RE = re.compile(r"\n{2,}")

# ─── Orthographic clean-up ───────────────────────────────────────────────────
# Split vowel sequences some keyboards produce instead of the single composed
# vowel sign (NFC does not compose these for Khmer), plus deprecated code
# points the Unicode standard says not to use.
_ORTHOGRAPHY_FIXES = [
    ("\u17c1\u17b6", "\u17c4"),  # េ + ា  → ោ
    ("\u17c1\u17b8", "\u17be"),  # េ + ី  → ើ
    ("\u17c1\u17b7", "\u17be"),  # េ + ិ  → ើ (common mistype)
    ("\u17a3", "\u17a2"),        # deprecated ឣ → អ
    ("\u17a4", "\u17a2\u17b6"),  # deprecated ឤ → អា
    ("\u17b4", ""),              # invisible inherent vowels (deprecated)
    ("\u17b5", ""),
    ("\u17d3", "\u17c6"),        # deprecated bathamasat → nikahit
    ("\u17d9", ""),              # phnaek muan: decorative start-of-text mark
]
# The same dependent sign typed twice in a row ("ាា") is always a typo.
_DUP_SIGN_RE = re.compile(rf"([{_KHMER_DEP_VOWEL}{_KHMER_DIACRITIC}])\1+")

# ─── Number → Khmer words ────────────────────────────────────────────────────
_DIGIT_WORDS = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]
_TEN = "ដប់"
_TENS_WORDS = {2: "ម្ភៃ", 3: "សាមសិប", 4: "សែសិប", 5: "ហាសិប",
               6: "ហុកសិប", 7: "ចិតសិប", 8: "ប៉ែតសិប", 9: "កៅសិប"}
# Khmer counts in ម៉ឺន (10⁴) and សែន (10⁵) below a million, like spoken Khmer:
# 25 000 = ពីរម៉ឺនប្រាំពាន់, 150 000 = មួយសែនប្រាំម៉ឺន.
_SCALE_WORDS = [
    (10 ** 12, "ទ្រីលាន"),
    (10 ** 9, "ប៊ីលាន"),
    (10 ** 6, "លាន"),
    (10 ** 5, "សែន"),
    (10 ** 4, "ម៉ឺន"),
    (10 ** 3, "ពាន់"),
    (100, "រយ"),
]
_MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា",
           "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]

_ASCII_DIGIT_MAP = str.maketrans("\u17e0\u17e1\u17e2\u17e3\u17e4\u17e5\u17e6\u17e7\u17e8\u17e9", "0123456789")

# A number is only blocked by a *Latin* letter on either side, so alphanumeric
# tokens like "VoxCPM2" / "GPT4" survive, while digits glued to Khmer words
# ("ឆ្នាំ២០២៤", "ទី១") — the normal way Khmer is written — are expanded.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[0-9]+(?:[.,][0-9]+)*(?![A-Za-z0-9])")
_NEGATIVE_RE = re.compile(r"(?<![^\s(])[-−](?=[0-9])")

_DATE_DMY_RE = re.compile(r"(?<![0-9])(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})(?![0-9])")
_DATE_ISO_RE = re.compile(r"(?<![0-9])(\d{4})-(\d{1,2})-(\d{1,2})(?![0-9])")
_TIME_RE = re.compile(
    r"(?<![0-9:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![0-9:])"
    r"(?:\s?([AaPp])\.?[Mm]\.?(?![A-Za-z]))?"
)
# Phone numbers start with 0 or +country code and come in digit groups
# separated by spaces/hyphens ("012 345 678", "+855 12-345-678"). Read them
# digit by digit, one group at a time, instead of as one huge quantity.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9+])(?:\+\d{1,3}[\s\-]?\d{1,3}|0\d{1,3})(?:[\s\-]?\d{2,4}){1,4}(?![A-Za-z0-9])"
)
# "12 000 000" — thousands grouped with spaces (not a leading-zero phone number).
_SPACE_THOUSANDS_RE = re.compile(r"(?<![0-9,.])([1-9]\d{0,2})((?:[ \u202f]\d{3})+)(?![0-9,.])")
# Numeric ranges "5-10" / "2020–2024". Latin-only guards: Khmer letters glue
# straight onto numbers ("ពី៥-១០នាក់"), so they must not block the match.
_RANGE_RE = re.compile(r"(?<![A-Za-z0-9.,\-])(\d+)\s?[\-–]\s?(\d+)(?![A-Za-z0-9.,\-])")
_ARITH_RE = [
    (re.compile(r"(?<=[0-9])\s?\+\s?(?=[0-9])"), " បូក "),
    (re.compile(r"(?<=[0-9])\s?[×x]\s?(?=[0-9])"), " គុណ "),
    (re.compile(r"(?<=[0-9])\s?=\s?(?=[0-9])"), " ស្មើ "),
]

_NUM = r"(\d+(?:[.,]\d+)*)"
_CURRENCY_PREFIX = {"$": "ដុល្លារ", "€": "អឺរ៉ូ", "£": "ផោន", "฿": "បាត", "៛": "រៀល"}
_CURRENCY_BEFORE_RE = re.compile(r"([$€£฿\u17db])\s?" + _NUM)
_CURRENCY_AFTER_RE = re.compile(_NUM + r"\s?([$€£฿\u17db]|USD|usd|KHR|khr)(?![A-Za-z])")
_CURRENCY_CODES = {"USD": "ដុល្លារ", "KHR": "រៀល"}
_PERCENT_RE = re.compile(_NUM + r"\s?%")

# Units after a number. Longest alternatives first so "km/h" wins over "km"
# and "mm" over "m".
_UNITS = [
    ("km/h", "គីឡូម៉ែត្រក្នុងមួយម៉ោង"),
    ("km²", "គីឡូម៉ែត្រការ៉េ"),
    ("m²", "ម៉ែត្រការ៉េ"),
    ("m³", "ម៉ែត្រគូប"),
    ("°C", "អង្សាសេ"),
    ("°F", "អង្សាហ្វារិនហៃ"),
    ("km", "គីឡូម៉ែត្រ"),
    ("kg", "គីឡូក្រាម"),
    ("cm", "សង់ទីម៉ែត្រ"),
    ("mm", "មីលីម៉ែត្រ"),
    ("mg", "មីលីក្រាម"),
    ("ml", "មីលីលីត្រ"),
    ("mL", "មីលីលីត្រ"),
    ("ha", "ហិកតា"),
    ("m", "ម៉ែត្រ"),
    ("g", "ក្រាម"),
    ("l", "លីត្រ"),
    ("L", "លីត្រ"),
]
_UNIT_WORDS = dict(_UNITS)
_UNIT_RE = re.compile(
    _NUM + r"\s?(" + "|".join(re.escape(u) for u, _ in _UNITS) + r")(?![A-Za-z0-9²³])"
)

# Standalone symbols worth speaking (after the number-aware rules above ran).
_SYMBOL_EXPANSIONS = [
    (re.compile(r"°C"), " អង្សាសេ"),
    (re.compile(r"°F"), " អង្សាហ្វារិនហៃ"),
    (re.compile(r"%"), " ភាគរយ"),
    (re.compile(r"\$"), " ដុល្លារ "),
    (re.compile(r"៛"), " រៀល"),
    (re.compile(r"&"), " និង "),
    (re.compile(r"@"), " អាត "),
]

# "et cetera": the ligature ៘ and its spelled-out form ។ល។ are read ជាដើម.
_ETC_RE = re.compile(r"\s?(?:\u17d8|\u17d4\s?\u179b\s?\u17d4)")
# Repetition mark ៗ (optionally typed with a space before it).
_LEK_TOO_RE = re.compile(r"[ \u200b]?\u17d7")

# Markup/decoration the model would otherwise try to "read" or stumble on.
_MARKUP_RE = re.compile(r"[*_#`~|<>\[\]{}•●▪►✓✔❖]+")
_QUOTES_RE = re.compile(r"[\"“”„«»‘’‚‹›「」『』]")


def _int_to_khmer_words(n: int) -> str:
    """Convert a non-negative integer to Khmer number words."""
    if n < 10:
        return _DIGIT_WORDS[n]
    if n < 20:
        rem = n - 10
        return _TEN if rem == 0 else f"{_TEN}{_DIGIT_WORDS[rem]}"
    if n < 100:
        tens, rem = divmod(n, 10)
        tens_word = _TENS_WORDS[tens]
        return tens_word if rem == 0 else f"{tens_word}{_DIGIT_WORDS[rem]}"
    for scale_val, scale_word in _SCALE_WORDS:
        if n >= scale_val:
            head, rem = divmod(n, scale_val)
            out = f"{_int_to_khmer_words(head)}{scale_word}"
            if rem:
                out += _int_to_khmer_words(rem)
            return out
    return str(n)  # unreachable given _SCALE_WORDS covers everything above 100


def _digits_to_words(digits: str) -> str:
    return "".join(_DIGIT_WORDS[int(d)] for d in digits if d.isdigit())


def number_to_khmer_words(raw: str) -> str:
    """Convert an Arabic/Khmer numeral string (optionally with , / . separators)
    to spoken Khmer words.

    - "1,000,000" / "1.000.000" (grouped thousands) → one quantity.
    - "3.14" → decimal, read ចុច + digits.
    - A leading zero ("007", "05") is an identifier, read digit by digit.
    - Anything else that isn't a clean number (e.g. "012-345") is read digit by
      digit so it's at least pronounceable rather than raw punctuation.
    """
    ascii_str = raw.translate(_ASCII_DIGIT_MAP).strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", ascii_str):
        ascii_str = ascii_str.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", ascii_str):
        ascii_str = ascii_str.replace(".", "")
    if re.fullmatch(r"\d+", ascii_str):
        if len(ascii_str) > 1 and ascii_str.startswith("0"):
            return _digits_to_words(ascii_str)
        if len(ascii_str) > 15:  # beyond ទ្រីលាន — an ID, not a quantity
            return _digits_to_words(ascii_str)
        return _int_to_khmer_words(int(ascii_str))
    if re.fullmatch(r"\d+[.,]\d+", ascii_str):
        whole, frac = re.split(r"[.,]", ascii_str)
        whole_words = number_to_khmer_words(whole) if whole else _DIGIT_WORDS[0]
        return f"{whole_words}ចុច{_digits_to_words(frac)}"
    return _digits_to_words(ascii_str) or raw


# ─── Pattern expanders (dates, times, money, units, phones) ──────────────────
def _expand_date(m: re.Match, day: str, month: str, year: str) -> str:
    d, mo = int(day), int(month)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    year_words = number_to_khmer_words(year.lstrip("0") or "0")
    # "ថ្ងៃទី 12/05/2024" already says ថ្ងៃទី — don't say it twice.
    before = m.string[: m.start()].rstrip()
    day_prefix = "" if before.endswith("ថ្ងៃទី") else "ថ្ងៃទី"
    return f" {day_prefix}{_int_to_khmer_words(d)} ខែ{_MONTHS[mo - 1]} ឆ្នាំ{year_words} "


def _date_dmy(m: re.Match) -> str:
    return _expand_date(m, m.group(1), m.group(2), m.group(3))


def _date_iso(m: re.Match) -> str:
    return _expand_date(m, m.group(3), m.group(2), m.group(1))


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return m.group(0)
    # Don't say ម៉ោង twice when the text already reads "ម៉ោង 8:30".
    before = m.string[: m.start()].rstrip()
    out = "" if before.endswith("ម៉ោង") else "ម៉ោង"
    out += _int_to_khmer_words(h)
    if mi:
        out += f" {_int_to_khmer_words(mi)}នាទី"
    if m.group(3) and int(m.group(3)):
        out += f" {_int_to_khmer_words(int(m.group(3)))}វិនាទី"
    if m.group(4):
        out += " ព្រឹក" if m.group(4).lower() == "a" else " ល្ងាច"
    return f" {out} "


def _phone(m: re.Match) -> str:
    raw = m.group(0)
    digits_only = re.sub(r"\D", "", raw)
    if len(digits_only) < 8:
        return raw  # too short to be a phone number — leave for other rules
    groups = [g for g in re.split(r"[\s\-]+", raw.strip()) if g]
    spoken: list[str] = []
    for g in groups:
        prefix = "បូក" if g.startswith("+") else ""
        spoken.append(prefix + _digits_to_words(g))
    return " " + " ".join(spoken) + " "


def _cents_words(frac: str) -> str | None:
    if not frac or len(frac) > 2:
        return None
    cents = int(frac.ljust(2, "0"))
    return f"{_int_to_khmer_words(cents)}សេន" if cents else ""


def _money(amount: str, unit_word: str, *, cents_ok: bool) -> str:
    ascii_amt = amount.translate(_ASCII_DIGIT_MAP)
    whole, sep, frac = ascii_amt.rpartition(".") if "." in ascii_amt else (ascii_amt, "", "")
    if sep and cents_ok and "," not in frac:
        cents = _cents_words(frac)
        if cents is not None:
            head = number_to_khmer_words(whole or "0")
            return f" {head}{unit_word}" + (f" {cents}" if cents else "") + " "
    return f" {number_to_khmer_words(ascii_amt)}{unit_word} "


def _currency_before(m: re.Match) -> str:
    sym, amount = m.group(1), m.group(2)
    return _money(amount, _CURRENCY_PREFIX[sym], cents_ok=sym == "$")


def _currency_after(m: re.Match) -> str:
    amount, sym = m.group(1), m.group(2)
    word = _CURRENCY_PREFIX.get(sym) or _CURRENCY_CODES[sym.upper()]
    return _money(amount, word, cents_ok=word == "ដុល្លារ")


def _percent(m: re.Match) -> str:
    return f" {number_to_khmer_words(m.group(1))}ភាគរយ "


def _unit(m: re.Match) -> str:
    return f" {number_to_khmer_words(m.group(1))}{_UNIT_WORDS[m.group(2)]} "


def _range(m: re.Match) -> str:
    return f" {number_to_khmer_words(m.group(1))} ដល់ {number_to_khmer_words(m.group(2))} "


# ─── Repetition mark ៗ ───────────────────────────────────────────────────────
_WORD_BREAK_CHARS = " \u200b\n\t,.!?;:()\u17d4\u17d5\u17d6\u17da\"'«»“”"


def _is_bare_consonant(cluster: str) -> bool:
    """A lone consonant (optionally with a final-position sign such as bantoc
    ់) — i.e. the coda of the previous syllable, not a syllable of its own."""
    if not cluster or not ("\u1780" <= cluster[0] <= "\u17a2"):
        return False
    return all("\u17cb" <= c <= "\u17d0" for c in cluster[1:])


def _is_vowelless_onset(cluster: str) -> bool:
    """A consonant or subscript stack with no written vowel or sign (ប, ប្រ):
    the unstressed "minor" first syllable of a two-syllable word (បន្តិច, ប្រហែល)."""
    return bool(cluster) and all(
        ("\u1780" <= c <= "\u17a2") or c == _COENG for c in cluster
    )


def _previous_word(text: str) -> str:
    """The word ៗ should repeat.

    With an explicit boundary right before it (a space or ZWSP — many Khmer
    typists put ZWSP between words) the whole short run is the word. Otherwise
    word edges are invisible, so fall back to the last orthographic syllable
    (onset cluster + optional coda consonant), which is right for the
    overwhelmingly common one-syllable reduplications (ផ្សេងៗ, តូចៗ, ធំៗ),
    extended over a vowelless minor syllable at the start of the run
    (បន្តិចៗ, ប្រហែលៗ)."""
    i = len(text)
    while i > 0 and text[i - 1] not in _WORD_BREAK_CHARS:
        i -= 1
    word = text[i:]
    if not word:
        return ""
    clusters = _khmer_clusters(word)
    if not all("\u1780" <= c[0] <= "\u17ff" for c in clusters):
        return word  # Latin/mixed token — repeat it as typed
    explicit = i > 0 and text[i - 1] in " \u200b"
    if explicit and len(clusters) <= 4:
        return word
    start = len(clusters) - 1
    if _is_bare_consonant(clusters[start]) and start >= 1 and not _is_bare_consonant(clusters[start - 1]):
        start -= 1
    if start == 1 and _is_vowelless_onset(clusters[0]):
        start = 0
    return "".join(clusters[start:])


def expand_repetition_marks(text: str) -> str:
    """Expand ៗ into an explicit repetition ("ផ្សេងៗ" → "ផ្សេងផ្សេង").

    TTS models see ៗ as an opaque symbol and usually skip it, dropping the
    reduplication that carries the meaning (plural/intensity)."""
    if "ៗ" not in text:
        return text
    out: list[str] = []
    last = 0
    for m in _LEK_TOO_RE.finditer(text):
        out.append(text[last:m.start()])
        prefix = "".join(out)
        word = _previous_word(prefix)
        out.append(word)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def neutralize_parentheses(text: str) -> str:
    """VoxCPM treats a leading "(...)" as a *style instruction*, so any
    parenthesised aside that lands at the start of a chunk would be silently
    swallowed as a voice prompt. Turn inline parentheses into comma pauses."""
    text = re.sub(r"\s*[(（\[]\s*", ", ", text)
    text = re.sub(r"\s*[)）\]]\s*", ", ", text)
    text = re.sub(r"^[,\s]+", "", text)
    return re.sub(r"(,\s*){2,}", ", ", text)


def split_style_prefix(text: str) -> tuple[str, str]:
    """Split a user-typed leading "(style)" instruction off the text.

    Returns (style, body). A deliberate leading "(warm, slow)" is VoxCPM's
    inline style syntax; keeping it separate lets the worker re-apply it to
    every chunk instead of only the first one."""
    m = re.match(r"^\s*[(（]([^()（）]{1,200})[)）]\s*", text or "")
    if not m:
        return "", text
    return m.group(1).strip(), text[m.end():]


def clean_khmer_orthography(text: str) -> str:
    """Invisible-character + spelling-variant clean-up (safe for any text)."""
    out = _ZERO_WIDTH_RE.sub("", text)
    for bad, good in _ORTHOGRAPHY_FIXES:
        out = out.replace(bad, good)
    return _DUP_SIGN_RE.sub(r"\1", out)


def normalize_khmer_text(text: str, *, expand_numbers: bool = True) -> str:
    """Clean + normalize Khmer (and mixed Khmer/Latin) text before synthesis.

    - Expands ៗ into the repeated word (using ZWSP word boundaries when the
      typist left them, before they're stripped) and ។ល។ into ជាដើម.
    - Strips zero-width characters, fixes split vowels / deprecated code points.
    - Collapses runs of horizontal whitespace; keeps paragraph breaks.
    - Removes markdown/bullet decoration and quotes; parentheses become pauses
      so they can't be mistaken for VoxCPM style prompts.
    - Expands dates, clock times, phone numbers, money ($, ៛, €, ฿), percents,
      units (km, kg, °C …), numeric ranges and plain numbers to spoken Khmer
      (toggleable — pass expand_numbers=False to keep digits as typed).
    """
    if not text:
        return text
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _ETC_RE.sub(" ជាដើម", out)
    out = expand_repetition_marks(out)
    out = clean_khmer_orthography(out)
    out = _WHITESPACE_RE.sub(" ", out)
    out = _MARKUP_RE.sub(" ", out)
    out = _QUOTES_RE.sub("", out)
    out = neutralize_parentheses(out)
    if expand_numbers:
        out = out.translate(_ASCII_DIGIT_MAP)
        out = _NEGATIVE_RE.sub("ដក", out)
        out = _DATE_ISO_RE.sub(_date_iso, out)
        out = _DATE_DMY_RE.sub(_date_dmy, out)
        out = _TIME_RE.sub(_time, out)
        out = _PHONE_RE.sub(_phone, out)
        out = _SPACE_THOUSANDS_RE.sub(lambda m: m.group(1) + re.sub(r"\s", "", m.group(2)), out)
        out = _CURRENCY_BEFORE_RE.sub(_currency_before, out)
        out = _CURRENCY_AFTER_RE.sub(_currency_after, out)
        out = _PERCENT_RE.sub(_percent, out)
        out = _UNIT_RE.sub(_unit, out)
        out = _RANGE_RE.sub(_range, out)
        for pattern, repl in _ARITH_RE:
            out = pattern.sub(repl, out)
        out = _NUMBER_RE.sub(lambda m: number_to_khmer_words(m.group(0)), out)
    for pattern, repl in _SYMBOL_EXPANSIONS:
        out = pattern.sub(repl, out)
    out = _WHITESPACE_RE.sub(" ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = _NEWLINES_RE.sub("\n", out)
    # Tidy spacing the expanders introduced before punctuation.
    out = re.sub(r" +([។៕៖,.!?;:])", r"\1", out)
    return out.strip()


# ─── Cluster-safe chunking ────────────────────────────────────────────────────
#: Kinds of break that can end a chunk, strongest first. The audio joiner maps
#: each to a pause length (see core/audio_join.py).
BOUNDARY_PARAGRAPH = "paragraph"
BOUNDARY_SENTENCE = "sentence"
BOUNDARY_CLAUSE = "clause"
BOUNDARY_NONE = "none"


@dataclass
class TextChunk:
    text: str
    index: int
    #: What kind of break follows this chunk (paragraph/sentence/clause/none).
    boundary: str = BOUNDARY_SENTENCE


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
    whitespace boundaries (Khmer writers use spaces as phrase separators, and
    they keep Latin words / numbers intact); only decomposes down to
    individual Khmer orthographic clusters for a run that has no whitespace
    and still exceeds max_chars on its own."""
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


def _split_sentences(text: str) -> list[tuple[str, str]]:
    """Split on Khmer/Latin sentence terminators, keeping the terminator
    attached to the sentence that precedes it. Returns (sentence, boundary)."""
    sentences: list[tuple[str, str]] = []
    last = 0
    for m in _SENTENCE_END_RE.finditer(text):
        s = text[last:m.end()].strip()
        if s:
            sentences.append((s, BOUNDARY_SENTENCE))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        sentences.append((tail, BOUNDARY_SENTENCE))
    return sentences


def _split_soft(sentence: str, boundary: str, max_chars: int) -> list[tuple[str, str]]:
    """A sentence that's still too long: try soft-break punctuation (commas,
    colons) before falling back to a cluster-safe hard split. Every piece but
    the last ends on a weaker boundary than the sentence itself."""
    if len(sentence) <= max_chars:
        return [(sentence, boundary)]
    parts: list[str] = []
    last = 0
    for m in _SOFT_BREAK_RE.finditer(sentence):
        piece = sentence[last:m.end()].strip()
        if piece:
            parts.append(piece)
        last = m.end()
    tail = sentence[last:].strip()
    if tail:
        parts.append(tail)

    out: list[tuple[str, str]] = []
    if len(parts) <= 1:
        pieces = _cluster_safe_hard_split(sentence, max_chars)
        for i, p in enumerate(pieces):
            out.append((p, boundary if i == len(pieces) - 1 else BOUNDARY_NONE))
        return out

    for pi, p in enumerate(parts):
        end = boundary if pi == len(parts) - 1 else BOUNDARY_CLAUSE
        if len(p) > max_chars:
            pieces = _cluster_safe_hard_split(p, max_chars)
            for i, q in enumerate(pieces):
                out.append((q, end if i == len(pieces) - 1 else BOUNDARY_NONE))
        else:
            out.append((p, end))
    return out


def _pack_units(units: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    buf, buf_end = "", BOUNDARY_SENTENCE
    for text, end in units:
        if buf and len(buf) + 1 + len(text) > limit:
            chunks.append((buf, buf_end))
            buf, buf_end = text, end
        else:
            buf = f"{buf} {text}" if buf else text
            buf_end = end
    if buf:
        chunks.append((buf, buf_end))
    return chunks


def _balanced_pack(units: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    """Greedy packing leaves a runt at the end (e.g. 215 + 12 chars), and very
    short inputs are exactly where autoregressive TTS is least stable. Re-pack
    toward an even size with the same chunk count when that's possible."""
    greedy = _pack_units(units, max_chars)
    if len(greedy) <= 1:
        return greedy
    total = sum(len(t) for t, _ in units) + len(units) - 1
    target = -(-total // len(greedy))  # ceil
    for slack in (1.1, 1.2, 1.35):
        limit = min(max_chars, int(target * slack))
        balanced = _pack_units(units, limit)
        if len(balanced) == len(greedy):
            return balanced
    return greedy


def chunk_khmer_text(text: str, max_chars: int = 220) -> list[TextChunk]:
    """Split text into TTS-safe chunks: sentence-respecting, cluster-safe,
    and bin-packed up to `max_chars` so short sentences merge into fewer
    requests (fewer join seams, better use of nanovllm's batching), with the
    sizes balanced so no chunk is a tiny leftover.

    Paragraph breaks (newlines) always start a new chunk, since inserting a
    natural pause there sounds better than gluing paragraphs together.
    """
    if not text or not text.strip():
        return []

    chunks: list[tuple[str, str]] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units: list[tuple[str, str]] = []
        for s, end in _split_sentences(paragraph):
            units.extend(_split_soft(s, end, max_chars))
        packed = _balanced_pack(units, max_chars)
        if packed:
            last_text, _ = packed[-1]
            packed[-1] = (last_text, BOUNDARY_PARAGRAPH)
        chunks.extend(packed)

    return [TextChunk(text=c, index=i, boundary=b) for i, (c, b) in enumerate(chunks)]


def is_khmer_text(text: str, threshold: float = 0.15) -> bool:
    """Heuristic: True if a meaningful fraction of the letters are Khmer
    script. Used to decide whether to apply Khmer-specific normalization to
    mixed-language input rather than assuming based on the active engine."""
    letters = [c for c in text if c.isalpha() or ("\u1780" <= c <= "\u17ff")]
    if not letters:
        return False
    khmer = sum(1 for c in letters if "\u1780" <= c <= "\u17ff")
    return (khmer / len(letters)) >= threshold
