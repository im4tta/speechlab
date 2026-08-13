"""Language-code helpers and the per-model supported-language lists.

The UI target picker uses each translator's own languages() (its own code space),
so only the SOURCE code needs normalizing: Whisper emits a few legacy/alias codes
that differ from M2M-100's ISO 639-1 set.
"""

from __future__ import annotations

# Whisper (or generic) code -> the translator's canonical code.
_SOURCE_ALIASES = {
    "iw": "he",   # Hebrew (legacy)
    "ji": "yi",   # Yiddish (legacy)
    "in": "id",   # Indonesian (legacy)
    "jw": "jv",   # Javanese (Whisper spelling)
    "mo": "ro",   # Moldavian -> Romanian
}


def normalize_source(code: str | None) -> str | None:
    """Normalize a source language code, or None to let the model auto-detect."""
    if not code:
        return None
    c = code.strip().lower()
    if not c or c == "auto":
        return None
    return _SOURCE_ALIASES.get(c, c)


# (code, label) pairs. M2M-100's 100 languages (ISO 639-1). Kept explicit so the
# UI can populate the picker before the model is downloaded (like Whisper).
M2M100_LANGS: list[tuple[str, str]] = [
    ("af", "Afrikaans"), ("am", "Amharic"), ("ar", "Arabic"), ("ast", "Asturian"),
    ("az", "Azerbaijani"), ("ba", "Bashkir"), ("be", "Belarusian"), ("bg", "Bulgarian"),
    ("bn", "Bengali"), ("br", "Breton"), ("bs", "Bosnian"), ("ca", "Catalan"),
    ("ceb", "Cebuano"), ("cs", "Czech"), ("cy", "Welsh"), ("da", "Danish"),
    ("de", "German"), ("el", "Greek"), ("en", "English"), ("es", "Spanish"),
    ("et", "Estonian"), ("fa", "Persian"), ("ff", "Fulah"), ("fi", "Finnish"),
    ("fr", "French"), ("fy", "Western Frisian"), ("ga", "Irish"), ("gd", "Scottish Gaelic"),
    ("gl", "Galician"), ("gu", "Gujarati"), ("ha", "Hausa"), ("he", "Hebrew"),
    ("hi", "Hindi"), ("hr", "Croatian"), ("ht", "Haitian Creole"), ("hu", "Hungarian"),
    ("hy", "Armenian"), ("id", "Indonesian"), ("ig", "Igbo"), ("ilo", "Iloko"),
    ("is", "Icelandic"), ("it", "Italian"), ("ja", "Japanese"), ("jv", "Javanese"),
    ("ka", "Georgian"), ("kk", "Kazakh"), ("km", "Central Khmer"), ("kn", "Kannada"),
    ("ko", "Korean"), ("lb", "Luxembourgish"), ("lg", "Ganda"), ("ln", "Lingala"),
    ("lo", "Lao"), ("lt", "Lithuanian"), ("lv", "Latvian"), ("mg", "Malagasy"),
    ("mk", "Macedonian"), ("ml", "Malayalam"), ("mn", "Mongolian"), ("mr", "Marathi"),
    ("ms", "Malay"), ("my", "Burmese"), ("ne", "Nepali"), ("nl", "Dutch"),
    ("no", "Norwegian"), ("ns", "Northern Sotho"), ("oc", "Occitan"), ("or", "Oriya"),
    ("pa", "Panjabi"), ("pl", "Polish"), ("ps", "Pashto"), ("pt", "Portuguese"),
    ("ro", "Romanian"), ("ru", "Russian"), ("sd", "Sindhi"), ("si", "Sinhala"),
    ("sk", "Slovak"), ("sl", "Slovenian"), ("so", "Somali"), ("sq", "Albanian"),
    ("sr", "Serbian"), ("ss", "Swati"), ("su", "Sundanese"), ("sv", "Swedish"),
    ("sw", "Swahili"), ("ta", "Tamil"), ("th", "Thai"), ("tl", "Tagalog"),
    ("tn", "Tswana"), ("tr", "Turkish"), ("uk", "Ukrainian"), ("ur", "Urdu"),
    ("uz", "Uzbek"), ("vi", "Vietnamese"), ("wo", "Wolof"), ("xh", "Xhosa"),
    ("yi", "Yiddish"), ("yo", "Yoruba"), ("zh", "Chinese"), ("zu", "Zulu"),
]
