"""Argos Translate's supported languages + package-resolution helper.

The language list mirrors the Argos package index (github.com/argosopentech/
argospm-index). Argos translates via an English pivot, so any pair reduces to
at most two installed packages.
"""

from __future__ import annotations

# (code, label) — the languages present in the Argos index. Stable; embedded so
# the UI picker populates before any package is installed.
ARGOS_LANGS: list[tuple[str, str]] = [
    ("ar", "Arabic"), ("az", "Azerbaijani"), ("bg", "Bulgarian"), ("bn", "Bengali"),
    ("ca", "Catalan"), ("cs", "Czech"), ("da", "Danish"), ("de", "German"),
    ("el", "Greek"), ("en", "English"), ("eo", "Esperanto"), ("es", "Spanish"),
    ("et", "Estonian"), ("eu", "Basque"), ("fa", "Persian"), ("fi", "Finnish"),
    ("fr", "French"), ("ga", "Irish"), ("gl", "Galician"), ("he", "Hebrew"),
    ("hi", "Hindi"), ("hu", "Hungarian"), ("id", "Indonesian"), ("it", "Italian"),
    ("ja", "Japanese"), ("ko", "Korean"), ("ky", "Kyrgyz"), ("lt", "Lithuanian"),
    ("lv", "Latvian"), ("ms", "Malay"), ("nb", "Norwegian"), ("nl", "Dutch"),
    ("pb", "Portuguese (Brazil)"), ("pl", "Polish"), ("pt", "Portuguese"),
    ("ro", "Romanian"), ("ru", "Russian"), ("sk", "Slovak"), ("sl", "Slovenian"),
    ("sq", "Albanian"), ("sv", "Swedish"), ("sw", "Swahili"), ("th", "Thai"),
    ("tl", "Tagalog"), ("tr", "Turkish"), ("uk", "Ukrainian"), ("ur", "Urdu"),
    ("vi", "Vietnamese"), ("zh", "Chinese"), ("zt", "Chinese (traditional)"),
]


def required_packages(source: str, target: str) -> list[tuple[str, str]]:
    """The (from,to) packages needed to translate source->target via English pivot."""
    pairs: list[tuple[str, str]] = []
    if source != "en":
        pairs.append((source, "en"))
    if target != "en":
        pairs.append(("en", target))
    return pairs
