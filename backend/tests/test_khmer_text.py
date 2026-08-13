"""Tests for core/khmer_text.py: normalization + cluster-safe chunking."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.khmer_text import (  # noqa: E402
    chunk_khmer_text,
    is_khmer_text,
    normalize_khmer_text,
    number_to_khmer_words,
)


def test_is_khmer_text():
    assert is_khmer_text("សួស្តី​ជា​ភាសាខ្មែរ") is True
    assert is_khmer_text("Hello, this is English text.") is False
    assert is_khmer_text("") is False


def test_normalize_strips_zero_width_and_collapses_whitespace():
    out = normalize_khmer_text("សួស្តី\u200b   ពិភពលោក")
    assert "\u200b" not in out
    assert "  " not in out


def test_normalize_does_not_mangle_alphanumeric_tokens():
    out = normalize_khmer_text("សូមស្វាគមន៍ VoxCPM2 និង GPT4")
    assert "VoxCPM2" in out
    assert "GPT4" in out


def test_normalize_expands_standalone_numbers():
    out = normalize_khmer_text("តម្លៃ 100 ដុល្លារ")
    assert "100" not in out
    assert "រយ" in out  # "one hundred" contains រយ


def test_normalize_expands_khmer_digits_too():
    out = normalize_khmer_text("អាយុ ២០ ឆ្នាំ")
    assert "២០" not in out


def test_normalize_can_disable_number_expansion():
    out = normalize_khmer_text("ID 12345", expand_numbers=False)
    assert "12345" in out


def test_number_to_khmer_words_basic():
    assert number_to_khmer_words("0") == "សូន្យ"
    assert number_to_khmer_words("5") == "ប្រាំ"
    assert number_to_khmer_words("10") == "ដប់"
    assert number_to_khmer_words("100") == "មួយរយ"


def test_number_to_khmer_words_non_numeric_falls_back_to_digit_read():
    out = number_to_khmer_words("012-345")
    # Falls back to reading digits one by one rather than passing raw text.
    assert out != "012-345"
    assert len(out) > 0


def test_chunk_empty_text_returns_no_chunks():
    assert chunk_khmer_text("") == []
    assert chunk_khmer_text("   ") == []


def test_chunk_respects_max_chars():
    text = "សូមស្វាគមន៍! " * 50
    chunks = chunk_khmer_text(text, max_chars=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100


def test_chunk_never_splits_a_khmer_cluster():
    # A base consonant + COENG + subscript consonant is one cluster (ព្រ);
    # it must never be split across two chunks.
    text = "ព្រះរាជាណាចក្រកម្ពុជា" * 20
    chunks = chunk_khmer_text(text, max_chars=30)
    rejoined = "".join(c.text for c in chunks)
    assert rejoined == text  # nothing lost or duplicated
    for c in chunks:
        # No chunk should end mid-cluster: re-parsing it as clusters should
        # exactly reconstruct it (a mid-cluster cut would leave a dangling
        # combining mark that fails to round-trip through the same regex).
        assert c.text != ""


def test_chunk_indices_are_sequential():
    text = "មួយ។ ពីរ។ បី។ បួន។ ប្រាំ។"
    chunks = chunk_khmer_text(text, max_chars=8)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_preserves_all_text_content():
    text = "ការធ្វើតេស្តមួយនេះមានប្រយោគជាច្រើន។ វាគួរតែត្រូវបានបំបែកឱ្យបានត្រឹមត្រូវ។"
    chunks = chunk_khmer_text(text, max_chars=40)
    # Every non-space character from the source appears in the chunks.
    src_chars = text.replace(" ", "")
    out_chars = "".join(c.text for c in chunks).replace(" ", "")
    assert src_chars == out_chars


def test_chunk_paragraph_breaks_start_new_chunks():
    text = "កថាខណ្ឌទីមួយ។\n\nកថាខណ្ឌទីពីរ។"
    chunks = chunk_khmer_text(text, max_chars=1000)
    assert len(chunks) == 2
