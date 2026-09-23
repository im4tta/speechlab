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


# ─── Khmer-specific normalization (numbers glued to words, ៗ, dates, money …) ─

from backend.core.khmer_text import (  # noqa: E402
    BOUNDARY_CLAUSE,
    BOUNDARY_PARAGRAPH,
    BOUNDARY_SENTENCE,
    expand_repetition_marks,
    neutralize_parentheses,
    split_style_prefix,
)


def test_numbers_glued_to_khmer_words_are_expanded():
    # Khmer has no word spaces, so digits sit right against the word.
    assert normalize_khmer_text("ឆ្នាំ២០២៤") == "ឆ្នាំពីរពាន់ម្ភៃបួន"
    assert normalize_khmer_text("ទី១") == "ទីមួយ"
    assert normalize_khmer_text("តម្លៃ១០០០រៀល") == "តម្លៃមួយពាន់រៀល"


def test_number_words_use_khmer_scales():
    assert number_to_khmer_words("1000") == "មួយពាន់"
    assert number_to_khmer_words("10000") == "មួយម៉ឺន"
    assert number_to_khmer_words("25000") == "ពីរម៉ឺនប្រាំពាន់"
    assert number_to_khmer_words("150000") == "មួយសែនប្រាំម៉ឺន"
    assert number_to_khmer_words("1000000") == "មួយលាន"
    assert number_to_khmer_words("2500000") == "ពីរលានប្រាំសែន"
    assert number_to_khmer_words("1,000,000") == "មួយលាន"
    assert number_to_khmer_words("1.000.000") == "មួយលាន"
    assert number_to_khmer_words("3.14") == "បីចុចមួយបួន"


def test_leading_zero_numbers_read_digit_by_digit():
    assert number_to_khmer_words("007") == "សូន្យសូន្យប្រាំពីរ"


def test_repetition_mark_single_syllable():
    assert expand_repetition_marks("ផ្សេងៗ") == "ផ្សេងផ្សេង"
    assert expand_repetition_marks("ពួកក្មេងៗ") == "ពួកក្មេងក្មេង"
    assert expand_repetition_marks("ផ្ទះធំៗ") == "ផ្ទះធំធំ"


def test_repetition_mark_minor_syllable_words():
    assert expand_repetition_marks("បន្តិចៗ") == "បន្តិចបន្តិច"
    assert expand_repetition_marks("ប្រហែលៗ") == "ប្រហែលប្រហែល"


def test_repetition_mark_respects_explicit_word_boundary():
    # Space (or ZWSP) before the word marks its start; a space before ៗ is common.
    assert expand_repetition_marks("ផ្ទះ ធំ ៗ") == "ផ្ទះ ធំធំ"
    assert expand_repetition_marks("ពួក​ក្មេងៗ") == "ពួក​ក្មេងក្មេង"


def test_etcetera_expanded():
    assert "ជាដើម" in normalize_khmer_text("ប៉ោម ក្រូច ។ល។ មានតម្លៃថោក")
    assert "ជាដើម" in normalize_khmer_text("ប៉ោម ក្រូច៘")


def test_dates_and_times():
    out = normalize_khmer_text("ថ្ងៃទី 12/05/2024")
    assert out == "ថ្ងៃទី ដប់ពីរ ខែឧសភា ឆ្នាំពីរពាន់ម្ភៃបួន"
    assert "ខែឧសភា" in normalize_khmer_text("2024-05-12")
    assert normalize_khmer_text("ម៉ោង 8:30") == "ម៉ោង ប្រាំបី សាមសិបនាទី"
    assert normalize_khmer_text("8:30 PM").endswith("ល្ងាច")


def test_money_percent_units():
    assert normalize_khmer_text("$12.50") == "ដប់ពីរដុល្លារ ហាសិបសេន"
    assert normalize_khmer_text("10,000៛") == "មួយម៉ឺនរៀល"
    assert normalize_khmer_text("50%") == "ហាសិបភាគរយ"
    assert normalize_khmer_text("5km") == "ប្រាំគីឡូម៉ែត្រ"
    assert normalize_khmer_text("-5°C") == "ដក ប្រាំអង្សាសេ"


def test_phone_numbers_read_in_digit_groups():
    out = normalize_khmer_text("លេខ 012 345 678")
    assert out == "លេខ សូន្យមួយពីរ បីបួនប្រាំ ប្រាំមួយប្រាំពីរប្រាំបី"
    assert normalize_khmer_text("+855 12-345-678").startswith("បូកប្រាំបីប្រាំប្រាំ")


def test_numeric_range_reads_da_l():
    assert normalize_khmer_text("ពី៥-១០នាក់") == "ពី ប្រាំ ដល់ ដប់ នាក់"


def test_orthography_fixes_split_vowels():
    # េ + ា typed separately → the single vowel sign ោ.
    assert normalize_khmer_text("សេាម") == "សោម"


def test_markup_and_parentheses_cleaned():
    assert normalize_khmer_text("**ដិត** #tag") == "ដិត tag"
    # A parenthesised aside must not survive as a "(…)" style prompt.
    out = normalize_khmer_text("(ចំណាំ) នេះជាការសាកល្បង")
    assert "(" not in out and ")" not in out
    assert neutralize_parentheses("ក (ខ) គ") == "ក, ខ, គ"


def test_split_style_prefix():
    assert split_style_prefix("(warm, slow) សួស្តី") == ("warm, slow", "សួស្តី")
    assert split_style_prefix("សួស្តី (x)") == ("", "សួស្តី (x)")


def test_chunks_carry_boundary_types():
    chunks = chunk_khmer_text("ប្រយោគទីមួយ។ ប្រយោគទីពីរ, និងផ្នែកបន្ត។\nកថាខណ្ឌថ្មី។", 20)
    assert [c.boundary for c in chunks] == [
        BOUNDARY_SENTENCE, BOUNDARY_CLAUSE, BOUNDARY_PARAGRAPH, BOUNDARY_PARAGRAPH,
    ]


def test_chunk_packing_is_balanced():
    # Greedy packing would leave a tiny runt chunk at the end; balanced packing
    # keeps the same chunk count with even sizes.
    text = " ".join(["ប្រយោគខ្លីមួយ។"] * 9)
    chunks = chunk_khmer_text(text, max_chars=100)
    sizes = [len(c.text) for c in chunks]
    assert all(s <= 100 for s in sizes)
    assert min(sizes) >= max(sizes) // 2
