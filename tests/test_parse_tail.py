"""
Tests for the verdict-parsing helpers in src/calibrate_judge.py.

These guard the weakest link in the calibration pipeline: turning LLM output
into "A", "B", or "tie". Parse failures here silently corrupt bias measurements,
so every behavior here deserves a regression test.

As of Session 18, prompts use the Zheng et al. 2023 bracket format
(arXiv:2306.05685, Appendix A / Figure 4): "[[A]]", "[[B]]", "[[C]]" (tie).
The parser is a single regex — no natural-language fallback. If the model
failed to emit the required format, that is a parse failure, not something
to guess at. Parser is judge-agnostic: same regex handles Claude, Ollama,
Llama-70B, Qwen-32B and GPT-4o outputs.

Run from project root:
    pytest tests/
"""
import pytest

from calibrate_judge import parse_tail, map_verdict, remap_verdict


# ---------------------------------------------------------------------------
# parse_tail — happy paths (bracket format)
# ---------------------------------------------------------------------------

def test_parse_tail_plain_A():
    assert parse_tail("[[A]]") == "A"


def test_parse_tail_plain_B():
    assert parse_tail("[[B]]") == "B"


def test_parse_tail_C_maps_to_tie():
    """Zheng format uses [[C]] for tie; parser normalizes to 'tie'."""
    assert parse_tail("[[C]]") == "tie"


def test_parse_tail_is_case_insensitive():
    assert parse_tail("[[a]]") == "A"
    assert parse_tail("[[b]]") == "B"
    assert parse_tail("[[c]]") == "tie"


def test_parse_tail_reads_cot_ending_with_bracket_A():
    """CoT variants reason first, then emit the bracket verdict at the end."""
    text = "Response A is clearer and more accurate.\n\n[[A]]"
    assert parse_tail(text) == "A"


def test_parse_tail_reads_cot_ending_with_bracket_B():
    text = "After weighing both, B is stronger overall.\n\n[[B]]"
    assert parse_tail(text) == "B"


def test_parse_tail_reads_cot_ending_with_bracket_C_tie():
    text = "Both responses are roughly equivalent in quality.\n\n[[C]]"
    assert parse_tail(text) == "tie"


def test_parse_tail_ignores_trailing_punct():
    """Bracket may be followed by '.', '!', whitespace — still parses."""
    assert parse_tail("[[A]].") == "A"
    assert parse_tail("[[B]]!") == "B"
    assert parse_tail("[[A]]   \n") == "A"


def test_parse_tail_strips_leading_whitespace():
    assert parse_tail("   [[A]]") == "A"
    assert parse_tail("\n\n[[B]]") == "B"


# ---------------------------------------------------------------------------
# parse_tail — last-match-wins semantics
#
# CoT reasoning may reference [[A]] or [[B]] mid-argument before declaring
# the final verdict. The parser must take the LAST bracketed letter so the
# declared verdict wins over incidental mentions.
# ---------------------------------------------------------------------------


def test_parse_tail_last_bracket_wins_over_earlier_mention():
    text = "I initially thought [[A]] was better, but on reflection: [[B]]"
    assert parse_tail(text) == "B"


def test_parse_tail_adjacent_brackets_take_last():
    """If a model emits [[A]][[B]] (unlikely but possible), trust the last one."""
    assert parse_tail("[[A]][[B]]") == "B"


def test_parse_tail_tie_wins_when_declared_last():
    text = "Response A has strengths [[A]]. Response B has strengths [[B]]. Overall: [[C]]"
    assert parse_tail(text) == "tie"


# ---------------------------------------------------------------------------
# parse_tail — no-answer cases
#
# The new parser is intentionally strict. If the model didn't emit the
# required bracket format, that counts as a parse failure and surfaces
# surface in the no-answer column instead of guessing.
# ---------------------------------------------------------------------------


def test_parse_tail_empty_string_is_no_answer():
    assert parse_tail("") == "no answer"


def test_parse_tail_whitespace_only_is_no_answer():
    assert parse_tail("   \n\n  ") == "no answer"


def test_parse_tail_bare_letter_is_no_answer():
    """Under the new format contract, a bare 'A' is not a valid verdict."""
    assert parse_tail("A") == "no answer"


def test_parse_tail_markdown_bold_is_no_answer():
    """Old parser handled **A**; new parser does not. Prompts now require brackets."""
    assert parse_tail("**A**") == "no answer"


def test_parse_tail_natural_language_without_bracket_is_no_answer():
    """Old parser had a NL fallback; new parser does not. Keep it strict."""
    assert parse_tail("Response A is clearly better than Response B.") == "no answer"


def test_parse_tail_invalid_bracket_letter_is_no_answer():
    """Only [[A]], [[B]], [[C]] are recognized. [[D]] is a failure."""
    assert parse_tail("[[D]]") == "no answer"


def test_parse_tail_single_bracket_forms_are_accepted():
    """
    Regex is `\\[+([ABCabc])\\]+` — accommodates Llama 3.2 3B, which reliably
    emits single-bracket `[A]` / `[B]` / `[C]` instead of Zheng's double form.
    50 of 52 Ollama Baseline parse failures in the first v2 run were this
    single-bracket pattern. Stronger models (Claude, GPT-4) still emit double
    brackets correctly; the relaxed regex covers both.
    """
    assert parse_tail("[A]") == "A"
    assert parse_tail("[B]") == "B"
    assert parse_tail("[C]") == "tie"
    # Lopsided forms (asymmetric bracket counts) also match under \[+...\]+
    assert parse_tail("[A]]") == "A"
    assert parse_tail("[[A]") == "A"


def test_parse_tail_word_in_brackets_is_no_answer():
    """[[tie]] (old-style word) does not match the single-letter regex."""
    assert parse_tail("[[tie]]") == "no answer"


def test_parse_tail_gibberish_is_no_answer():
    assert parse_tail("hmm I'm not sure about this one") == "no answer"


# ---------------------------------------------------------------------------
# map_verdict — A/B/tie → model_a/model_b/tie
# ---------------------------------------------------------------------------

def test_map_verdict_A_to_model_a():
    assert map_verdict("A") == "model_a"


def test_map_verdict_B_to_model_b():
    assert map_verdict("B") == "model_b"


def test_map_verdict_tie_stays_tie():
    assert map_verdict("tie") == "tie"


def test_map_verdict_no_answer_stays_distinct_from_tie():
    """
    Parse failures must NOT collapse into tie — that silently inflated
    tie_rate, tie_sensitivity, and agreement on human-tie rows in earlier
    runs (a parse failure on a tie row was wrongly credited as a correct
    tie call). "no answer" rides through as its own category so every
    downstream metric treats it as a miss.
    """
    assert map_verdict("no answer") == "no answer"


# ---------------------------------------------------------------------------
# remap_verdict — used by swap-consistency to flip the B-first call back
# ---------------------------------------------------------------------------

def test_remap_verdict_swaps_A_and_B():
    assert remap_verdict("A") == "B"
    assert remap_verdict("B") == "A"


def test_remap_verdict_passes_tie_through_unchanged():
    assert remap_verdict("tie") == "tie"


def test_remap_verdict_passes_no_answer_through_unchanged():
    assert remap_verdict("no answer") == "no answer"
