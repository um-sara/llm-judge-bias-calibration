"""
Tests for src/api_cache.py.

The cache lives in a single SQLite file. These tests use pytest's tmp_path
fixture to point the cache at a throwaway temporary file, so no test ever
touches the real cache at data/api_cache.sqlite.
"""
import pytest

import api_cache
from api_cache import (
    _make_key,
    cache_stats,
    cached_call,
    clear_cache,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """
    Redirect CACHE_PATH to a fresh file under pytest's tmp_path for every test.
    autouse=True means every test in this file gets this fixture automatically.
    """
    fake_path = str(tmp_path / "test_cache.sqlite")
    monkeypatch.setattr(api_cache, "CACHE_PATH", fake_path)
    yield fake_path


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def test_key_is_deterministic():
    """Same inputs must always produce the same key — otherwise the cache
    would miss on every call."""
    k1 = _make_key("claude", "claude-haiku-4-5", "hello", 10, 0.0)
    k2 = _make_key("claude", "claude-haiku-4-5", "hello", 10, 0.0)
    assert k1 == k2


def test_key_changes_when_prompt_changes():
    k1 = _make_key("claude", "claude-haiku-4-5", "hello", 10, 0.0)
    k2 = _make_key("claude", "claude-haiku-4-5", "hellO", 10, 0.0)
    assert k1 != k2


def test_key_changes_when_model_changes():
    """Different model = different key, even with identical prompts.
    Prevents cross-contamination between judges."""
    k1 = _make_key("claude", "claude-haiku-4-5", "hi", 10, 0.0)
    k2 = _make_key("claude", "claude-opus-4-7", "hi", 10, 0.0)
    assert k1 != k2


def test_key_changes_when_max_tokens_changes():
    """CoT variants use max_tokens=600, baseline uses 10 — these must
    produce different cache entries even with the same prompt."""
    k1 = _make_key("claude", "claude-haiku-4-5", "hi", 10, 0.0)
    k2 = _make_key("claude", "claude-haiku-4-5", "hi", 600, 0.0)
    assert k1 != k2


def test_key_changes_when_temperature_changes():
    k1 = _make_key("claude", "claude-haiku-4-5", "hi", 10, 0.0)
    k2 = _make_key("claude", "claude-haiku-4-5", "hi", 10, 0.7)
    assert k1 != k2


# ---------------------------------------------------------------------------
# Miss / hit behavior
# ---------------------------------------------------------------------------

def test_miss_calls_call_fn_and_returns_its_value():
    calls = {"count": 0}

    def fake_api():
        calls["count"] += 1
        return "A"

    value = cached_call("claude", "model-x", "prompt-1", 10, 0.0, fake_api)
    assert value == "A"
    assert calls["count"] == 1


def test_hit_does_not_call_call_fn():
    """Once a value is cached, call_fn must not be invoked again for the
    same key. This is the whole point of the cache."""
    calls = {"count": 0}

    def fake_api():
        calls["count"] += 1
        return "A"

    # First call — miss, call_fn runs
    cached_call("claude", "model-x", "prompt-1", 10, 0.0, fake_api)
    # Second call, identical inputs — hit, call_fn must NOT run
    cached_call("claude", "model-x", "prompt-1", 10, 0.0, fake_api)

    assert calls["count"] == 1


def test_different_prompts_are_cached_independently():
    calls = {"count": 0}

    def fake_api():
        calls["count"] += 1
        return f"response-{calls['count']}"

    v1 = cached_call("claude", "m", "prompt-a", 10, 0.0, fake_api)
    v2 = cached_call("claude", "m", "prompt-b", 10, 0.0, fake_api)
    # Same prompt as v1 — hits cache
    v3 = cached_call("claude", "m", "prompt-a", 10, 0.0, fake_api)

    assert v1 == "response-1"
    assert v2 == "response-2"
    assert v3 == "response-1"  # hit, returns original value
    assert calls["count"] == 2


def test_exceptions_propagate_and_do_not_cache():
    """If the API call blows up (rate limit, network error), a failure must
    not be cached — the next call should retry."""
    attempts = {"count": 0}

    def flaky_api():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated rate limit")
        return "A"

    # First call raises — nothing cached
    with pytest.raises(RuntimeError):
        cached_call("claude", "m", "p", 10, 0.0, flaky_api)

    # Second call: miss (no cached failure), call_fn runs, succeeds
    value = cached_call("claude", "m", "p", 10, 0.0, flaky_api)
    assert value == "A"
    assert attempts["count"] == 2


def test_raw_text_with_newlines_and_unicode_roundtrips():
    """Cache must preserve the response byte-for-byte — CoT replies contain
    newlines, punctuation, and occasionally unicode. If any of that gets
    mangled, downstream parsing would break in ways that are hard to debug."""
    raw = "Step 1: consider A.\nStep 2: consider B.\nAnswer: A \u2713"

    def api():
        return raw

    v1 = cached_call("claude", "m", "p", 600, 0.0, api)
    v2 = cached_call("claude", "m", "p", 600, 0.0, api)
    assert v1 == raw
    assert v2 == raw


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def test_stats_on_empty_cache():
    assert cache_stats() == {"total": 0, "per_judge": {}}


def test_stats_count_calls_per_judge():
    cached_call("claude",  "m1", "p1", 10, 0.0, lambda: "A")
    cached_call("claude",  "m1", "p2", 10, 0.0, lambda: "B")
    cached_call("ollama",  "m2", "p3", 10, 0.0, lambda: "tie")

    stats = cache_stats()
    assert stats["total"] == 3
    assert stats["per_judge"] == {"claude": 2, "ollama": 1}


def test_clear_cache_removes_all_entries():
    cached_call("claude", "m", "p", 10, 0.0, lambda: "A")
    assert cache_stats()["total"] == 1

    clear_cache()
    assert cache_stats()["total"] == 0
