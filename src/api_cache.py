"""
Disk-backed cache for LLM API calls.

Every judge call in this project is deterministic in theory (temperature=0)
but not in practice: providers route to different nodes, retry silently,
and occasionally return slightly different tokens for the same prompt.
That makes reruns expensive AND irreproducible — the opposite of what a
reliability audit needs.

This module wraps the API call. Before hitting the provider, it checks a
local SQLite cache keyed on (judge, model, prompt, max_tokens, temperature).
  - Hit  -> return cached response, no network call.
  - Miss -> call the provider, store the response, return it.

The raw response text is cached (not the parsed verdict) so that if parsing
logic changes later, historical calls can be re-parsed for free, and so that
silent parse failures remain diagnosable after the fact.

Storage: data/api_cache.sqlite (gitignored alongside data/).

Typical use:
    from api_cache import cached_call

    def _make_api_call():
        response = client.messages.create(...)
        return response.content[0].text

    raw_text = cached_call(
        judge="claude",
        model="claude-haiku-4-5",
        prompt=prompt,
        max_tokens=10,
        temperature=0.0,
        call_fn=_make_api_call,
    )
"""

import hashlib
import json
import os
import sqlite3
from typing import Callable, Optional

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "api_cache.sqlite",
)


def _make_key(
    judge: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    n: Optional[int] = None,
) -> str:
    """
    Stable SHA256 hash of all inputs. sort_keys=True guarantees the byte
    representation is dict-order-insensitive — same logical inputs always
    produce the same key.

    n: experiment sample size. When provided (calibration path), it's part
    of the hash so an n=400 run and an n=1689 run on the same logical inputs
    produce distinct cache entries. When omitted (bias-suite path), the hash
    is computed as before so existing bias-suite entries remain valid.
    """
    payload = {
        "judge": judge,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if n is not None:
        payload["n"] = n
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _connect():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  judge TEXT,"
        "  model TEXT,"
        "  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "  n INTEGER"
        ")"
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
    if "n" not in cols:
        conn.execute("ALTER TABLE cache ADD COLUMN n INTEGER")
    return conn


def cached_call(
    judge: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    call_fn: Callable[[], str],
    n: Optional[int] = None,
) -> str:
    """
    Return the cached response for this call, or invoke call_fn and cache it.

    call_fn is a zero-arg function that performs the actual API request and
    returns the raw response text. Keeping it as a closure means this module
    stays provider-agnostic — works for Anthropic, Ollama, OpenRouter, etc.

    call_fn is allowed to raise — exceptions propagate out unchanged and
    nothing is cached. That way transient failures (rate limits, timeouts)
    don't poison the cache. Empty responses are likewise not cached, so a
    dropped completion is retried on the next run rather than served forever.

    n: optional experiment sample size. When provided, it's part of the key
    so different-n runs don't collide and it's stored in the n column for
    later querying. Bias-suite callers can omit it.
    """
    key = _make_key(judge, model, prompt, max_tokens, temperature, n)

    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            return row[0]

    value = call_fn()

    # Don't cache empty responses. An empty string usually means a transient
    # provider hiccup (rate limit, dropped completion) rather than a real "no
    # answer"; caching it would serve the blank forever. Leaving it uncached
    # lets the next run retry the call.
    if value:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, judge, model, n) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, value, judge, model, n),
            )
            conn.commit()
    return value


def get_cached(
    judge: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    n: Optional[int] = None,
) -> Optional[str]:
    """
    Return the cached raw response for this call, or None if not present.
    Read-only — never calls the API. Used by diagnostic tooling that needs
    access to raw text *after* a call has already gone through cached_call.
    """
    if not os.path.exists(CACHE_PATH):
        return None
    key = _make_key(judge, model, prompt, max_tokens, temperature, n)
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def cache_stats() -> dict:
    """Row count overall and per judge, for quick inspection."""
    if not os.path.exists(CACHE_PATH):
        return {"total": 0, "per_judge": {}}
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        per_judge = dict(
            conn.execute(
                "SELECT COALESCE(judge, '(unknown)'), COUNT(*) "
                "FROM cache GROUP BY judge"
            ).fetchall()
        )
    return {"total": total, "per_judge": per_judge}


def clear_cache():
    """Wipe the cache. Use when prompt templates change materially and you
    want a fresh run rather than accumulating stale entries."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)


if __name__ == "__main__":
    stats = cache_stats()
    print(f"Cache path: {CACHE_PATH}")
    print(f"Total cached calls: {stats['total']}")
    if stats["per_judge"]:
        print("By judge:")
        for judge, count in sorted(stats["per_judge"].items()):
            print(f"  {judge}: {count}")
