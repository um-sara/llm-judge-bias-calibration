"""
Calibration experiment: can prompt engineering or swap-consistency fix tie under-calling?

Runs 6 variants (5 prompt + swap-consistency) on a stratified sample for one judge (default n=400).
Uses proper paired statistics: exact McNemar's pairwise tests against baseline with
Benjamini-Hochberg correction. Cochran's Q is reported as an omnibus check alongside
the pairwise tests (it is not used as a gate; BH controls the false-discovery rate).

Intended primary variant: "Explicit tie" (pre-registered before data collection).

Supports 5 judges: claude, ollama, llama70b, qwen32b, gpt4o. All share the same
5 prompt templates (bracket format, Zheng et al. 2023) and the same parse_tail —
judge selection only changes the API/model dispatch.

Usage:
    python src/calibrate_judge.py --judge claude
    python src/calibrate_judge.py --judge ollama
    python src/calibrate_judge.py --judge llama70b
    python src/calibrate_judge.py --judge qwen32b
    python src/calibrate_judge.py --judge gpt4o
"""

import argparse
import ast
import os
import re
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from api_cache import cached_call

SEED = 42

# Prompt variants (pre-registered; do not modify after data collection begins)

# Output-format line matches Zheng et al. 2023 (arXiv:2306.05685) Appendix A /
# Figure 4 verbatim: [[A]] / [[B]] / [[C]]. Each variant's body
# (CoT steps, rubric criteria, explicit-tie instruction) is kept unchanged — only the
# final format instruction is replaced so that parsing becomes a single regex
# and parse-failure rate for long-reasoning variants collapses to ~0.

BASELINE_PROMPT = """You are evaluating two responses to a question. Decide which response is better.

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Output your verdict by strictly following this format: "[[A]]" if Response A is better, "[[B]]" if Response B is better, and "[[C]]" for a tie."""


COT_PROMPT = """You are evaluating two responses to a question. Decide which response is better.

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Think step by step:
1. What does Response A do well or poorly?
2. What does Response B do well or poorly?
3. Which is better overall?

After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if Response A is better, "[[B]]" if Response B is better, and "[[C]]" for a tie."""


TIE_PROMPT = """You are evaluating two responses to a question. Decide which response is better.

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

If one response is clearly better, output "[[A]]" or "[[B]]".
If the responses are equally good or too close to call, output "[[C]]" for a tie.
Output your verdict by strictly following this format: "[[A]]" if Response A is better, "[[B]]" if Response B is better, and "[[C]]" for a tie."""


COT_TIE_PROMPT = """You are evaluating two responses to a question. Decide which response is better.

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Think step by step:
1. What does Response A do well or poorly?
2. What does Response B do well or poorly?
3. Which is better overall, or are they equally good?

If one response is clearly better, end with "[[A]]" or "[[B]]".
If the responses are equally good or too close to call, end with "[[C]]" for a tie.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if Response A is better, "[[B]]" if Response B is better, and "[[C]]" for a tie."""


RUBRIC_PROMPT = """You are a fair and rigorous evaluator. Judge which response better answers the question.

Criteria (in order of importance):
1. Accuracy — is the information correct?
2. Completeness — does it fully address the question?
3. Clarity — is it easy to understand?

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Based on the criteria above, which response is better?
If one is clearly better, output "[[A]]" or "[[B]]".
If they are equally good, output "[[C]]" for a tie.
Output your verdict by strictly following this format: "[[A]]" if Response A is better, "[[B]]" if Response B is better, and "[[C]]" for a tie."""


# (name, prompt_template, max_tokens)
# Uniform 1024 across all variants: removes token budget as a confounder so
# any parse failure is unambiguously format noncompliance, not truncation.
# Earlier 600 cap on CoT/Rubric correlated with 8-12% parse failures on
# Claude and Ollama; gpt4o/llama70b finished in <600 regardless.
PROMPT_VARIANTS = [
    ("Baseline",           BASELINE_PROMPT, 1024),
    ("Chain-of-thought",   COT_PROMPT,      1024),
    ("Explicit tie",       TIE_PROMPT,      1024),  # pre-registered primary
    ("CoT + explicit tie", COT_TIE_PROMPT,  1024),
    ("Rubric-based",       RUBRIC_PROMPT,   1024),
]

# Map --variant CLI choice → PROMPT_VARIANTS display name. Explicit map because
# auto-derivation (lower+replace-space) breaks on names containing punctuation
# like "CoT + explicit tie" → would normalize to "cot-+-explicit-tie".
VARIANT_CLI_TO_NAME = {
    "baseline":          "Baseline",
    "chain-of-thought":  "Chain-of-thought",
    "explicit-tie":      "Explicit tie",
    "cot-explicit-tie":  "CoT + explicit tie",
    "rubric-based":      "Rubric-based",
}

# Helpers

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


def parse_conversation(raw) -> list:
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def extract_question_and_answer(conversation: list) -> tuple[str, str]:
    question, answer = "", ""
    for turn in conversation:
        if turn["role"] == "user" and not question:
            question = turn["content"]
        elif turn["role"] == "assistant" and not answer:
            answer = turn["content"]
        if question and answer:
            break
    return question, answer


def parse_tail(text: str) -> str:
    """
    Extract verdict from model output in Zheng et al. bracket format.

    Returns "A", "B", "tie", or "no answer".

    Prompts instruct the model to output "[[A]]" / "[[B]]" / "[[C]]" (tie).
    The regex is relaxed to `\\[+([ABCabc])\\]+` — one or more `[`, the
    verdict letter, one or more `]`. Reason: Llama 3.2 3B reliably drops
    the outer bracket pair and emits `[A]` / `[B]` / `[C]` instead. 50 of 52
    Ollama Baseline parse failures on the first v2 run (2026-04-21) were
    single-bracket outputs. Stronger models (Claude, GPT-4) emit double
    brackets correctly, and this regex handles both.

    The LAST match is taken so CoT reasoning that references `[A]` / `[B]`
    mid-argument still resolves to the declared verdict at the end.

    No natural-language fallback: if the model didn't emit *any* bracketed
    letter, that is a parse failure ("no answer"). The bracket requirement
    keeps the parser strict — NL prose like "Response A is better" will not
    silently match.
    """
    if not text:
        return "no answer"

    matches = re.findall(r"\[+([ABCabc])\]+", text)
    if not matches:
        return "no answer"

    last = matches[-1].upper()
    return {"A": "A", "B": "B", "C": "tie"}[last]


def call_claude(prompt: str, max_tokens: int, n: int) -> str:
    from judge_claude import client

    def _call_api():
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raw = cached_call(
        judge="claude",
        model="claude-haiku-4-5",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        call_fn=_call_api,
        n=n,
    )
    return parse_tail(raw)


def call_ollama_variant(prompt: str, n: int) -> str:
    import ollama

    def _call_api():
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    # max_tokens=0 as placeholder — ollama doesn't take a token cap. Prompt
    # differs per variant so cache keys still separate cleanly.
    raw = cached_call(
        judge="ollama",
        model="llama3.2:3b",
        prompt=prompt,
        max_tokens=0,
        temperature=0.0,
        call_fn=_call_api,
        n=n,
    )
    return parse_tail(raw)


def call_openrouter_variant(prompt: str, model_key: str, max_tokens: int = 10, n: Optional[int] = None) -> str:
    import time
    from openai import OpenAI, RateLimitError
    from judge_openrouter import MODEL_IDS, PROVIDER_MAP

    model_id = MODEL_IDS[model_key]
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    extra = {
        "provider": {
            "order": PROVIDER_MAP[model_id],
            "allow_fallbacks": False,
        }
    }

    # Qwen3-32B: disable thinking mode via /no_think suffix. Token budget
    # is the caller's max_tokens unchanged — Qwen gets the same headroom as
    # every other judge so CoT/Rubric variants aren't silently truncated.
    content = prompt + " /no_think" if model_id == "qwen/qwen3-32b" else prompt
    actual_max_tokens = max_tokens

    def _call_api():
        # Retry loop inside _call_api so rate-limit exceptions never reach the
        # cache layer — only successful responses get persisted.
        max_retries = 5
        backoff = 10
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=actual_max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": content}],
                    extra_body=extra,
                )
                break
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = backoff * (2 ** attempt)
                print(f"  Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries - 1}...")
                time.sleep(wait)

        if not response.choices:
            return ""
        text = response.choices[0].message.content
        return text if text else ""

    raw = cached_call(
        judge=f"openrouter:{model_key}",
        model=model_id,
        prompt=content,
        max_tokens=actual_max_tokens,
        temperature=0.0,
        call_fn=_call_api,
        n=n,
    )
    if not raw:
        return "no answer"
    return parse_tail(raw)


def map_verdict(verdict: str) -> str:
    """
    Map raw verdict token to the human_winner space for agreement checks.

    "no answer" stays as "no answer" — explicitly NOT collapsed into "tie".
    The old behavior (folding parse failures into tie) silently inflated
    tie_rate, tie_sensitivity, and agreement on human-tie rows (a parse
    failure on a tie row was wrongly credited as "correctly called a tie").
    Keeping "no answer" distinct means those rows count as misses in every
    metric, which is the honest accounting for a reliability audit.
    """
    if verdict == "A":
        return "model_a"
    if verdict == "B":
        return "model_b"
    if verdict == "tie":
        return "tie"
    return "no answer"


def remap_verdict(verdict: str) -> str:
    """Flip A↔B for swap-consistency (B-first call mapped back to original space)."""
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


# Strategy B: permutation-based self-consistency

# Stratified sample

def build_sample(df_full: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified sample of size n from MT-Bench turn=1, random_state=42.

    The per-stratum loop floors each share (int()), so it lands a few rows short
    of n; the top-up draws the remainder from rows NOT already sampled. That
    dedup must compare against turn1's ORIGINAL index, so reset_index is deferred
    until the very end. Resetting before the top-up (the prior bug) relabelled
    the sampled rows 0..len-1 and then filtered turn1's original labels against
    those positions — a meaningless comparison that let the top-up redraw an
    already-sampled row, producing a duplicate subject. Duplicates violate the
    one-observation-per-subject assumption of the downstream paired tests
    (McNemar's, Cochran's Q).
    """
    turn1 = df_full[df_full["turn"] == 1].copy()
    sample_parts = []
    for _, g in turn1.groupby("winner"):
        n_stratum = min(len(g), int(n * len(g) / len(turn1)))
        sample_parts.append(g.sample(n_stratum, random_state=SEED))
    sample = pd.concat(sample_parts)
    if len(sample) < n:
        remaining = turn1[~turn1.index.isin(sample.index)]
        extra = remaining.sample(n - len(sample), random_state=SEED)
        sample = pd.concat([sample, extra])
    # Invariant on the ORIGINAL turn1 index (each row's identity): no row may be
    # drawn twice. Checked before reset_index discards those labels. Keyed on the
    # index, not (question_id, model_a, model_b) — MT-Bench has multiple annotator
    # rows per pair, so pair-key collisions are legitimate distinct subjects.
    assert sample.index.is_unique, "build_sample drew the same row twice (top-up dedup failed)"
    return sample.reset_index(drop=True)


# Evaluate one variant

def evaluate_variant(
    name: str,
    sample: pd.DataFrame,
    call_fn,
) -> dict:
    """
    Run one variant on the sample. Returns per-row verdicts and aggregate metrics.
    call_fn(question, answer_a, answer_b) -> "A" | "B" | "tie" | "no answer"
    """
    verdicts = []
    no_answer = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        conv_a = parse_conversation(row["conversation_a"])
        conv_b = parse_conversation(row["conversation_b"])
        question, answer_a = extract_question_and_answer(conv_a)
        _, answer_b = extract_question_and_answer(conv_b)

        verdict = call_fn(question, answer_a, answer_b)
        if verdict == "no answer":
            no_answer += 1
        verdicts.append(verdict)

        if (i + 1) % 25 == 0:
            human_winners = sample["human_winner"].iloc[:i+1].values
            agreed_so_far = sum(
                map_verdict(v) == hw
                for v, hw in zip(verdicts, human_winners)
            )
            print(f"    [{i+1}/{len(sample)}] agreement so far: {100*agreed_so_far/(i+1):.0f}%")

    verdicts_s = pd.Series(verdicts, index=sample.index)
    judge_winners = verdicts_s.map(map_verdict)
    agreed = (judge_winners == sample["human_winner"].values).astype(int)

    # Tie sensitivity/specificity
    tie_mask = sample["human_winner"].values == "tie"
    nontie_mask = np.logical_not(tie_mask)
    is_tie_pred = (judge_winners == "tie").astype(int).values

    n_tie = tie_mask.sum()
    n_nontie = nontie_mask.sum()

    tie_sensitivity = is_tie_pred[tie_mask].mean() if n_tie > 0 else float("nan")
    tie_specificity = (1 - is_tie_pred[nontie_mask]).mean() if n_nontie > 0 else float("nan")
    tie_rate = is_tie_pred.mean()
    agree_pct = agreed.mean()

    ci_agree = wilson_ci(int(agreed.sum()), len(agreed))
    ci_tie_sens = wilson_ci(int(is_tie_pred[tie_mask].sum()), n_tie) if n_tie > 0 else (float("nan"), float("nan"))

    # Per-row verdict records (for verdicts CSV)
    verdict_rows = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        verdict_rows.append({
            "question_id": row.get("question_id"),
            "model_a": row.get("model_a"),
            "model_b": row.get("model_b"),
            "human_winner": row.get("human_winner"),
            "variant": name,
            "verdict": verdicts[idx],
            "judge_winner": map_verdict(verdicts[idx]),
            "agreed": int(map_verdict(verdicts[idx]) == row.get("human_winner")),
            "is_tie_pred": int(map_verdict(verdicts[idx]) == "tie"),
        })

    return {
        "variant": name,
        "n": len(sample),
        "no_answer": no_answer,
        "agree_pct": agree_pct,
        "agree_ci_lo": ci_agree[0],
        "agree_ci_hi": ci_agree[1],
        "tie_rate": tie_rate,
        "tie_sensitivity": tie_sensitivity,
        "tie_sens_ci_lo": ci_tie_sens[0],
        "tie_sens_ci_hi": ci_tie_sens[1],
        "tie_specificity": tie_specificity,
        "_agreed": agreed.values,           # shape (n,) for McNemar's
        "_is_tie": is_tie_pred,             # shape (n,) for McNemar's
        "_tie_mask": tie_mask,              # shape (n,) boolean
        "_verdict_rows": verdict_rows,      # per-row records for verdicts CSV
    }


# Statistics

def cochrans_q(outcome_matrix: np.ndarray) -> tuple[float, float]:
    """
    Cochran's Q test for K paired binary conditions on the same n subjects.
    outcome_matrix: shape (n_subjects, K_conditions), binary (0/1).
    Returns (Q_statistic, p_value).
    """
    _, k = outcome_matrix.shape
    row_totals = outcome_matrix.sum(axis=1)
    col_totals = outcome_matrix.sum(axis=0)
    grand_total = outcome_matrix.sum()

    Q = (k - 1) * (k * np.sum(col_totals**2) - grand_total**2) / \
        (k * grand_total - np.sum(row_totals**2))
    p = 1 - stats.chi2.cdf(Q, df=k - 1)
    return float(Q), float(p)


def exact_mcnemar_with_midp(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Exact binomial McNemar's test comparing paired binary arrays a (baseline) and b (variant).
    Returns (statistic, exact_p, mid_p).
    Uses exact=True (binomial) because discordant cell counts are often <25.
    Mid-p correction reduces conservatism of exact test.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    # Build 2x2 table: rows=baseline, cols=variant
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())  # baseline 0, variant 1
    n10 = int(((a == 1) & (b == 0)).sum())  # baseline 1, variant 0
    n11 = int(((a == 1) & (b == 1)).sum())
    table = np.array([[n11, n10], [n01, n00]])

    result = mcnemar(table, exact=True)
    exact_p = result.pvalue

    # Mid-p: P(X >= c) + 0.5 * P(X == c) where X ~ Bin(n01+n10, 0.5), c = max(n01, n10)
    disc = n01 + n10
    if disc == 0:
        mid_p = 1.0
    else:
        c = max(n01, n10)
        binom = stats.binom(disc, 0.5)
        mid_p = binom.sf(c - 1) - 0.5 * binom.pmf(c)
        mid_p = min(1.0, 2 * mid_p)  # two-sided

    return result.statistic, exact_p, mid_p


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH FDR correction. Returns adjusted p-values (compared against q at the call site)."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), 1):
        adj = min(prev, p * n / (n - rank + 1))
        adjusted[i] = adj
        prev = adj
    return adjusted


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni FWER correction. Returns adjusted p-values."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 0.0
    for rank, (i, p) in enumerate(indexed, 1):
        adj = max(prev, p * (n - rank + 1))
        adjusted[i] = min(adj, 1.0)
        prev = adjusted[i]
    return adjusted


def mde_mcnemar(n: int, p_baseline: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Approximate MDE for McNemar's test: minimum p_variant detectable at given power.
    Uses normal approximation: discordant pairs m ≈ n*(p_b*(1-p_v) + p_v*(1-p_b)).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # Solve for p_variant given target power — scan numerically
    for p_v in np.arange(p_baseline + 0.01, 1.0, 0.01):
        b = n * p_baseline * (1 - p_v)
        c = n * p_v * (1 - p_baseline)
        m = b + c
        if m < 1:
            continue
        z = (abs(c - b) - 1) / np.sqrt(m)
        if z >= z_beta:  # approximate power check
            achieved_power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)
            if achieved_power >= power:
                return p_v
    return float("nan")


def main(judge: str, full: bool = False, variant_filter: str = None, n: int = 400):
    OPENROUTER_JUDGES = ("llama70b", "qwen32b", "gpt4o")

    # Lazy imports — only load what's needed for the selected judge
    if judge == "claude":
        from judge_claude import client as _claude_client  # validates API key early
        assert _claude_client is not None

    run_label = "full" if full else f"n{n}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # v2: bracket-format prompts + single-regex parse_tail (Zheng et al. 2023).
    # v1 runs used free-text prompts and a heuristic parser — files from that
    # era do NOT have the _v2_ token, so `calibration_<judge>_v2_*.csv` globs
    # cleanly separate the two generations.
    version = "v2"
    print(f"CALIBRATION EXPERIMENT — judge: {judge.upper()} | mode: {'FULL (n=1689)' if full else f'SAMPLED (n={n})'} | format: {version}")
    print("Intended primary variant: Explicit tie")

    # --- Load data ---
    print("Loading data...")
    df_full = pd.read_csv("data/mt_bench.csv")
    if full:
        sample = df_full[df_full["turn"] == 1].copy().reset_index(drop=True)
        print(f"Full run: using all {len(sample)} turn=1 rows")
    else:
        sample = build_sample(df_full, n)

    # Strata breakdown (using mt_bench 'winner' column)
    strata = sample["winner"].value_counts()
    print(f"Sample: n={len(sample)} | " +
          " | ".join(f"{k}: {v}" for k, v in strata.items()))

    # Use mt_bench 'winner' as ground truth (individual annotator label, per Zheng et al. 2023)
    sample["human_winner"] = sample["winner"]
    tie_mask = (sample["human_winner"] == "tie").values

    # run_n is the actual sample size for this invocation; threaded into
    # cache keys so n=400 and n=1689 runs on the same logical inputs produce
    # distinct cache entries (no cross-contamination between sample sizes).
    run_n = len(sample)

    # --- Build call functions ---
    if judge == "claude":
        def make_prompt_fn(template, max_tokens):
            def fn(question, answer_a, answer_b):
                prompt = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
                return call_claude(prompt, max_tokens, run_n)
            return fn

        def swap_fn(question, answer_a, answer_b):
            def _call(q, a, b):
                prompt = BASELINE_PROMPT.format(question=q, answer_a=a, answer_b=b)
                return call_claude(prompt, 1024, run_n)
            v1 = _call(question, answer_a, answer_b)
            v2 = remap_verdict(_call(question, answer_b, answer_a))
            return v1 if v1 == v2 else "tie"
    elif judge == "ollama":
        def make_prompt_fn(template, _max_tokens):
            def fn(question, answer_a, answer_b):
                prompt = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
                return call_ollama_variant(prompt, run_n)
            return fn

        def swap_fn(question, answer_a, answer_b):
            v1 = call_ollama_variant(BASELINE_PROMPT.format(
                question=question, answer_a=answer_a, answer_b=answer_b), run_n)
            v2 = remap_verdict(call_ollama_variant(BASELINE_PROMPT.format(
                question=question, answer_a=answer_b, answer_b=answer_a), run_n))
            return v1 if v1 == v2 else "tie"
    elif judge in OPENROUTER_JUDGES:
        def make_prompt_fn(template, max_tokens):
            def fn(question, answer_a, answer_b):
                prompt = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
                return call_openrouter_variant(prompt, judge, max_tokens, run_n)
            return fn

        def swap_fn(question, answer_a, answer_b):
            v1 = call_openrouter_variant(
                BASELINE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b),
                judge, 1024, run_n)
            v2 = remap_verdict(call_openrouter_variant(
                BASELINE_PROMPT.format(question=question, answer_a=answer_b, answer_b=answer_a),
                judge, 1024, run_n))
            return v1 if v1 == v2 else "tie"
    else:
        raise ValueError(f"Unknown judge: '{judge}'. Supported: claude, ollama, llama70b, qwen32b, gpt4o")

    # --- Run prompt variants ---
    # Baseline is always run fresh as the paired reference for McNemar's. This
    # avoids temperature-0 drift between any prior bias-suite run and this
    # calibration run (observed ~45% on Claude CoT reruns), and keeps every
    # paired comparison locked to the same physical model/cache state.
    results = []

    target_variant_name = VARIANT_CLI_TO_NAME.get(variant_filter) if variant_filter else None
    for name, template, max_tokens in PROMPT_VARIANTS:
        if name != "Baseline" and variant_filter == "swap":
            continue
        if name != "Baseline" and target_variant_name is not None and name != target_variant_name:
            continue
        print(f"Variant: {name}")
        call_fn = make_prompt_fn(template, max_tokens)
        r = evaluate_variant(name, sample, call_fn)
        results.append(r)
        print(f"  Agreement: {r['agree_pct']:.1%} [{r['agree_ci_lo']:.1%}–{r['agree_ci_hi']:.1%}]  "
              f"Tie sensitivity: {r['tie_sensitivity']:.1%}  "
              f"Tie rate: {r['tie_rate']:.1%}  "
              f"No-answer: {r['no_answer']}")

    # --- Run swap-consistency (Strategy B) separately ---
    if variant_filter is None or variant_filter == "swap":
        print("Variant: Swap-consistency (permutation-based self-consistency)")
        swap_result = evaluate_variant("Swap-consistency", sample, swap_fn)
        print(f"  Agreement: {swap_result['agree_pct']:.1%}  "
              f"Tie sensitivity: {swap_result['tie_sensitivity']:.1%}  "
              f"Tie rate: {swap_result['tie_rate']:.1%}  "
              f"No-answer: {swap_result['no_answer']}")
    else:
        swap_result = None

    # --- Statistics ---
    print("STATISTICAL ANALYSIS")

    # Fresh Baseline is always results[0] (Baseline is first in PROMPT_VARIANTS
    # and its filter-skip conditions never fire on it). Every paired test below
    # uses this as the reference.
    baseline_fresh = results[0]
    assert baseline_fresh["variant"] == "Baseline", \
        f"Expected fresh Baseline at results[0], got {baseline_fresh['variant']!r}"
    baseline_agreed = baseline_fresh["_agreed"]
    baseline_is_tie = baseline_fresh["_is_tie"]

    # Power / MDE
    p_baseline_agree = baseline_fresh["agree_pct"]
    p_baseline_tie_sens = baseline_fresh["tie_sensitivity"]
    mde_agree = mde_mcnemar(len(sample), p_baseline_agree)
    mde_tie = mde_mcnemar(tie_mask.sum(), p_baseline_tie_sens)
    print("\nPower (α=0.05, 80%):")
    print(f"  Agreement outcome (n={len(sample)}):       MDE = {mde_agree:.0%} lift from {p_baseline_agree:.0%} baseline")
    print(f"  Tie sensitivity (n={tie_mask.sum()} tie rows): MDE = {mde_tie:.0%} lift from {p_baseline_tie_sens:.0%} baseline")
    print("  → Agreement outcome is powered; tie sensitivity is screening only (large effects only)")

    # Cochran's Q — prompt variants only (skip if only swap was run)
    q_agree, p_agree, q_tie, p_tie = None, None, None, None
    mcnemar_rows, bh_agree, bh_tie, holm_agree, holm_tie = [], [], [], [], []
    p_swap_agree, p_mid_swap_agree, p_swap_tie, p_mid_swap_tie = None, None, None, None

    if len(results) > 1:
        # Fresh Baseline is results[0]; non-baseline prompt variants follow.
        agree_matrix = np.column_stack([r["_agreed"] for r in results])
        tie_matrix = np.column_stack([r["_is_tie"][tie_mask] for r in results])
        q_agree, p_agree = cochrans_q(agree_matrix)
        q_tie, p_tie = cochrans_q(tie_matrix)
        print(f"\nCochran's Q (prompt variants only — baseline + {len(results)-1} prompts):")
        print(f"  Agreement outcome:    Q={q_agree:.2f}, p={p_agree:.4f}  {'(significant)' if p_agree < 0.05 else '(n.s.)'}")
        print(f"  Tie sensitivity:      Q={q_tie:.2f}, p={p_tie:.4f}  {'(significant)' if p_tie < 0.05 else '(n.s.)'}")

        agree_pvals, tie_pvals = [], []
        for r in results[1:]:
            _, p_exact_agree, p_mid_agree = exact_mcnemar_with_midp(baseline_agreed, r["_agreed"])
            _, p_exact_tie, p_mid_tie = exact_mcnemar_with_midp(baseline_is_tie[tie_mask], r["_is_tie"][tie_mask])
            agree_pvals.append(p_exact_agree)
            tie_pvals.append(p_exact_tie)
            mcnemar_rows.append({
                "variant": r["variant"],
                "agree_p_exact": p_exact_agree,
                "agree_p_midp": p_mid_agree,
                "tie_sens_p_exact": p_exact_tie,
                "tie_sens_p_midp": p_mid_tie,
            })
        bh_agree = benjamini_hochberg(agree_pvals)
        bh_tie = benjamini_hochberg(tie_pvals)
        holm_agree = holm_bonferroni(agree_pvals)
        holm_tie = holm_bonferroni(tie_pvals)

        print("\nPairwise McNemar's vs baseline (exact binomial, BH q=0.10):")
        print(f"  {'Variant':<22} {'Agree p':<10} {'BH adj':<10} {'Holm adj':<10} {'Tie-sens p':<12} {'BH adj':<10}")
        for i, row in enumerate(mcnemar_rows):
            marker = " ← PRIMARY" if row["variant"] == "Explicit tie" else ""
            print(f"  {row['variant']:<22} {row['agree_p_exact']:<10.4f} {bh_agree[i]:<10.4f} {holm_agree[i]:<10.4f} "
                  f"{row['tie_sens_p_exact']:<12.4f} {bh_tie[i]:<10.4f}{marker}")

    if swap_result is not None:
        _, p_swap_agree, p_mid_swap_agree = exact_mcnemar_with_midp(baseline_agreed, swap_result["_agreed"])
        _, p_swap_tie, p_mid_swap_tie = exact_mcnemar_with_midp(baseline_is_tie[tie_mask], swap_result["_is_tie"][tie_mask])
        print("\nSwap-consistency (analyzed separately — structural rule, not prompt variant):")
        print(f"  Agreement:       p={p_swap_agree:.4f} (mid-p={p_mid_swap_agree:.4f})")
        print(f"  Tie sensitivity: p={p_swap_tie:.4f} (mid-p={p_mid_swap_tie:.4f})")
        print(f"  Tie rate: {swap_result['tie_rate']:.1%} (expected ≈ flip rate)")

    # --- Summary table ---
    print("RESULTS SUMMARY")
    all_results = results + ([swap_result] if swap_result is not None else [])
    print(f"  {'Variant':<26} {'Agree%':>7} {'95% CI':>14} {'Tie sens%':>10} {'Tie rate%':>10} {'No-ans':>7}")
    for r in all_results:
        ci = f"[{r.get('agree_ci_lo', 0):.0%}–{r.get('agree_ci_hi', 0):.0%}]" if 'agree_ci_lo' in r else "n/a"
        sep = " ←" if r["variant"] == "Explicit tie" else ""
        tie_s = f"{r['tie_sensitivity']:.1%}" if not np.isnan(r['tie_sensitivity']) else "n/a"
        print(f"  {r['variant']:<26} {r['agree_pct']:>6.1%} {ci:>14} {tie_s:>10} {r['tie_rate']:>9.1%} {r['no_answer']:>7}{sep}")

    # --- Save results ---
    # Mirror run_bias_suite.py convention: ollama judge writes to results/llama3b/
    # so all artifacts for the local Llama 3.2 3B model live under one directory.
    _dir_name = {"ollama": "llama3b"}.get(judge, judge)
    out_dir = f"results/{_dir_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Verdicts CSV — one row per (sample_row, variant). Fresh Baseline is
    # already in `results`, so its rows come through `_verdict_rows` like any
    # other variant.
    all_verdict_rows = []
    for r in results + ([swap_result] if swap_result is not None else []):
        if "_verdict_rows" in r:
            all_verdict_rows.extend(r["_verdict_rows"])
    verdicts_path = f"{out_dir}/calibration_{judge}_{version}_{run_label}_{ts}_verdicts.csv"
    pd.DataFrame(all_verdict_rows).to_csv(verdicts_path, index=False)

    # 2. Summary CSV — one row per variant, full metrics + stats.
    # Fresh Baseline is included (row 0); it has None in McNemar/BH columns
    # because it is the paired reference, not a variant under test.
    summary_rows = []
    # Build a lookup from variant name to mcnemar/BH stats (safe when variant_filter skips some)
    mcnemar_lookup = {m["variant"]: (m, bh_agree[i], bh_tie[i], holm_agree[i], holm_tie[i])
                      for i, m in enumerate(mcnemar_rows)}
    for r in results:
        m = mcnemar_lookup.get(r["variant"])
        summary_rows.append({
            "variant": r["variant"],
            "type": "prompt",
            "is_primary": r["variant"] == "Explicit tie",
            "n": r["n"],
            "no_answer": r["no_answer"],
            "agree_pct": r["agree_pct"],
            "agree_ci_lo": r.get("agree_ci_lo"),
            "agree_ci_hi": r.get("agree_ci_hi"),
            "tie_rate": r["tie_rate"],
            "tie_sensitivity": r["tie_sensitivity"],
            "tie_sens_ci_lo": r.get("tie_sens_ci_lo"),
            "tie_sens_ci_hi": r.get("tie_sens_ci_hi"),
            "tie_specificity": r["tie_specificity"],
            "agree_p_exact": m[0]["agree_p_exact"] if m else None,
            "agree_p_midp": m[0]["agree_p_midp"] if m else None,
            "agree_bh_adj": m[1] if m else None,
            "agree_holm_adj": m[3] if m else None,
            "tie_sens_p_exact": m[0]["tie_sens_p_exact"] if m else None,
            "tie_sens_p_midp": m[0]["tie_sens_p_midp"] if m else None,
            "tie_sens_bh_adj": m[2] if m else None,
            "tie_sens_holm_adj": m[4] if m else None,
        })
    if swap_result is not None:
        summary_rows.append({
            "variant": "Swap-consistency",
            "type": "structural",
            "is_primary": False,
            "n": swap_result["n"],
            "no_answer": swap_result["no_answer"],
            "agree_pct": swap_result["agree_pct"],
            "agree_ci_lo": swap_result.get("agree_ci_lo"),
            "agree_ci_hi": swap_result.get("agree_ci_hi"),
            "tie_rate": swap_result["tie_rate"],
            "tie_sensitivity": swap_result["tie_sensitivity"],
            "tie_sens_ci_lo": swap_result.get("tie_sens_ci_lo"),
            "tie_sens_ci_hi": swap_result.get("tie_sens_ci_hi"),
            "tie_specificity": swap_result["tie_specificity"],
            "agree_p_exact": p_swap_agree,
            "agree_p_midp": p_mid_swap_agree,
            "agree_bh_adj": None,
            "agree_holm_adj": None,
            "tie_sens_p_exact": p_swap_tie,
            "tie_sens_p_midp": p_mid_swap_tie,
            "tie_sens_bh_adj": None,
            "tie_sens_holm_adj": None,
        })
    summary_path = f"{out_dir}/calibration_{judge}_{version}_{run_label}_{ts}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # 3. Stats CSV — Cochran's Q, MDE, sample info
    stats_rows = [
        {"stat": "judge", "value": judge},
        {"stat": "n_sample", "value": len(sample)},
        {"stat": "n_tie_rows", "value": int(tie_mask.sum())},
        {"stat": "n_nontie_rows", "value": int(np.logical_not(tie_mask).sum())},
        {"stat": "mde_agree", "value": mde_agree},
        {"stat": "mde_tie_sensitivity", "value": mde_tie},
        {"stat": "cochrans_q_agree", "value": q_agree},
        {"stat": "cochrans_q_agree_p", "value": p_agree},
        {"stat": "cochrans_q_tie_sens", "value": q_tie},
        {"stat": "cochrans_q_tie_sens_p", "value": p_tie},
        {"stat": "swap_agree_p_exact", "value": p_swap_agree},
        {"stat": "swap_agree_p_midp", "value": p_mid_swap_agree},
        {"stat": "swap_tie_sens_p_exact", "value": p_swap_tie},
        {"stat": "swap_tie_sens_p_midp", "value": p_mid_swap_tie},
    ]
    stats_path = f"{out_dir}/calibration_{judge}_{version}_{run_label}_{ts}_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    print("\nSaved:")
    print(f"  {verdicts_path}")
    print(f"  {summary_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["claude", "ollama", "llama70b", "qwen32b", "gpt4o"], required=True,
                        help="Which judge to run: claude, ollama, llama70b, qwen32b, or gpt4o")
    parser.add_argument("--full", action="store_true",
                        help="Run on all 1689 turn=1 rows instead of a stratified sample")
    parser.add_argument("--n", type=int, default=400,
                        help="Stratified sample size (default: 400; ignored with --full)")
    parser.add_argument("--variant", choices=["swap", "baseline", "explicit-tie", "chain-of-thought", "cot-explicit-tie", "rubric-based"], default=None,
                        help="Run only one variant (default: run all). Use 'swap' for swap-consistency.")
    args = parser.parse_args()
    main(args.judge, full=args.full, variant_filter=args.variant, n=args.n)
