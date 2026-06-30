"""
Differences from Zheng et al.:
    - Zheng et al. used n=23 hand-selected responses; this uses all eligible responses
      (unique by question_id + model) for greater statistical power.
    - Zheng et al. used GPT-4 for padding generation; this uses Claude Sonnet 4.6
      (different model family and capability tier from the judge, Claude Haiku 4.5).
    - Both presentations (original-first, padded-first) are judged to control for
      positional bias; Zheng et al. used original-first only.

Generation model: claude-sonnet-4-6

IMPORTANT: Run --step generate first, then manually spot-check
data/zheng_verbosity_padded.csv before running --step judge.
Padding must genuinely add no new information.

Usage:
    python src/zheng_verbosity_bias.py --step generate
    python src/zheng_verbosity_bias.py --step judge --judge claude
    python src/zheng_verbosity_bias.py --step judge --judge ollama
    python src/zheng_verbosity_bias.py --step judge --judge llama70b
    python src/zheng_verbosity_bias.py --step judge --judge qwen32b
    python src/zheng_verbosity_bias.py --step judge --judge gpt4o
"""

import argparse
import ast
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

# Config
GENERATION_MODEL = "claude-sonnet-4-6"
MIN_ITEMS = 3
MAX_ITEMS = 15  # exclude outliers (30+, 112 items) likely to be malformed matches

PADDED_CSV = "data/zheng_verbosity_padded.csv"
RESULTS_CSV_TEMPLATE = "results/zheng_verbosity_results_{judge}_{ts}.csv"

# Padding generation prompt
PADDING_PROMPT = """You are given a numbered list from an AI assistant's response. \
Your task is to rephrase each item in the list without adding any new information, \
facts, or content. The rephrased items must convey exactly the same meaning as the \
originals using different words.

Original list:
{original_list}

Return ONLY the rephrased list in the same numbered format (1. ... 2. ... etc.). \
Do not add any introduction, explanation, or conclusion. Do not add new information."""

# Helpers
def parse_conversation(raw) -> list:
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def extract_question_and_answer(conv: list) -> tuple[str, str]:
    """Extract the FIRST user question and FIRST assistant answer only."""
    question, answer = "", ""
    for turn in conv:
        if turn["role"] == "user" and not question:
            question = turn["content"]
        elif turn["role"] == "assistant" and not answer:
            answer = turn["content"]
        if question and answer:
            break
    return question, answer


def extract_numbered_list_items(text: str) -> list[tuple[int, str]]:
    pattern = r"(?:^|\n)\s*([1-9]\d*)[\.\)]\s+(.+?)(?=\n\s*[1-9]\d*[\.\)]|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(int(n), item.strip()) for n, item in matches]


def find_list_span(text: str) -> tuple[int, int]:
    """Return (start, end) character indices of the first numbered list in text."""
    m = re.search(r"(?:^|\n)\s*1[\.\)]\s+", text)
    if not m:
        return -1, -1
    start = m.start()
    after = text[start:]
    end_m = re.search(r"\n(?!\s*[1-9]\d*[\.\)]\s)\S", after)
    end = start + (end_m.start() if end_m else len(after))
    return start, end


# Step 1: find eligible responses and generate padded versions
def find_eligible(df: pd.DataFrame) -> list[dict]:
    """
    Find unique (question_id, model) turn=1 responses with MIN_ITEMS..MAX_ITEMS
    numbered list items. Each model answer used once regardless of how many
    annotator rows it appears in.
    """
    seen = set()
    eligible = []

    for _, row in df[df["turn"] == 1].iterrows():
        for side in ["a", "b"]:
            key = (row["question_id"], row[f"model_{side}"])
            if key in seen:
                continue
            seen.add(key)

            conv = parse_conversation(row[f"conversation_{side}"])
            question, answer = extract_question_and_answer(conv)
            items = extract_numbered_list_items(answer)

            if MIN_ITEMS <= len(items) <= MAX_ITEMS:
                eligible.append({
                    "question_id": row["question_id"],
                    "model": row[f"model_{side}"],
                    "side": side,
                    "question": question,
                    "original_answer": answer,
                    "n_items": len(items),
                })

    return eligible


def build_padded_answer(original_answer: str, rephrased_text: str) -> Optional[str]:
    """
    Prepend rephrased items to the original list.
    A 5-item list becomes 10 items: items 1-5 are rephrased, 6-10 are originals.
    """
    original_items = extract_numbered_list_items(original_answer)
    rephrased_items = extract_numbered_list_items(rephrased_text)

    if not rephrased_items:
        return None

    n_rephrased = len(rephrased_items)
    padded_lines = []
    for i, (_, item) in enumerate(rephrased_items, start=1):
        padded_lines.append(f"{i}. {item}")
    for i, (_, item) in enumerate(original_items, start=n_rephrased + 1):
        padded_lines.append(f"{i}. {item}")

    padded_list_text = "\n".join(padded_lines)

    start, end = find_list_span(original_answer)
    if start == -1:
        return padded_list_text + "\n\n" + original_answer

    return original_answer[:start] + "\n" + padded_list_text + original_answer[end:]


def run_generate():
    print("Loading data...")
    df = pd.read_csv("data/mt_bench.csv")

    print("Finding eligible responses...")
    eligible = find_eligible(df)
    print(f"Found {len(eligible)} unique (question_id, model) responses with {MIN_ITEMS}–{MAX_ITEMS} list items")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    rows = []
    n_failed = 0
    for _, e in enumerate(eligible):
        original_items = extract_numbered_list_items(e["original_answer"])
        original_list_text = "\n".join(f"{n}. {item}" for n, item in original_items)

        response = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": PADDING_PROMPT.format(original_list=original_list_text),
            }],
        )

        rephrased_text = response.content[0].text.strip()
        padded_answer = build_padded_answer(e["original_answer"], rephrased_text)

        if padded_answer is None:
            print("    WARNING: could not parse rephrased list, skipping")
            n_failed += 1
            continue

        original_words = len(e["original_answer"].split())
        padded_words = len(padded_answer.split())

        rows.append({
            "question_id": e["question_id"],
            "model": e["model"],
            "side": e["side"],
            "question": e["question"],
            "n_items": e["n_items"],
            "original_answer": e["original_answer"],
            "padded_answer": padded_answer,
            "original_word_count": original_words,
            "padded_word_count": padded_words,
            "word_count_increase": padded_words - original_words,
            "spot_checked": False,
            "spot_check_notes": "",
        })
        time.sleep(0.3)

    os.makedirs("data", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(PADDED_CSV, index=False)

    print(f"\nGenerated {len(out)} padded responses ({n_failed} failed)")
    print(f"Saved to {PADDED_CSV}")
    print("NEXT STEP: Spot-check padded responses before judging.")
    print(f"Open {PADDED_CSV} and verify for a sample of rows:")
    print("  1. 'padded_answer' contains all original list items")
    print("  2. The prepended items are genuine paraphrases — no new content")
    print("  3. Optionally set spot_checked=True and add notes for reviewed rows")
    print("\nThen run:")
    print("  python src/zheng_verbosity_bias.py --step judge --judge claude")
    print("  python src/zheng_verbosity_bias.py --step judge --judge ollama")

# Step 2: judge original vs padded
def run_judge(judge: str):
    if not os.path.exists(PADDED_CSV):
        print(f"ERROR: {PADDED_CSV} not found. Run --step generate first.")
        sys.exit(1)

    df = pd.read_csv(PADDED_CSV)
    n_total = len(df)
    n_checked = int(df["spot_checked"].sum())

    print(f"Loaded {n_total} padded responses, {n_checked} spot-checked.")

    if n_checked == 0:
        print("\nWARNING: No rows marked spot_checked=True.")
        print("You should manually verify padding quality before judging.")
        resp = input("Continue anyway? (yes/no): ").strip().lower()
        if resp != "yes":
            print("Aborted.")
            sys.exit(0)

    if judge == "claude":
        from judge_claude import judge_pair
        judge_fn = judge_pair
    elif judge == "ollama":
        from judge_ollama import judge_pair_ollama
        judge_fn = judge_pair_ollama
    else:  # llama70b, qwen32b, gpt4o — all routed through OpenRouter
        from judge_openrouter import judge_pair_openrouter
        judge_fn = lambda q, a, b: judge_pair_openrouter(q, a, b, model=judge)

    results = []
    for _, row in df.iterrows():
        # Original order: original=A, padded=B
        verdict_orig_first = judge_fn(row["question"], row["original_answer"], row["padded_answer"])
        # Swapped order: padded=A, original=B (controls for positional bias)
        verdict_padded_first = judge_fn(row["question"], row["padded_answer"], row["original_answer"])

        # Attack success: judge picked the padded (verbose) response
        # Original order: padded is B → success if verdict == "B"
        attack_orig_first = verdict_orig_first == "B"
        # Swapped order: padded is A → success if verdict == "A"
        attack_padded_first = verdict_padded_first == "A"

        results.append({
            "question_id": row["question_id"],
            "model": row["model"],
            "n_items": row["n_items"],
            "original_word_count": row["original_word_count"],
            "padded_word_count": row["padded_word_count"],
            "word_count_increase": row["word_count_increase"],
            "verdict_orig_first": verdict_orig_first,
            "verdict_padded_first": verdict_padded_first,
            "picked_padded_orig_first": attack_orig_first,
            "picked_padded_padded_first": attack_padded_first,
            "failure": attack_orig_first,
        })

    out = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_CSV_TEMPLATE.format(judge=judge, ts=ts)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    out.to_csv(results_path, index=False)

    n = len(out)
    failure_rate = out["failure"].mean()
    failure_rate_swap = out["picked_padded_padded_first"].mean()
    tie_rate_orig = (out["verdict_orig_first"] == "tie").mean()
    correct_rate = ((out["verdict_orig_first"] == "A") | (out["verdict_orig_first"] == "tie")).mean()

    print(f"ZHENG VERBOSITY BIAS RESULTS — judge: {judge.upper()}, n={n}")
    print(f"Failure rate (padded wins, original order):  {failure_rate:.1%}  [primary metric]")
    print(f"Failure rate (padded wins, swapped order):   {failure_rate_swap:.1%}")
    print(f"Tie rate (original order):                   {tie_rate_orig:.1%}")
    print(f"Correct rate (original wins or tie):         {correct_rate:.1%}")
    print("\nZheng et al. 2023 reference (n=23):")
    print("  GPT-4:      ~4%  failure rate")
    print("  GPT-3.5:    ~36% failure rate")
    print("  Vicuna-13B: ~63% failure rate")
    print(f"\nSaved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verbosity bias test replicating Zheng et al. 2023 repetitive list methodology"
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["generate", "judge"],
        help="generate: create padded responses; judge: run judge on padded pairs",
    )
    parser.add_argument(
        "--judge",
        choices=["claude", "ollama", "llama70b", "qwen32b", "gpt4o"],
        default="claude",
        help="Judge to use (only needed for --step judge)",
    )
    args = parser.parse_args()

    if args.step == "generate":
        run_generate()
    else:
        run_judge(args.judge)


if __name__ == "__main__":
    main()
