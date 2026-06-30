"""
Positional bias test.

For each row, the judge is called twice:
  - Original order: judge(question, answer_a, answer_b)
  - Flipped order:  judge(question, answer_b, answer_a)

If the judge is unbiased, flipping shouldn't change the outcome.
If it does flip, that's positional bias.
"""

import os
import pandas as pd
import ast
from judge_claude import judge_pair


def extract_question_and_answer(conversation: list) -> tuple[str, str]:
    """Pull the first question and first answer out of a conversation list."""
    question = ""
    answer = ""
    for turn in conversation:
        if turn["role"] == "user" and not question:
            question = turn["content"]
        elif turn["role"] == "assistant" and not answer:
            answer = turn["content"]
        if question and answer:
            break
    return question, answer


def parse_conversation(raw) -> list:
    """The CSV stores conversations as strings — convert back to list."""
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def test_positional_bias(df: pd.DataFrame, n_samples: int = None, judge_fn=judge_pair) -> pd.DataFrame:
    """
    Run positional bias test on n_samples rows (or all rows if n_samples is None).
    Returns a dataframe with original verdict, flipped verdict, and whether it changed.
    """
    # Use only turn=1 (first turn) to keep things simple
    df = df[df["turn"] == 1].copy()
    if n_samples is not None:
        df = df.head(n_samples)

    results = []

    for i, row in df.iterrows():
        conv_a = parse_conversation(row["conversation_a"])
        conv_b = parse_conversation(row["conversation_b"])

        question, answer_a = extract_question_and_answer(conv_a)
        _, answer_b = extract_question_and_answer(conv_b)

        # Original order
        verdict_original = judge_fn(question, answer_a, answer_b)

        # Flipped order — note A and B are swapped
        verdict_flipped_raw = judge_fn(question, answer_b, answer_a)
        # Re-map so the verdict is always in terms of model_a vs model_b
        if verdict_flipped_raw == "A":
            verdict_flipped = "B"
        elif verdict_flipped_raw == "B":
            verdict_flipped = "A"
        else:
            verdict_flipped = verdict_flipped_raw

        flipped = verdict_original != verdict_flipped

        results.append({
            "question_id": row["question_id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "human_winner": row["winner"],
            "verdict_original": verdict_original,
            "verdict_flipped": verdict_flipped,
            "position_changed_verdict": flipped,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from judge_ollama import judge_pair_ollama

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["claude", "ollama"], required=True)
    parser.add_argument("--out-dir", default=None, help="Output directory (default: results/<judge>)")
    args = parser.parse_args()

    judge_fn = judge_pair if args.judge == "claude" else judge_pair_ollama
    out_dir = args.out_dir or f"results/{args.judge}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = pd.read_csv("data/mt_bench.csv")
    print(f"Loaded {len(df)} rows")

    print(f"\nRunning positional bias test on all turn=1 rows (judge={args.judge})...")
    results = test_positional_bias(df, judge_fn=judge_fn)

    # Summary — report both S1 and S2 following Zheng et al. 2023 framing.
    # S1 (ties/no-answer included): among ALL pairs, what fraction had a
    #   differing verdict across orderings. A tie↔tie pair counts as "didn't
    #   flip" here, an A↔tie pair counts as "flipped".
    # S2 (ties/no-answer excluded): restrict to pairs where the judge
    #   committed with A or B in BOTH orderings, then ask "of the pairs
    #   where the judge actually picked a side, did flipping the order
    #   change its mind?" This is the cleaner signal for positional bias.
    # Reasonable people disagree on which to prefer, so report both.
    n_total = len(results)
    n_flipped_s1 = int(results["position_changed_verdict"].sum())
    s1 = 100 * n_flipped_s1 / n_total if n_total > 0 else 0

    clean = results["verdict_original"].isin(["A", "B"]) & results["verdict_flipped"].isin(["A", "B"])
    n_clean = int(clean.sum())
    n_flipped_s2 = int(results.loc[clean, "position_changed_verdict"].sum())
    s2 = 100 * n_flipped_s2 / n_clean if n_clean > 0 else 0

    print("\n--- Results ---")
    print(f"S1 flip rate (ties/no-answer included): {n_flipped_s1}/{n_total} ({s1:.0f}%)")
    print(f"S2 flip rate (committed pairs only):    {n_flipped_s2}/{n_clean} ({s2:.0f}%)")
    print(f"  ({n_total - n_clean} pairs dropped from S2: at least one verdict was tie or no-answer)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/positional_bias_{ts}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
