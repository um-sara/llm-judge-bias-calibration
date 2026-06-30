"""
Verbosity bias test.

Does the judge prefer longer answers regardless of quality?
For each row, answer lengths and the judge's pick are recorded.
If the judge picks the longer answer significantly more than 50% of the time,
that's verbosity bias.
"""

import os
import pandas as pd
import ast
from judge_claude import judge_pair


def parse_conversation(raw) -> list:
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def extract_question_and_answer(conversation: list) -> tuple[str, str]:
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


def word_count(text: str) -> int:
    return len(text.split())


def test_verbosity_bias(df: pd.DataFrame, n_samples: int = None, judge_fn=judge_pair) -> pd.DataFrame:
    df = df[df["turn"] == 1].copy()
    if n_samples is not None:
        df = df.head(n_samples)

    results = []

    for i, row in df.iterrows():
        conv_a = parse_conversation(row["conversation_a"])
        conv_b = parse_conversation(row["conversation_b"])

        question, answer_a = extract_question_and_answer(conv_a)
        _, answer_b = extract_question_and_answer(conv_b)

        len_a = word_count(answer_a)
        len_b = word_count(answer_b)
        longer = "A" if len_a > len_b else "B" if len_b > len_a else "equal"

        verdict = judge_fn(question, answer_a, answer_b)
        # 4-class verdict: A / B / tie / no answer. picked_longer is only
        # defined when both the verdict is a clean A|B pick AND the lengths
        # are unequal. tie and no-answer must become None — coding them as
        # False would silently conflate "picked shorter" with "failed to
        # answer" or "called a tie", contaminating the aggregate.
        if longer == "equal" or verdict not in ("A", "B"):
            picked_longer = None
        else:
            picked_longer = verdict == longer

        results.append({
            "question_id": row["question_id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "human_winner": row["winner"],
            "len_a": len_a,
            "len_b": len_b,
            "longer_answer": longer,
            "verdict": verdict,
            "picked_longer": picked_longer,
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

    print(f"\nRunning verbosity bias test on all turn=1 rows (judge={args.judge})...")
    results = test_verbosity_bias(df, judge_fn=judge_fn)

    # Summary — drop ties / no-answer / equal-length. The question is
    # "when the judge picks a side, does it pick the longer one?" — rows
    # where the judge didn't commit (tie, no answer) or where there is no
    # longer answer (equal length) can't answer that question.
    n_total = len(results)
    n_equal = int((results["longer_answer"] == "equal").sum())
    n_tie = int((results["verdict"] == "tie").sum())
    n_noans = int((results["verdict"] == "no answer").sum())

    scorable = results[results["picked_longer"].notna()]
    n_picked_longer = int(scorable["picked_longer"].sum())
    n_scorable = len(scorable)
    pct = 100 * n_picked_longer / n_scorable if n_scorable > 0 else 0

    print("\nResults")
    print(f"Total rows: {n_total}")
    print(f"  Dropped: equal-length={n_equal}, tie={n_tie}, no-answer={n_noans}")
    print(f"Scorable rows (clean A/B verdict, unequal length): {n_scorable}")
    print(f"Judge picked longer answer: {n_picked_longer}/{n_scorable} ({pct:.0f}%)")
    if pct > 60:
        print("-> Verbosity bias detected (judge prefers longer answers)")
    elif pct < 40:
        print("-> Reverse verbosity bias (judge prefers shorter answers)")
    else:
        print("-> No strong verbosity bias")

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/verbosity_bias_{ts}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
