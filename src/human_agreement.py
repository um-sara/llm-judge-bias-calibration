"""
Human agreement test.

How often does the judge's verdict match the human ground truth label?
This gives context for all the other bias tests — if the judge barely agrees
with humans to begin with, the bias findings matter even more.
"""

import os
import pandas as pd
import ast
from judge_claude import judge_pair
from sklearn.metrics import confusion_matrix, cohen_kappa_score


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


def map_verdict_to_winner(verdict: str) -> str:
    """
    4-label mapping: A/B/tie/no answer → model_a/model_b/tie/no answer.

    Keep "no answer" distinct from "tie" — a parse failure is not a correct
    tie call and must not inflate agreement on human-tie rows.
    """
    if verdict == "A":
        return "model_a"
    elif verdict == "B":
        return "model_b"
    elif verdict == "tie":
        return "tie"
    else:
        return "no answer"


def test_human_agreement(df: pd.DataFrame, n_samples: int = None, judge_fn=judge_pair) -> pd.DataFrame:
    df = df[df["turn"] == 1].copy()
    if n_samples is not None:
        df = df.head(n_samples)

    results = []

    for i, row in df.iterrows():
        conv_a = parse_conversation(row["conversation_a"])
        conv_b = parse_conversation(row["conversation_b"])

        question, answer_a = extract_question_and_answer(conv_a)
        _, answer_b = extract_question_and_answer(conv_b)

        verdict = judge_fn(question, answer_a, answer_b)
        judge_winner = map_verdict_to_winner(verdict)
        agreed = judge_winner == row["winner"]

        results.append({
            "question_id": row["question_id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "human_winner": row["winner"],
            "judge": row["judge"],
            "judge_verdict": verdict,
            "judge_winner": judge_winner,
            "agreed": agreed,
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

    print(f"\nRunning human agreement test on all turn=1 rows (judge={args.judge})...")
    results = test_human_agreement(df, judge_fn=judge_fn)

    overall = results["agreed"].mean() * 100
    print(f"\n--- Results ---")
    print(f"Overall agreement with humans: {overall:.0f}%")

    # Break down by what the human actually picked
    print(f"\nAgreement by human winner:")
    for winner in ["model_a", "model_b", "tie"]:
        subset = results[results["human_winner"] == winner]
        if len(subset) > 0:
            rate = subset["agreed"].mean() * 100
            print(f"  When human picked {winner}: judge agreed {rate:.0f}% ({len(subset)} rows)")

    # Confusion matrix: rows = human label, cols = judge label.
    # 4-label grid so "no answer" is reported as its own column, not silently
    # dropped (sklearn confusion_matrix with labels= excludes values not in
    # the label list).
    labels = ["model_a", "model_b", "tie", "no answer"]
    cm = confusion_matrix(results["human_winner"], results["judge_winner"], labels=labels)

    print(f"\nConfusion matrix (rows=human, cols=judge):")
    print(f"{'':>10}  {'model_a':>9}  {'model_b':>9}  {'tie':>9}  {'no answer':>11}")
    for i, label in enumerate(labels):
        # human_winner has only 3 classes (no "no answer" on human side), so
        # skip printing that row — it is always all zeros and adds noise.
        if label == "no answer":
            continue
        print(f"  {label:>8}  {cm[i][0]:>9}  {cm[i][1]:>9}  {cm[i][2]:>9}  {cm[i][3]:>11}")

    # Cohen's kappa on the full 4-label grid. Parse failures remain in the
    # denominator as disagreements — reflecting the true reliability of the
    # judge, not a filtered best-case view.
    kappa = cohen_kappa_score(results["human_winner"], results["judge_winner"])
    print(f"\nCohen's kappa (4-label, no-answer counted as disagreement): {kappa:.3f}")
    print("  (0=no better than chance, 1=perfect agreement, <0=worse than chance)")

    # Also report the no-answer rate and a filtered kappa for comparison —
    # people reading the paper will want to know how much of the disagreement
    # comes from parse failures vs. genuine label disagreement.
    n_no_ans = int((results["judge_winner"] == "no answer").sum())
    print(f"\nParse failures (judge emitted 'no answer'): {n_no_ans}/{len(results)} ({100*n_no_ans/len(results):.1f}%)")
    if n_no_ans > 0:
        scorable = results[results["judge_winner"] != "no answer"]
        kappa_scorable = cohen_kappa_score(scorable["human_winner"], scorable["judge_winner"])
        print(f"Cohen's kappa (excluding no-answer rows, n={len(scorable)}): {kappa_scorable:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/human_agreement_{ts}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
