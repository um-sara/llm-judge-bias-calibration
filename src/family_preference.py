"""
Family-preference bias test.

Does a judge systematically favor outputs from a model in its own family?
For MT-Bench:
  - Claude (Haiku) vs "claude-v1"     — Anthropic family
  - Ollama (Llama 3.2 3B) vs "llama-13b" — Llama family
  - Llama-70B (Llama 3.1 70B) vs "llama-13b" — Llama family
  - GPT-4o vs "gpt-4" / "gpt-3.5-turbo" — OpenAI GPT family
  - Qwen — no MT-Bench counterpart, skipped

This is distinct from the stricter "self-preference" test (see self_preference.py)
which compares the judge's own fresh outputs against opponents; this test asks whether
the judge favors any model sharing its family branding, following Zheng et al. 2023's
family-level grouping.
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


def test_family_preference(df: pd.DataFrame, target_model: str = "claude-v1", judge_fn=judge_pair) -> pd.DataFrame:
    """Test whether the judge systematically favors target_model (a same-family model)
    compared to the human rate.

    A verdict of tie or no-answer is coded as "not a family pick" — if the output
    isn't clearly the family member, it isn't family preference.

    Args:
        df: MT-Bench dataframe
        target_model: model name to check preference for (e.g. "claude-v1", "llama-13b")
        judge_fn: callable(question, answer_a, answer_b) -> verdict
    """
    df = df[df["turn"] == 1]
    family_rows = df[(df["model_a"] == target_model) | (df["model_b"] == target_model)].copy()
    print(f"Found {len(family_rows)} rows involving {target_model}")

    results = []

    for i, row in family_rows.iterrows():
        conv_a = parse_conversation(row["conversation_a"])
        conv_b = parse_conversation(row["conversation_b"])

        question, answer_a = extract_question_and_answer(conv_a)
        _, answer_b = extract_question_and_answer(conv_b)

        verdict = judge_fn(question, answer_a, answer_b)

        # Did the judge pick target_model?
        if row["model_a"] == target_model:
            judge_picked_family = verdict == "A"
            human_picked_family = row["winner"] == "model_a"
        else:
            judge_picked_family = verdict == "B"
            human_picked_family = row["winner"] == "model_b"

        col = target_model.replace("-", "_")
        results.append({
            "question_id": row["question_id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "human_winner": row["winner"],
            "verdict": verdict,
            f"judge_picked_{col}": judge_picked_family,
            f"human_picked_{col}": human_picked_family,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from judge_ollama import judge_pair_ollama

    DEFAULT_TARGET = {"claude": "claude-v1", "ollama": "llama-13b"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["claude", "ollama"], required=True)
    parser.add_argument("--target-model", default=None, help="Family-member model to test preference for (default: claude-v1 for claude, llama-13b for ollama)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: results/<judge>)")
    args = parser.parse_args()

    judge_fn = judge_pair if args.judge == "claude" else judge_pair_ollama
    target_model = args.target_model or DEFAULT_TARGET[args.judge]
    out_dir = args.out_dir or f"results/{args.judge}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = pd.read_csv("data/mt_bench.csv")

    print(f"\nRunning family-preference bias test (judge={args.judge}, target={target_model})...")
    results = test_family_preference(df, target_model=target_model, judge_fn=judge_fn)

    col = target_model.replace("-", "_")
    judge_rate = results[f"judge_picked_{col}"].mean() * 100
    human_rate = results[f"human_picked_{col}"].mean() * 100

    print("\n--- Results ---")
    print(f"Rows tested: {len(results)}")
    print(f"Judge picked {target_model}: {judge_rate:.0f}%")
    print(f"Humans picked {target_model}: {human_rate:.0f}%")
    print(f"Difference: {judge_rate - human_rate:+.0f}% (positive = judge favors {target_model} more than humans do)")

    if judge_rate - human_rate > 10:
        print("-> Family-preference bias detected")
    else:
        print("-> No strong family-preference bias")

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/family_preference_{ts}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
