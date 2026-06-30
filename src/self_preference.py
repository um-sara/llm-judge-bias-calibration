"""
Self-preference: do judges favor outputs from their own model?

Uses fresh model-generated outputs (data/fresh_outputs/). Each judge is tested against each of the 4 other models on
the same 80 MT-Bench questions * 3 samples = 240 pairs per opponent.

Self-preference is coded position-invariantly: a pair counts as self-preference
only when the judge picks its own model in BOTH orderings (own→A and other→A
after remapping). This eliminates positional bias contamination.

Judge -> own model mapping:
    claude    -> claude_haiku
    ollama    -> llama3b
    llama70b  -> llama70b
    qwen32b   -> qwen32b
    gpt4o     -> gpt4o

This script only collects raw verdicts and writes one CSV per judge.
All inferential statistics (cluster-robust logit, opponent FEs, BH correction)
live in src/mixed_effects.py (run_self_preference), matching how the other
bias tests are structured.

Usage:
    python src/self_preference.py                        # all judges
    python src/self_preference.py --judge claude
    python src/self_preference.py --judge claude llama70b
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


# Judge -> own model mapping

JUDGE_OWN_MODEL = {
    "claude":   "claude_haiku",
    "ollama":   "llama3b",
    "llama70b": "llama70b",
    "qwen32b":  "qwen32b",
    "gpt4o":    "gpt4o",
}

ALL_JUDGES = list(JUDGE_OWN_MODEL.keys())
ALL_MODELS = list(JUDGE_OWN_MODEL.values())


def get_judge_fn(judge: str):
    if judge == "claude":
        from judge_claude import judge_pair
        return judge_pair
    elif judge == "ollama":
        from judge_ollama import judge_pair_ollama
        return judge_pair_ollama
    elif judge in ("llama70b", "qwen32b", "gpt4o"):
        from judge_openrouter import judge_pair_openrouter
        return lambda q, a, b: judge_pair_openrouter(q, a, b, model=judge)
    else:
        raise ValueError(f"Unknown judge: '{judge}'")


def remap_verdict(verdict: str) -> str:
    """Flip A↔B — maps other-first verdict back to own/other space."""
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


# Core data collection

def test_self_preference(
    outputs: dict,
    judge: str,
    judge_fn,
    out_dir: str,
    ts: str,
) -> pd.DataFrame:
    """
    Run self-preference for one judge against all 4 opponents.

    Args:
        outputs:  {model_key: DataFrame(question_id, category, sample, question_text, answer)}
        judge:    judge key (e.g. "claude")
        judge_fn: callable(question, answer_a, answer_b) -> "A" | "B" | "tie" | "no answer"
        out_dir:  directory to save results
        ts:       timestamp string for filenames

    Returns:
        Raw results DataFrame. Inferential statistics are produced downstream
        by src/mixed_effects.py (run_self_preference).
    """
    own_model = JUDGE_OWN_MODEL[judge]
    opponent_models = [m for m in ALL_MODELS if m != own_model]
    own_df = outputs[own_model]

    all_rows = []

    for opponent in opponent_models:
        opp_df = outputs[opponent]

        # Pair same question_id + sample across models
        paired = own_df.merge(
            opp_df[["question_id", "sample", "answer"]],
            on=["question_id", "sample"],
            suffixes=("_own", "_opp"),
        )
        print(f"\n  {judge} vs {opponent}: {len(paired)} pairs")

        for i, (_, row) in enumerate(paired.iterrows()):
            question = row["question_text"]
            own_ans  = row["answer_own"]
            opp_ans  = row["answer_opp"]

            # Ordering 1: own → A, other → B
            v1 = judge_fn(question, own_ans, opp_ans)
            # Ordering 2: other → A, own → B  (remapped back to own/other space)
            v2_raw = judge_fn(question, opp_ans, own_ans)
            v2 = remap_verdict(v2_raw)

            # Position-invariant self-preference: picked own in both orderings
            self_pref  = int(v1 == "A" and v2 == "A")
            other_pref = int(v1 == "B" and v2 == "B")

            all_rows.append({
                "judge":                       judge,
                "own_model":                   own_model,
                "opponent_model":              opponent,
                "question_id":                 row["question_id"],
                "category":                    row["category"],
                "sample":                        row["sample"],
                "verdict_own_first":           v1,
                "verdict_other_first":         v2_raw,
                "verdict_other_first_remapped": v2,
                "self_pref":                   self_pref,
                "other_pref":                  other_pref,
                "inconsistent":                int(not self_pref and not other_pref),
            })

            if (i + 1) % 60 == 0:
                done = len(all_rows)
                print(f"    [{i+1}/{len(paired)}]  ({done} total rows so far)")

    results_df = pd.DataFrame(all_rows)
    out_path = f"{out_dir}/self_preference_{ts}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved raw results to {out_path}")

    # --- Descriptive per-opponent counts (no inference) ---
    print(f"\n  {'Opponent':<14} {'Pairs':>6} {'Self':>6} {'Other':>7} "
          f"{'Inconsist':>11} {'Self%':>7}")
    for opponent in opponent_models:
        sub = results_df[results_df["opponent_model"] == opponent]
        n_total       = len(sub)
        n_self        = int(sub["self_pref"].sum())
        n_other       = int(sub["other_pref"].sum())
        n_inconsist   = int(sub["inconsistent"].sum())
        print(f"  {opponent:<14} {n_total:>6} {n_self:>6} {n_other:>7} "
              f"{n_inconsist:>11} {100*n_self/n_total:>6.1f}%")

    print(f"\n  Run src/mixed_effects.py --results-dir {out_dir} for inference "
          f"(cluster-robust logit + BH correction).")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Self-preference: collect judge verdicts on fresh model outputs."
    )
    parser.add_argument(
        "--judge",
        choices=ALL_JUDGES,
        nargs="+",
        default=ALL_JUDGES,
        help="Judge(s) to run (default: all). Multiple allowed: --judge claude llama70b",
    )
    args = parser.parse_args()

    # Load all fresh outputs once before looping over judges
    print("Loading fresh model outputs...")
    outputs = {}
    for model in ALL_MODELS:
        path = f"data/fresh_outputs/fresh_outputs_{model}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing fresh outputs for {model}: {path}\n"
                f"Run: python src/generate_model_outputs.py --models {model}"
            )
        outputs[model] = pd.read_csv(path)
        print(f"  {model}: {len(outputs[model])} rows")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    _dir_map = {"ollama": "llama3b"}
    for judge in args.judge:
        out_dir = f"results/{_dir_map.get(judge, judge)}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"Judge: {judge.upper()}  (own model: {JUDGE_OWN_MODEL[judge]})")

        judge_fn = get_judge_fn(judge)
        test_self_preference(outputs, judge, judge_fn, out_dir, ts)

    print("Done.")


if __name__ == "__main__":
    main()
