"""
Run the MT-Bench evaluation suite for any judge.

Four core tests judge MT-Bench directly (verbosity, positional, human agreement,
family-preference). Two further tests run only when their pre-generated data exists,
and skip otherwise: self-preference (needs generate_model_outputs.py) and the Zheng
verbosity attack (needs a spot-checked zheng_verbosity_padded.csv).

Usage:
    python src/run_bias_suite.py --judge ollama
    python src/run_bias_suite.py --judge claude
    python src/run_bias_suite.py --judge ollama --out-dir results/my_custom_dir

    # Resume from a specific test after a crash (skips earlier tests):
    python src/run_bias_suite.py --judge llama70b --skip-to positional
    python src/run_bias_suite.py --judge llama70b --skip-to human_agreement
    python src/run_bias_suite.py --judge llama70b --skip-to family_preference

To add a new judge:
    1. Create src/judge_<name>.py with a judge_pair_<name>(question, a, b) -> str function
    2. Add an entry to get_judge_fn() below
    3. Run: python src/run_bias_suite.py --judge <name>

Judge function signature:
    def judge_pair_<name>(question: str, answer_a: str, answer_b: str) -> str
    Returns: "A", "B", or "tie"

Family-preference test requires specifying which same-family MT-Bench model the
judge might favor:
    --target-model  model name in mt_bench.csv (default: "claude-v1" for claude,
                    "llama-13b" for ollama/llama70b, see DEFAULT_TARGET_MODEL).
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


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
        raise ValueError(
            f"Unknown judge: '{judge}'. Add it to get_judge_fn() in run_bias_suite.py."
        )


DEFAULT_TARGET_MODEL = {
    "claude": "claude-v1",
    "ollama": "llama-13b",
    "llama70b": "llama-13b",
    "gpt4o": "gpt-4",
    # qwen32b has no Qwen model in MT-Bench — family-preference skipped.
    # Self-preference (generate_model_outputs.py) handles all five judges separately.
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the MT-Bench evaluation suite for a given judge."
    )
    parser.add_argument(
        "--judge", required=True, help="Judge name: claude, ollama, or custom"
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default: results/<judge>)"
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Same-family MT-Bench model to use for family-preference test (default: claude-v1 for claude, llama-13b for ollama/llama70b)",
    )
    parser.add_argument(
        "--skip-to",
        default=None,
        choices=["positional", "human_agreement", "family_preference", "self_preference", "zheng"],
        help="Skip earlier tests and resume from this one (useful after a crash)",
    )
    parser.add_argument(
        "--dataset",
        default="mt_bench",
        choices=["mt_bench", "judgeBench"],
        help="Dataset to run tests on (default: mt_bench). judgeBench runs positional + human agreement only.",
    )
    args = parser.parse_args()

    _dir_name = {"ollama": "llama3b"}.get(args.judge, args.judge)
    if args.out_dir:
        out_dir = args.out_dir
    elif args.dataset == "judgeBench":
        out_dir = f"results/{_dir_name}/judgebench"
    else:
        out_dir = f"results/{_dir_name}"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_model = args.target_model or DEFAULT_TARGET_MODEL.get(args.judge)

    judge_fn = get_judge_fn(args.judge)

    print("Loading data...")
    if args.dataset == "judgeBench":
        data_path = "data/judgebench.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"{data_path} not found. Run: python src/load_judgeBench.py"
            )
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} JudgeBench rows ({df['question_id'].nunique()} unique pairs)")
    else:
        df = pd.read_csv("data/mt_bench.csv")
        print(
            f"Loaded {len(df)} rows ({df[df['turn']==1]['question_id'].nunique()} unique questions, turn=1)"
        )

    skip_state = {"until": args.skip_to}
    def should_run(test_name):
        if skip_state["until"] is None:
            return True
        if test_name == skip_state["until"]:
            skip_state["until"] = None
            return True
        return False

    print("1/6  Verbosity bias")
    if args.dataset == "judgeBench":
        print("  Skipped — verbosity requires model FEs not available in JudgeBench")
    elif not should_run("verbosity"):
        print("  Skipped (--skip-to)")
    else:
        from verbosity_bias import test_verbosity_bias
        vb = test_verbosity_bias(df, judge_fn=judge_fn)
        vb_path = f"{out_dir}/verbosity_bias_{ts}.csv"
        vb.to_csv(vb_path, index=False)
        # Verbosity drops ties / no-answer / equal-length — those rows can't
        # answer "when the judge picks a side, does it pick the longer one?"
        scorable = vb[vb["picked_longer"].notna()]
        n_dropped = len(vb) - len(scorable)
        pct = 100 * scorable["picked_longer"].sum() / len(scorable)
        print(f"\n  {pct:.0f}% picked longer ({len(scorable)} scorable rows, {n_dropped} dropped: tie/no-answer/equal)")
        print(f"  Saved to {vb_path}")

    print("2/6  Positional bias")
    if not should_run("positional"):
        print("  Skipped (--skip-to)")
    else:
        from positional_bias import test_positional_bias
        pb = test_positional_bias(df, judge_fn=judge_fn)
        pb_path = f"{out_dir}/positional_bias_{ts}.csv"
        pb.to_csv(pb_path, index=False)
        # Report S1 and S2 side-by-side (Zheng et al. 2023 framing).
        n_total = len(pb)
        s1 = 100 * pb["position_changed_verdict"].sum() / n_total
        clean = pb["verdict_original"].isin(["A", "B"]) & pb["verdict_flipped"].isin(["A", "B"])
        n_clean = int(clean.sum())
        s2 = 100 * pb.loc[clean, "position_changed_verdict"].sum() / n_clean if n_clean else 0
        print(f"\n  S1 flip rate (all pairs):       {s1:.0f}% ({n_total} rows)")
        print(f"  S2 flip rate (committed pairs): {s2:.0f}% ({n_clean} rows)")
        print(f"  Saved to {pb_path}")

    print("3/6  Human agreement")
    if not should_run("human_agreement"):
        print("  Skipped (--skip-to)")
    else:
        from human_agreement import test_human_agreement
        ha = test_human_agreement(df, judge_fn=judge_fn)
        ha_path = f"{out_dir}/human_agreement_{ts}.csv"
        ha.to_csv(ha_path, index=False)
        agree_rate = 100 * ha["agreed"].mean()
        print(f"\n  {agree_rate:.0f}% agreement with humans ({len(ha)} rows)")
        print(f"  Saved to {ha_path}")

    print("4/6  Family-preference bias")
    if args.dataset == "judgeBench":
        print("  Skipped — family-preference uses MT-Bench model identity, not applicable to JudgeBench")
    elif target_model is None or not should_run("family_preference"):
        print("  Skipped — no --target-model specified and no default for this judge.")
    else:
        from family_preference import test_family_preference
        fp = test_family_preference(df, target_model=target_model, judge_fn=judge_fn)
        fp_path = f"{out_dir}/family_preference_{ts}.csv"
        fp.to_csv(fp_path, index=False)
        col = target_model.replace("-", "_")
        judge_rate = 100 * fp[f"judge_picked_{col}"].mean()
        human_rate = 100 * fp[f"human_picked_{col}"].mean()
        diff = judge_rate - human_rate
        print(f"\n  Judge picked {target_model}: {judge_rate:.0f}%")
        print(f"  Humans picked {target_model}: {human_rate:.0f}%")
        print(f"  Difference: {diff:+.0f}pp")
        print(f"  Saved to {fp_path}")

    print("5/6  Self-preference")
    if args.dataset == "judgeBench":
        print("  Skipped — self-preference uses fresh MT-Bench outputs, not applicable to JudgeBench")
    elif not should_run("self_preference"):
        print("  Skipped (--skip-to)")
    else:
        from self_preference import ALL_MODELS, JUDGE_OWN_MODEL, test_self_preference
        fresh_dir = "data/fresh_outputs"
        missing = [m for m in ALL_MODELS if not os.path.exists(f"{fresh_dir}/fresh_outputs_{m}.csv")]
        if args.judge not in JUDGE_OWN_MODEL:
            print(f"  Skipped — no own-model mapping for judge '{args.judge}'")
        elif missing:
            print(f"  Skipped — missing fresh outputs: {', '.join(missing)}. "
                  "Run: python src/generate_model_outputs.py")
        else:
            outputs = {m: pd.read_csv(f"{fresh_dir}/fresh_outputs_{m}.csv") for m in ALL_MODELS}
            test_self_preference(outputs, args.judge, judge_fn, out_dir, ts)

    print("6/6  Verbosity attack (Zheng replication)")
    if args.dataset == "judgeBench":
        print("  Skipped — Zheng padding uses MT-Bench responses, not applicable to JudgeBench")
    elif not should_run("zheng"):
        print("  Skipped (--skip-to)")
    else:
        from zheng_verbosity_bias import PADDED_CSV, run_judge as run_zheng_judge
        if not os.path.exists(PADDED_CSV):
            print("  Skipped — no padded data. "
                  "Run: python src/zheng_verbosity_bias.py --step generate (then spot-check it)")
        elif int(pd.read_csv(PADDED_CSV)["spot_checked"].sum()) == 0:
            print(f"  Skipped — padding not spot-checked. Verify {PADDED_CSV}, "
                  "set spot_checked=True on reviewed rows, then re-run.")
        else:
            run_zheng_judge(args.judge)

    print("Done.")
    if args.dataset == "mt_bench":
        print(f"  Run mixed_effects.py next: python src/mixed_effects.py --results-dir {out_dir}")


if __name__ == "__main__":
    main()
