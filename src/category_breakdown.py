"""
Category breakdown analysis.

Joins MT-Bench question categories onto existing bias result CSVs and
reports per-category summaries for all four bias tests plus the Zheng
verbosity replication.

No API calls — reads result files only.

Usage:
    python src/category_breakdown.py --judge claude
    python src/category_breakdown.py --judge ollama
    python src/category_breakdown.py --zheng          # both judges, Zheng only
"""

import argparse
import glob
import os
import sys
import pandas as pd


# MT-Bench question ID → category (Zheng et al. standard mapping)
CATEGORY_MAP = [
    (81,  91,  "writing"),
    (91,  101, "roleplay"),
    (101, 111, "reasoning"),
    (111, 121, "math"),
    (121, 131, "coding"),
    (131, 141, "extraction"),
    (141, 151, "stem"),
    (151, 161, "humanities"),
]

CATEGORY_ORDER = [c for _, _, c in CATEGORY_MAP]


def get_category(qid: int) -> str:
    for lo, hi, cat in CATEGORY_MAP:
        if lo <= qid < hi:
            return cat
    return "unknown"


def latest(pattern: str):
    """Return the most recently modified file matching pattern, or None."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["question_id"].apply(get_category)
    return df


# Per-bias summaries

def summarize_verbosity(df: pd.DataFrame) -> pd.DataFrame:
    """
    picked_longer: did the judge choose the longer answer?
    Reports rate per category.
    """
    df = add_category(df)
    df["picked_longer"] = df["picked_longer"].map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
    grp = df.groupby("category", observed=True)
    out = pd.DataFrame({
        "n_pairs":        grp["picked_longer"].count(),
        "picked_longer_rate": grp["picked_longer"].mean().round(3),
    }).reindex(CATEGORY_ORDER)
    out.index.name = "category"
    return out


def summarize_positional(df: pd.DataFrame) -> pd.DataFrame:
    """
    position_changed_verdict: did swapping A↔B flip the verdict?
    Reports flip rate per category.
    """
    df = add_category(df)
    grp = df.groupby("category", observed=True)
    out = pd.DataFrame({
        "n_pairs":   grp["position_changed_verdict"].count(),
        "flip_rate": grp["position_changed_verdict"].mean().round(3),
    }).reindex(CATEGORY_ORDER)
    out.index.name = "category"
    return out


def summarize_human_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two metrics per category (following Zheng et al.):
      strict_agreement  — over all rows (includes tie rows)
      s2_agreement      — only rows where both judge AND human gave non-tie verdict
    """
    df = add_category(df)

    results = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        n_total = len(sub)

        strict = sub["agreed"].mean() if n_total > 0 else float("nan")

        # S2: exclude rows where judge_winner or human_winner is a tie
        non_tie = sub[
            (sub["judge_winner"] != "tie") & (sub["human_winner"] != "tie")
        ]
        s2 = non_tie["agreed"].mean() if len(non_tie) > 0 else float("nan")

        results.append({
            "category":        cat,
            "n_rows":          n_total,
            "strict_agreement": round(strict, 3),
            "n_non_tie":       len(non_tie),
            "s2_agreement":    round(s2, 3) if not pd.isna(s2) else float("nan"),
        })

    out = pd.DataFrame(results).set_index("category")
    return out


def summarize_family_preference(df: pd.DataFrame) -> pd.DataFrame:
    """
    judge_picked_<target> vs human_picked_<target>.
    Target column is detected dynamically so this works for any judge/family
    pairing (claude_v1, llama_13b, gpt_4, etc.).
    """
    df = add_category(df)
    judge_col = next(c for c in df.columns if c.startswith("judge_picked_"))
    human_col = next(c for c in df.columns if c.startswith("human_picked_"))
    grp = df.groupby("category", observed=True)
    out = pd.DataFrame({
        "n_pairs":             grp[judge_col].count(),
        "judge_pick_rate":     grp[judge_col].mean().round(3),
        "human_pick_rate":     grp[human_col].mean().round(3),
    }).reindex(CATEGORY_ORDER)
    out["pick_rate_gap"] = (out["judge_pick_rate"] - out["human_pick_rate"]).round(3)
    out.index.name = "category"
    return out


def summarize_zheng(df: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """
    Zheng verbosity replication.
    Primary metric: failure rate = picked_padded_orig_first (padded wins, original order).
    Also reports swapped-order failure rate and tie rate.
    One row per question_id, so category is joined directly.
    """
    df = add_category(df)
    grp = df.groupby("category", observed=True)
    out = pd.DataFrame({
        "judge":              judge_name,
        "n_questions":        grp["failure"].count(),
        "failure_rate":       grp["failure"].mean().round(3),
        "failure_rate_swap":  grp["picked_padded_padded_first"].mean().round(3),
    }).reindex(CATEGORY_ORDER)
    out.index.name = "category"
    return out


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Per-category bias breakdown")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--judge", choices=["claude", "ollama", "llama70b", "gpt4o", "qwen32b"])
    group.add_argument("--zheng", action="store_true",
                       help="Zheng verbosity replication breakdown for both judges")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Zheng-only mode: compare both judges side by side
    if args.zheng:
        all_summaries = []
        judges_found = []
        for judge_name in ("claude", "ollama", "llama70b", "qwen32b", "gpt4o"):
            path = latest(f"results/zheng_verbosity_results_{judge_name}_*.csv")
            if path is None:
                print(f"[zheng/{judge_name}] No result file found — skipping.\n")
                continue
            judges_found.append(judge_name)
            df = pd.read_csv(path)
            summary = summarize_zheng(df, judge_name)
            print(f"=== ZHENG VERBOSITY — {judge_name.upper()} ===")
            print(f"Source: {os.path.basename(path)}")
            print(summary.to_string())
            print()
            summary_out = summary.reset_index()
            summary_out.insert(0, "bias", "zheng_verbosity")
            all_summaries.append(summary_out)

        if not judges_found:
            print("No Zheng verbosity result files found. Run zheng_verbosity_bias.py first.")
            sys.exit(1)

        # Cross-judge artifacts live under results/cross_judge/. Filename embeds
        # which judges were actually present so the file is self-describing —
        # if a future run only finds claude data, the filename reflects that.
        cross_dir = "results/cross_judge"
        os.makedirs(cross_dir, exist_ok=True)
        judges_token = "_".join(judges_found)
        out_path = f"{cross_dir}/category_breakdown_zheng_{judges_token}_{timestamp}.csv"
        pd.concat(all_summaries, ignore_index=True).to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        return

    # Standard per-judge bias breakdown
    _dir_name = {"ollama": "llama3b"}.get(args.judge, args.judge)
    results_dir = f"results/{_dir_name}"

    bias_configs = [
        ("verbosity",        "verbosity_bias_*.csv",    summarize_verbosity),
        ("positional",       "positional_bias_*.csv",   summarize_positional),
        ("human_agreement",  "human_agreement_*.csv",   summarize_human_agreement),
    ]
    # Family-preference applies to any judge with a same-family MT-Bench model:
    # claude→claude-v1, ollama→llama-13b, llama70b→llama-13b, gpt4o→gpt-4.
    # The CSV-presence check below handles judges (e.g. qwen32b) with no target.
    bias_configs.append(
        ("family_preference", "family_preference_*.csv", summarize_family_preference)
    )

    # Also include Zheng if results exist for this judge
    zheng_path = latest(f"results/zheng_verbosity_results_{args.judge}_*.csv")
    if zheng_path:
        bias_configs.append(
            ("zheng_verbosity", zheng_path, lambda df: summarize_zheng(df, args.judge))
        )

    out_path = os.path.join(results_dir, f"category_breakdown_{timestamp}.csv")

    all_summaries = []
    any_found = False
    for bias_name, glob_pat_or_path, summarize_fn in bias_configs:
        # zheng entry already has a resolved path; others need glob resolution
        if os.path.isfile(glob_pat_or_path):
            path = glob_pat_or_path
        else:
            path = latest(os.path.join(results_dir, glob_pat_or_path))
        if path is None:
            print(f"[{bias_name}] No result file found in {results_dir}/ — skipping.\n")
            continue

        any_found = True
        df = pd.read_csv(path)
        summary = summarize_fn(df)

        print(f"=== {bias_name.upper().replace('_', ' ')} ===")
        print(f"Source: {os.path.basename(path)}")
        print(summary.to_string())
        print()

        summary_out = summary.copy().reset_index()
        summary_out.insert(0, "bias", bias_name)
        all_summaries.append(summary_out)

    if not any_found:
        print(f"No result files found in {results_dir}/. Run the bias suite first.")
        sys.exit(1)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
