"""
Cluster-robust logistic regression — clustering robustness check.

Uses cluster-robust standard errors (Huber-White sandwich estimator) throughout.
This is the standard frequentist approach for correlated binary outcomes and
produces valid p-values unlike variational Bayes approximations.

Five models:
  1. Verbosity        — length_diff + model FEs, clustered by pair_id
  2. Positional       — long-format picked_position_A ~ position_is_A, clustered by pair_id
  3. Human Agmt       — intercept-only, clustered by question_id
  4. Family Pref      — self_pref ~ C(opponent_model), clustered by question_id
  5. Self-Pref        — position-invariant self-preference, clustered by question_id
"""

import argparse
import glob
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


# Helpers

def response_pair_id(row) -> str:
    """Order-invariant pair key: sorts model names so (A,B) == (B,A)."""
    pair = tuple(sorted([row["model_a"], row["model_b"]]))
    return f"{row['question_id']}_{pair[0]}_{pair[1]}"


def logit_to_prob(log_odds: float) -> float:
    return 1.0 / (1.0 + np.exp(-log_odds))


def prob_ci(log_odds: float, se: float, z: float = 1.96):
    lo = logit_to_prob(log_odds - z * se)
    hi = logit_to_prob(log_odds + z * se)
    return lo, hi


def pvalue_from_z(z_stat: float) -> float:
    return 2 * (1 - stats.norm.cdf(abs(z_stat)))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


# 1. Verbosity bias

def run_verbosity(path: str = "results/verbosity_bias.csv"):
    print("Verbosity Bias")

    df = pd.read_csv(path)
    # Exclude ties, no-answer (parse failures), and equal-length rows.
    # "no answer" must be filtered explicitly — without it, rows where the
    # judge couldn't emit a verdict would be coded as picked_model_a=0,
    # contaminating the outcome with parse failures.
    df = df[
        (df["verdict"] != "tie")
        & (df["verdict"] != "no answer")
        & (df["longer_answer"] != "equal")
    ].copy()
    df["picked_model_a"] = (df["verdict"] == "A").astype(int)
    # Scale to 100-word units for numerical stability
    df["length_diff_100"] = (df["len_a"] - df["len_b"]) / 100.0

    n = len(df)
    q = df["question_id"].nunique()
    print(f"  n={n} scorable rows | clusters: {q} questions")
    print("  Model: picked_model_a ~ length_diff_100 + C(model_a) + C(model_b)")
    print("  (model FEs control for quality confound — better models tend to be longer)")

    # Cluster-robust logit with model identity fixed effects.
    # Rows sharing the same pair (e.g. gpt4 vs llama at Q1) share the same
    # answer texts, so their errors are correlated at the pair level.
    df["pair_id"] = df.apply(response_pair_id, axis=1)
    res = smf.logit(
        "picked_model_a ~ length_diff_100 + C(model_a) + C(model_b)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["pair_id"]}, disp=False)

    beta1_100 = res.params["length_diff_100"]
    p1 = res.pvalues["length_diff_100"]

    or_per_word = np.exp(beta1_100 / 100.0)
    or_per_100 = np.exp(beta1_100)

    sig = "YES" if p1 < 0.05 else "NO"
    p_fmt = "< 0.0001" if p1 < 0.0001 else f"{p1:.4f}"

    # Empirical P(pick longer) with Wilson 95% CI
    n_longer = int(df["picked_longer"].sum())
    p_longer = n_longer / n
    ci_lo, ci_hi = wilson_ci(n_longer, n)

    print(f"  β₁: OR={or_per_word:.4f} per word  |  OR={or_per_100:.3f} per 100 words")
    print(f"  P(pick longer): {p_longer:.0%}  [95% CI: {ci_lo:.0%}–{ci_hi:.0%}]")
    print(f"  p-value: {p_fmt}  →  Significant: {sig}")

    # Question-level robustness check
    q_rates = df.groupby("question_id")["picked_longer"].mean()
    print("\n  Question-level robustness check (descriptive):")
    print(f"    min={q_rates.min():.0%}  median={q_rates.median():.0%}  max={q_rates.max():.0%}")
    print("    (result is not driven by one question cluster)")

    return {
        "test": "Verbosity",
        "n": n,
        "clusters": f"{q} questions",
        "estimate": f"{p_longer:.0%} pick longer",
        "ci_95": f"[{ci_lo:.0%}, {ci_hi:.0%}]",
        "p_value": p_fmt,
        "finding": "Bias detected" if sig == "YES" else "No bias",
        "method": "cluster-robust logit + model FEs",
    }


# 2. Positional bias


def run_positional(path: str = "results/positional_bias.csv"):
    print("Positional Bias")

    df_raw = pd.read_csv(path)
    df_raw["pair_id"] = df_raw.apply(response_pair_id, axis=1)

    # S1 flip rate (ties/no-answer included) — descriptive only. Counts a
    # tie↔tie pair as "didn't flip" and an A↔tie pair as "flipped", which
    # mixes directional flips with refusals. Reported for comparability.
    n_total = len(df_raw)
    n_flipped_s1 = int(df_raw["position_changed_verdict"].sum())
    s1_rate = n_flipped_s1 / n_total if n_total else 0
    s1_ci_lo, s1_ci_hi = wilson_ci(n_flipped_s1, n_total) if n_total else (0, 0)

    # S2: restrict to pairs where both verdicts are clean A|B — "of the
    # pairs where the judge committed, did flipping the order change its
    # mind?" This is the subset the cluster-robust logit is fit on; the
    # long-format coding int(verdict_flipped != "A") is only well-defined
    # here (otherwise tie/no-answer would land asymmetrically in
    # position_is_A=0).
    clear = df_raw["verdict_original"].isin(["A", "B"]) & df_raw["verdict_flipped"].isin(["A", "B"])
    n_dropped = int((~clear).sum())
    df = df_raw[clear].copy()

    n_pairs = len(df)
    q = df["question_id"].nunique()
    pairs = df["pair_id"].nunique()
    print(f"  n={n_pairs} swap pairs ({n_dropped} dropped: tie/no-answer in either ordering)"
          f" | clusters: {pairs} response pairs ({q} questions)")
    print(f"  S1 flip rate (all pairs, ties/no-answer included): {s1_rate:.0%}  "
          f"[95% CI: {s1_ci_lo:.0%}–{s1_ci_hi:.0%}]  ({n_flipped_s1}/{n_total})")
    print("  Design: long format — each pair contributes 2 rows (original + flipped)")
    print("  Model: picked_position_A ~ position_is_A, clustered by pair_id")

    # Reshape to long format: 2 rows per swap pair
    # Row 1 (original order): position_is_A=1, picked=1 if verdict_original=="A"
    # Row 2 (flipped order):  position_is_A=0, picked=1 if verdict_flipped=="B"
    #   (verdict_flipped is already re-mapped to model_a/model_b perspective;
    #    "B" in the re-mapped space means the response that was in position A won)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "pair_id": r["pair_id"],
            "question_id": r["question_id"],
            "position_is_A": 1,
            "picked_position_A": int(r["verdict_original"] == "A"),
        })
        rows.append({
            "pair_id": r["pair_id"],
            "question_id": r["question_id"],
            "position_is_A": 0,
            # In the flipped trial, the response that WAS model_a is now in position B.
            # verdict_flipped is re-mapped to model space, so "A" = model_a won.
            # model_a being in position B and winning means position B was picked,
            # i.e. position A was NOT picked.
            "picked_position_A": int(r["verdict_flipped"] != "A"),
        })
    long_df = pd.DataFrame(rows)

    n_long = len(long_df)
    print(f"  Long-format rows: {n_long}")

    # Cluster-robust logit, clustered by pair_id.
    # Each pair produces exactly 2 rows (original + flipped trial) that share
    # the same prompt, model outputs, and comparison — so the correlated unit
    # is the pair, not the question. Different pairs within a question involve
    # different response texts, so dependence arises at the response-pair level
    # rather than the question level.
    res = smf.logit(
        "picked_position_A ~ position_is_A",
        data=long_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": long_df["pair_id"]}, disp=False)

    beta1 = res.params["position_is_A"]
    p1 = res.pvalues["position_is_A"]

    beta0 = res.params["Intercept"]
    se0 = res.bse["Intercept"]

    # P(pick response | in position A) vs P(pick response | in position B)
    p_pick_when_A = logit_to_prob(beta0 + beta1)
    p_pick_when_B = logit_to_prob(beta0)
    cov = res.cov_params()
    var_sum = (cov.loc["Intercept", "Intercept"]
               + cov.loc["position_is_A", "position_is_A"]
               + 2 * cov.loc["Intercept", "position_is_A"])
    ci_A_lo, ci_A_hi = prob_ci(beta0 + beta1, np.sqrt(var_sum))
    ci_B_lo, ci_B_hi = prob_ci(beta0, se0)

    or_position = np.exp(beta1)
    sig = "YES" if p1 < 0.05 else "NO"
    p_fmt = "< 0.0001" if p1 < 0.0001 else f"{p1:.4f}"

    print(f"  P(pick response | in position A): {p_pick_when_A:.0%}  [95% CI: {ci_A_lo:.0%}–{ci_A_hi:.0%}]")
    print(f"  P(pick response | in position B): {p_pick_when_B:.0%}  [95% CI: {ci_B_lo:.0%}–{ci_B_hi:.0%}]")
    print(f"  Position effect OR: {or_position:.3f}  (>1 = prefer position A)")
    print(f"  p-value: {p_fmt}  →  Significant positional bias: {sig}")

    # S2 flip rate — among clear-verdict pairs only. This is the clean
    # signal: "of the pairs where the judge actually picked a side, did
    # flipping the order change its mind?"
    n_flipped_s2 = int(df["position_changed_verdict"].sum())
    s2_rate = n_flipped_s2 / n_pairs
    s2_ci_lo, s2_ci_hi = wilson_ci(n_flipped_s2, n_pairs)
    print(f"\n  S2 flip rate (committed pairs only): {s2_rate:.0%}  "
          f"[95% CI: {s2_ci_lo:.0%}–{s2_ci_hi:.0%}]")
    print(f"  ({n_flipped_s2}/{n_pairs} pairs changed verdict when positions reversed)")
    print(f"  (S1 {s1_rate:.0%} vs S2 {s2_rate:.0%} — gap reflects how often tie/no-answer landed in one ordering but not the other)")

    return {
        "test": "Positional",
        "n": n_pairs,
        "clusters": f"{pairs} pairs, {q} questions",
        "estimate": f"OR={or_position:.3f} position effect | S1={s1_rate:.0%} S2={s2_rate:.0%}",
        "ci_95": f"p={p_fmt}",
        "p_value": p_fmt,
        "finding": "Bias detected" if sig == "YES" else "Robust (no bias)",
        "method": "cluster-robust logit (long format)",
    }


# 3. Human agreement

def run_human_agreement(path: str = "results/human_agreement.csv"):
    print("Human Agreement")

    df = pd.read_csv(path)
    df["agreed_int"] = df["agreed"].astype(int)

    n = len(df)
    q = df["question_id"].nunique()
    j = df["judge"].nunique()
    print(f"  n={n} rows | clusters: {q} questions, {j} judges")
    print("  Model: agreed_int ~ 1, clustered by question_id")

    # Cluster-robust logit, intercept-only, clustered by question_id
    res = smf.logit(
        "agreed_int ~ 1",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["question_id"]}, disp=False)

    beta0 = res.params["Intercept"]
    se0 = res.bse["Intercept"]

    p_agree, (ci_lo, ci_hi) = logit_to_prob(beta0), prob_ci(beta0, se0)

    # Two-sided z-test vs empirical majority-class baseline — the rate of
    # "always predict the most frequent human label". Computed from the actual
    # human_winner distribution so this works for any dataset (MT-Bench ~0.385,
    # JudgeBench ~0.542).
    baseline = df["human_winner"].value_counts(normalize=True).max()
    print(f"  Majority-class baseline (from data): {baseline:.3f}")
    baseline_logodds = np.log(baseline / (1 - baseline))
    z_stat = (beta0 - baseline_logodds) / se0
    p_val = pvalue_from_z(z_stat)
    p_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
    sig = "YES" if p_val < 0.05 else "NO"

    print(f"  P(agree with human): {p_agree:.0%}  [95% CI: {ci_lo:.0%}–{ci_hi:.0%}]")
    print(f"  p-value: {p_fmt} (vs {baseline:.1%} max-class baseline)  →  Significant: {sig}")

    # Judge-level breakdown (judges are the human raters being compared against)
    print("\n  Agreement by judge (top 5 judges by volume):")
    judge_stats = df.groupby("judge")["agreed_int"].agg(["mean", "count"])
    judge_stats = judge_stats.sort_values("count", ascending=False).head(5)
    for judge, row in judge_stats.iterrows():
        print(f"    {judge:<30}  {row['mean']:.0%}  (n={int(row['count'])})")

    return {
        "test": "Human Agreement",
        "n": n,
        "clusters": f"{q} questions, {j} judges",
        "estimate": f"{p_agree:.0%} agreement",
        "ci_95": f"[{ci_lo:.0%}, {ci_hi:.0%}]",
        "p_value": p_fmt,
        "finding": (
            ("Above baseline" if p_agree > baseline else "Below baseline")
            if sig == "YES"
            else "Not different from baseline"
        ),
        "method": "cluster-robust logit",
    }


# 4. Family-preference bias (judge vs same-family MT-Bench model)

def run_family_preference(path: str = "results/claude/family_preference.csv"):
    df = pd.read_csv(path)
    df["pair_id"] = df.apply(response_pair_id, axis=1)

    # Detect column names — supports judge_picked_claude_v1, judge_picked_llama_13b, etc.
    judge_col = next(c for c in df.columns if c.startswith("judge_picked_"))
    human_col = next(c for c in df.columns if c.startswith("human_picked_"))
    target_model = judge_col.replace("judge_picked_", "")

    print(f"Family-Preference Bias (judge vs {target_model})")

    n = len(df)
    q = df["question_id"].nunique()
    pairs = df["pair_id"].nunique()
    print(f"  n={n} rows ({target_model} involved) | clusters: {pairs} pairs, {q} questions")
    print(f"  Model: {judge_col} ~ 1, clustered by pair_id")

    # Cluster-robust logit: did judge pick family-member model?
    df["_judge_picked_int"] = df[judge_col].astype(int)
    res = smf.logit(
        "_judge_picked_int ~ 1",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["pair_id"]}, disp=False)

    beta0 = res.params["Intercept"]
    se0 = res.bse["Intercept"]

    p_judge, (ci_lo, ci_hi) = logit_to_prob(beta0), prob_ci(beta0, se0)

    # Human rate as baseline
    human_rate = df[human_col].mean()
    baseline_logodds = np.log(human_rate / (1 - human_rate))
    z_stat = (beta0 - baseline_logodds) / se0
    p_val = pvalue_from_z(z_stat)
    p_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
    sig = "YES" if p_val < 0.05 else "NO"
    diff = p_judge - human_rate

    print(f"  Judge picks {target_model}: {p_judge:.1%}  [95% CI: {ci_lo:.1%}–{ci_hi:.1%}]")
    print(f"  Humans pick {target_model}: {human_rate:.1%}")
    print(f"  Difference: {diff:+.1%}  (positive = judge favors {target_model} more than humans)")
    print(f"  p-value: {p_fmt} (vs human rate baseline)  →  Significant: {sig}")

    return {
        "test": "Family-Preference",
        "n": n,
        "clusters": f"{pairs} pairs, {q} questions",
        "estimate": f"judge {p_judge:.1%} vs human {human_rate:.1%} ({diff:+.1%})",
        "ci_95": f"[{ci_lo:.1%}, {ci_hi:.1%}]",
        "p_value": p_fmt,
        "finding": "Bias detected" if sig == "YES" else "No bias",
        "method": "cluster-robust logit",
    }


# 5. Self-preference (fresh outputs — separate from MT-Bench bias suite)

def run_self_preference(path: str):

    df = pd.read_csv(path)
    judge = df["judge"].iloc[0]
    own_model = df["own_model"].iloc[0]
    opponents = sorted(df["opponent_model"].unique())

    print(f"Self-Preference — judge: {judge.upper()} (own: {own_model})")
    print(f"  n={len(df)} rows | clusters: {df['question_id'].nunique()} questions | opponents: {opponents}")
    print("  Model: self_pref ~ C(opponent_model), clustered by question_id")
    print("  self_pref=1 only when judge picks own model in BOTH orderings (position-invariant)")

    # Overall intercept-only model
    res_overall = smf.logit("self_pref ~ 1", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["question_id"]}, disp=False
    )
    beta0 = res_overall.params["Intercept"]
    se0 = res_overall.bse["Intercept"]
    p_overall, (ci_lo, ci_hi) = logit_to_prob(beta0), prob_ci(beta0, se0)

    # Null: p=0.25 (random judge picking own in both orderings: 0.5 × 0.5)
    null_logodds = np.log(0.25 / 0.75)
    z_stat = (beta0 - null_logodds) / se0
    p_val = pvalue_from_z(z_stat)
    p_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
    sig = "YES" if p_val < 0.05 else "NO"

    print(f"\n  Overall self-preference rate: {p_overall:.1%}  [95% CI: {ci_lo:.1%}–{ci_hi:.1%}]")
    print(f"  p-value: {p_fmt} (vs 25% null)  →  Significant: {sig}")

    # Per-opponent model with FEs
    res_fe = smf.logit("self_pref ~ C(opponent_model)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["question_id"]}, disp=False
    )
    print(f"\n  Per-opponent estimates (reference = {opponents[0]}):")
    print(f"  {'Opponent':<16} {'Self-pref%':>10} {'95% CI':>18} {'p-value':>10} {'BH':>8}")

    b0 = res_fe.params["Intercept"]
    se_b0 = res_fe.bse["Intercept"]
    cov = res_fe.cov_params()

    opponent_results = []
    for opp in opponents:
        if opp == opponents[0]:
            b = b0
            se = se_b0
        else:
            pname = f"C(opponent_model)[T.{opp}]"
            if pname not in res_fe.params:
                continue
            b = b0 + res_fe.params[pname]
            se = np.sqrt(
                cov.loc["Intercept", "Intercept"]
                + cov.loc[pname, pname]
                + 2 * cov.loc["Intercept", pname]
            )
        p_opp = logit_to_prob(b)
        ci_opp = prob_ci(b, se)
        z = (b - null_logodds) / se
        p_raw = pvalue_from_z(z)
        opponent_results.append((opp, p_opp, ci_opp, p_raw))

    # BH(q=0.10) within judge
    raw_pvals = [r[3] for r in opponent_results]
    n_p = len(raw_pvals)
    indexed = sorted(enumerate(raw_pvals), key=lambda x: x[1])
    bh = [1.0] * n_p
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), 1):
        adj = min(prev, p * n_p / (n_p - rank + 1))
        bh[i] = adj
        prev = adj

    for i, (opp, p_opp, ci_opp, p_raw) in enumerate(opponent_results):
        p_str = "< 0.0001" if p_raw < 0.0001 else f"{p_raw:.4f}"
        bh_str = "< 0.0001" if bh[i] < 0.0001 else f"{bh[i]:.4f}"
        sig_marker = " *" if bh[i] < 0.10 else ""
        print(f"  {opp:<16} {p_opp:>9.1%}  [{ci_opp[0]:.1%}–{ci_opp[1]:.1%}]  "
              f"p={p_str}  BH={bh_str}{sig_marker}")

    return {
        "test": "Self-Preference",
        "n": len(df),
        "clusters": f"{df['question_id'].nunique()} questions",
        "estimate": f"{p_overall:.1%} overall self-pref",
        "ci_95": f"[{ci_lo:.1%}, {ci_hi:.1%}]",
        "p_value": p_fmt,
        "finding": "Bias detected" if sig == "YES" else "No bias",
        "method": "cluster-robust logit + opponent FEs",
    }


def latest(results_dir: str, prefix: str):
    """Return the most recently modified file matching results_dir/<prefix>*.csv,
    or None if no matches. Excludes _summary.csv files so summary outputs
    don't get picked up as raw results."""
    matches = [
        f for f in glob.glob(f"{results_dir}/{prefix}*.csv")
        if "_summary" not in f
    ]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing bias CSVs and where summary is saved (default: results)"
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset coverage
    df_full = pd.read_csv("data/mt_bench.csv")
    turn1 = df_full[df_full["turn"] == 1]
    print("CLUSTER-ROBUST LOGISTIC REGRESSION — CLUSTERING ROBUSTNESS CHECK")
    print("\nDataset coverage")
    print(f"  Rows:               {len(turn1)}")
    print(f"  Unique questions:   {turn1['question_id'].nunique()}")
    pairs = turn1.apply(lambda r: tuple(sorted([r.model_a, r.model_b])), axis=1).nunique()
    print(f"  Unique model pairs: {pairs}")
    print(f"  Unique judges:      {turn1['judge'].nunique()}")

    results = []
    for name, runner in [
        ("verbosity_bias",   run_verbosity),
        ("positional_bias",  run_positional),
        ("human_agreement",  run_human_agreement),
    ]:
        path = latest(results_dir, name)
        if path is None:
            print(f"\n  ({name} skipped — no file found in {results_dir})")
            continue
        results.append(runner(path=path))

    # Family-preference (MT-Bench same-family model identity)
    _fp = glob.glob(f"{results_dir}/family_preference_*.csv")
    if _fp:
        results.append(run_family_preference(path=max(_fp, key=os.path.getmtime)))
    else:
        print(f"\n  (Family-preference skipped — no file found in {results_dir})")

    # MT-Bench summary 
    print("MT-BENCH BIAS SUMMARY (verbosity / positional / human agmt / family-pref)")
    header = f"{'Bias':<18} {'Estimate':<28} {'95% CI':<18} {'p-value':<12} {'Finding'}"
    print(header)
    for r in results:
        print(f"{r['test']:<18} {r['estimate']:<28} {r['ci_95']:<18} {r['p_value']:<12} {r['finding']}")

    # Self-preference (fresh outputs, separate section) 
    _sp_v2 = [f for f in glob.glob(f"{results_dir}/self_preference_*.csv") if "_summary" not in f]
    if _sp_v2:
        sp2_path = max(_sp_v2, key=os.path.getmtime)
        print("SELF-PREFERENCE — FRESH OUTPUTS (cluster-robust, separate from MT-Bench suite)")
        sp2_result = run_self_preference(path=sp2_path)
        print("SELF-PREFERENCE SUMMARY")
        print(f"  {sp2_result['estimate']}  {sp2_result['ci_95']}  p={sp2_result['p_value']}  → {sp2_result['finding']}")
        results.append(sp2_result)

    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/mixed_effects_summary_{ts}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
