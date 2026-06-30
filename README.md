# LLM-as-Judge Reliability Auditor

An end-to-end reliability audit of LLM-as-a-judge systems. Each judge evaluates pairs of model responses from MT-Bench; six biases are measured, then modeled with cluster-robust statistics to account for non-independence in the data, then run a pre-registered calibration experiment to fix the one failure mode that can actually be corrected (tie under-calling). Findings are then re-tested on a second dataset (JudgeBench) to check whether they generalize.

**Five judges are audited on equal footing**, all through the identical pipeline:

| Judge            | Access            |
| ---------------- | ----------------- |
| Claude Haiku 4.5 | Anthropic API     |
| Llama 3.2 3B     | local, via Ollama |
| Llama 3.1 70B    | OpenRouter        |
| Qwen3-32B        | OpenRouter        |
| GPT-4o           | OpenRouter        |

---

## Datasets

**MT-Bench Human Judgments** (`lmsys/mt_bench_human_judgments`): primary
- 3,355 rows total; **1,689 turn=1 rows** used throughout
- 80 questions across 8 categories: writing, roleplay, extraction, reasoning, math, coding, STEM, humanities
- 15 model pairs, 65 human annotators
- One row per annotator per model pair, ground truth is per-annotator votes, not majority labels

**JudgeBench**: out-of-distribution external-validity check
- 620 pairs (350 GPT-4o split + 270 Claude-3.5-Sonnet split)
- Binary, objective ground-truth labels (`A>B` / `B>A`); **no ties**
- Only positional bias and label agreement are re-run on it (verbosity and self-preference don't apply)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_key_here     # required for the Claude judge
OPENROUTER_API_KEY=your_key_here    # required for the llama70b / qwen32b / gpt4o judges
```

For the Llama 3.2 3B judge, install [Ollama](https://ollama.com), then:
```bash
ollama pull llama3.2:3b
ollama serve            
```

Download the data:
```bash
python src/load_data.py        # MT-Bench
python src/load_judgeBench.py  # JudgeBench
```

---

## Project structure

```
src/
├── load_data.py            # Download MT-Bench from HuggingFace
├── load_judgeBench.py      # Download JudgeBench (cross-dataset validation)
├── judge_claude.py         # Claude Haiku 4.5 judge — judge_pair(), temperature=0
├── judge_ollama.py         # Llama 3.2 3B judge (drop-in, same interface)
├── judge_openrouter.py        # OpenRouter judges (llama70b, qwen32b, gpt4o)
├── api_cache.py               # On-disk cache for judge API calls
├── run_bias_suite.py          # Runs all bias tests for one judge
├── verbosity_bias.py          # Does the judge prefer longer answers?
├── positional_bias.py         # Does the verdict flip when A/B are swapped?
├── family_preference.py       # Does the judge favor a same-family MT-Bench model?
├── generate_model_outputs.py  # Generate fresh outputs (input to self-preference)
├── self_preference.py         # Does the judge favor its own fresh outputs vs opponents?
├── human_agreement.py         # How often does the judge agree with human annotators?
├── mixed_effects.py           # Cluster-robust logistic regression for every bias test
├── calibrate_judge.py         # Calibration experiment: 6 variants to fix tie under-calling
├── category_breakdown.py      # Per-category agreement breakdown
└── zheng_verbosity_bias.py    # Replicates Zheng et al. repetitive-list verbosity bias
```

---

## Running the pipeline

`--judge` accepts `claude`, `ollama`, `llama70b`, `qwen32b`, or `gpt4o`.

```bash
# 1. Evaluation suite — runs all six tests for one judge in one command:
#    verbosity, positional, human agreement, family-preference, plus
#    self-preference and Zheng verbosity test when their data is present.
python src/run_bias_suite.py --judge claude

#    External validity: re-run the applicable tests on JudgeBench
python src/run_bias_suite.py --judge claude --dataset judgeBench

# 2. Cluster-robust models (auto-resolves the most recent CSV per bias type)
python src/mixed_effects.py --results-dir results/claude

# 3. Calibration — pre-registered primary run (n=400, all 6 variants)
python src/calibrate_judge.py --judge claude

#    Full-dataset confirmation (n=1689, swap-consistency only)
python src/calibrate_judge.py --judge claude --full --variant swap
```

---

## Key results

### Verbosity bias — does the judge prefer the longer answer?

Rate of picking the longer answer, with `C(model_a) + C(model_b)` fixed effects controlling for the quality confound (better models also write longer):

| Judge | % pick longer | 95% CI | p-value | Finding |
|---|---|---|---|---|
| Claude Haiku 4.5 | 68% | [66%, 70%] | 0.28 | **No bias** |
| Llama 3.2 3B | 67% | [65%, 70%] | <0.0001 | Bias detected |
| Llama 3.1 70B | 72% | [69%, 74%] | 0.0019 | Bias detected |
| Qwen3-32B | 71% | [69%, 73%] | 0.0064 | Bias detected |
| GPT-4o | 72% | [69%, 74%] | 0.0349 | Bias detected |

Takeaway: the raw "picks longer" rate is ~67–72% for *every* judge, but the fixed effects dissolve it only for Claude (p=0.28). Reporting the raw rate alone would have falsely accused Claude of verbosity bias. For Claude, the longer answer was usually just the better one, so length stood in for quality rather than driving the verdict.

### Verbosity Bias (Zheng replication) — is the judge fooled by pure padding?

The observational test above measures correlation with length, but this one is causal. Following Zheng et al. 2023, each list answer is padded with rephrased duplicates of its own items, more words, zero new information, and the rate at which the judge then prefers the padded version over the original is measured. Reported is the failure rate in original order (padded shown second), n=85 eligible list responses:

| Judge            | Failure rate | Finding          |
| ---------------- | ------------ | ---------------- |
| Claude Haiku 4.5 | 2.4%         | Immune           |
| Llama 3.2 3B     | **96.5%**    | **Catastrophic** |
| Llama 3.1 70B    | 0.0%         | Immune           |
| Qwen3-32B        | 10.6%        | Mostly robust    |
| GPT-4o           | 2.4%         | Immune           |

Reference points from Zheng et al. (n=23): GPT-4 ~4%, GPT-3.5 ~36%, Vicuna-13B ~63%.

Takeaway: three of five judges are robust to pure padding (Llama 3.1 70B, Claude, GPT-4o). Llama 3.2 3B is the outlier, fooled almost every time and worse than any model Zheng tested. This is the same judge that flips on 57% of swapped pairs: it isn't reasoning about content, so more words reliably win. The causal test confirms Llama 3.2 3B's length preference is genuine bias.

### Positional bias — does the verdict flip when A and B are swapped?

| Judge | Position OR | p-value | Flip rate (S1) | Finding |
|---|---|---|---|---|
| Claude Haiku 4.5 | 1.056 | 0.58 | 17% | No directional bias |
| Llama 3.2 3B | 0.963 | 0.68 | **57%** | No directional bias |
| Llama 3.1 70B | 1.069 | 0.48 | 15% | No directional bias |
| Qwen3-32B | 1.002 | 0.98 | 15% | No directional bias |
| GPT-4o | 1.031 | 0.75 | 13% | No directional bias |

Takeaway: No judge has a *directional* A-vs-B preference. But the **flip rate** (how often the verdict changes when you swap the order) tells a different story: Llama 3.2 3B flips on 57% of pairs, near coin-flip noise, while the others sit at 13–17%. This single number determines whether the calibration fix below works.

### Human agreement — agreement with individual MT-Bench annotators

| Judge            | Strict agreement | 95% CI     | Zheng S2 | 95% CI     | vs. chance baseline |
| ---------------- | ---------------- | ---------- | -------- | ---------- | ------------------- |
| Claude Haiku 4.5 | 63%              | [59%, 66%] | 82%      | [79%, 85%] | Above (p<0.0001)    |
| Llama 3.2 3B     | 48%              | [44%, 52%] | 64%      | [61%, 68%] | Above (p<0.0001)    |
| Llama 3.1 70B    | 62%              | [58%, 65%] | 81%      | [77%, 83%] | Above (p<0.0001)    |
| Qwen3-32B        | 62%              | [59%, 65%] | 80%      | [77%, 83%] | Above (p<0.0001)    |
| GPT-4o           | 63%              | [60%, 67%] | 81%      | [78%, 84%] | Above (p<0.0001)    |

Takeaway: Two metrics are reported throughout. 
1. **Strict** agreement over all rows 
2. **Zheng S2**, restricted to rows where both judge and human gave a clear winner. 
The gap between them is large for every judge because all five **under-call ties**. 

### Family-preference — does the judge favor a same-family MT-Bench model?

Compares the judge's pick rate for its own model family against the human pick rate for that same family (the baseline). Qwen3-32B is skipped since MT-Bench contains no Qwen model.

| Judge            | Family target | Judge rate | Human rate | Δ          | Finding                   |
| ---------------- | ------------- | ---------- | ---------- | ---------- | ------------------------- |
| Claude Haiku 4.5 | claude-v1     | 67.9%      | 49.1%      | +18.8pp    | Bias detected             |
| Llama 3.2 3B     | llama-13b     | 29.2%      | 9.2%       | +20.0pp    | Bias detected             |
| Llama 3.1 70B    | llama-13b     | 3.1%       | 9.2%       | **−6.1pp** | Bias detected (negative)  |
| Qwen3-32B        | —             | —          | —          | —          | n/a (no Qwen in MT-Bench) |
| GPT-4o           | gpt-4         | 79.6%      | 62.8%      | +16.8pp    | Bias detected             |

Takeaway: Most judges over-favor their family. Llama 3.1 70B notably *under*-picks the older llama-13b (a significant negative effect, p=0.0001), it doesn't mistake a weak same-family model for a good answer.

### Self-preference — does the judge favor its own fresh outputs? (null = 25%)

Each pair is judged in both orderings, and a pair is counted as self-preference only when the judge picks its own output in *both*, which removes positional bias. Under that coding, a judge with no preference picks its own output with probability ½ in each ordering, so the probability of picking it in both is ½ × ½ = 25%. The null is therefore **25%, not 50%**: a rate above 25% indicates genuine self-preference, while a rate below it indicates a judge that systematically avoids its own outputs (as Llama 3.2 3B does, at 7.0%).

| Judge | Self-pref rate | 95% CI | p-value | Finding |
|---|---|---|---|---|
| Claude Haiku 4.5 | 66.5% | [59.6%, 72.7%] | <0.0001 | Strong self-preference |
| Llama 3.2 3B | 7.0% | [4.4%, 10.8%] | <0.0001 | **Anti**-self-preference |
| Llama 3.1 70B | 21.1% | [17.1%, 25.9%] | 0.11 | No bias |
| Qwen3-32B | 57.9% | [51.8%, 63.8%] | <0.0001 | Strong self-preference |
| GPT-4o | 22.9% | [18.9%, 27.5%] | 0.36 | No bias |

Takeaway: Claude and Qwen3-32B strongly prefer their own outputs; GPT-4o and
Llama 3.1 70B sit at the 25% null; Llama 3.2 3B actively prefers its opponents.

---

## Calibration experiment — fixing tie under-calling

Every judge under-calls ties. Humans rate ~24% of MT-Bench pairs a tie (405/1,689), but baseline tie sensitivity (how often the judge says "tie" when the human did) is far lower for four of the five, GPT-4o is the only one that calls ties at roughly the human rate:

| Judge | Baseline tie sensitivity |
|---|---|
| Claude Haiku 4.5 | 7.4% |
| Llama 3.2 3B | 4.9% |
| Llama 3.1 70B | 8.1% |
| Qwen3-32B | 9.6% |
| GPT-4o | 24.0% |

Six variants, five prompt-based (Baseline, Explicit tie, Chain-of-thought, CoT + explicit tie, Rubric) and one structural rule, **Swap-consistency**: call a tie whenever the A-first and B-first verdicts disagree.

### Prompt variants barely move anything

At n=400, the Cochran's Q omnibus for agreement is non-significant for four of five judges, so the gate to pairwise testing never even opens:

| Judge | Q (agreement) p | Q (tie-sens) p | Verdict |
|---|---|---|---|
| Claude Haiku 4.5 | 0.96 | 0.20 | Prompt variants do nothing |
| Llama 3.1 70B | 0.21 | 0.15 | Prompt variants do nothing |
| Qwen3-32B | 0.70 | 0.87 | Prompt variants do nothing |
| GPT-4o | 0.68 | 0.062 | Prompt variants do nothing |
| Llama 3.2 3B | 7.7e-13 | 0.0002 | Significant — CoT+explicit-tie raises agreement 47.8%→59.0% |

Llama 3.2 3B is the only judge where prompts matter, and even there the gain comes from CoT + explicit-tie, not the pre-registered "Explicit tie" primary.

### Swap-consistency is the structural fix — and it works for 4 of 5 judges

Full-dataset run (n=1,689), Baseline → Swap-consistency:

| Judge | Flip rate | Agreement | Tie sensitivity | Tie specificity | Clean fix? |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | 17% | 64.0% → 65.2% (p=0.09) | 7.4% → 27.4% | 0.99 → 0.91 | Yes |
| Llama 3.1 70B | 15% | 62.8% → 64.7% (p=0.009) | 8.1% → 29.6% | 1.00 → 0.89 | Yes |
| Qwen3-32B | 15% | 62.0% → 63.1% (p=0.18) | 9.6% → 31.9% | 0.99 → 0.88 | Yes |
| GPT-4o | 13% | 65.4% → 66.6% (p=0.13) | 24.0% → 45.9% | 0.95 → 0.86 | Yes |
| Llama 3.2 3B | 57% | 48.3% → 44.0% (p=0.003 ↓) | 4.9% → 66.7% | 0.97 → **0.45** | No (noise) |

(All tie-sensitivity gains are individually significant, p < 1e-20; the question is whether
they're *real* ties or noise.)

For the four low-flip-rate judges, swap-consistency raises tie sensitivity 3–5×, holds
agreement flat-to-up, and keeps tie specificity high (≥0.86), the tie calls stay reserved for genuinely ambiguous pairs. **Deployable recommendation: swap-consistency, at 2× API call cost.**

For Llama 3.2 3B it backfires: it flips on 57% of pairs, so swap-consistency calls a tie on 58% of pairs, specificity collapses to 0.45, and overall agreement *drops* significantly. The "ties" it produces are coin flips, not ambiguity detection.

### Signal vs. noise: three diagnostics

Swap-consistency manufactures a tie whenever the judge disagrees with itself. To tell a real ambiguity signal from a noisy judge flipping coins, a judge must pass all three:

1. **Positional flip rate** (`positional_bias.py`) — an upper bound on the noise. Above ~30% and swap-consistency is unlikely to be clean. (Llama 3.2 3B: 57%.)
2. **Tie specificity vs. ground truth** — high (≥~0.85) means tie calls are reserved for
   ambiguous pairs. (Llama 3.2 3B collapses to 0.45.)
3. **Agreement doesn't fall** — noise drags real winners into the tie bucket. (Llama 3.2 3B agreement drops.)

---

## External validity — JudgeBench (n=620)

JudgeBench is an OOD second dataset used to test whether the MT-Bench findings hold elsewhere. Only two tests transfer and are re-run here, positional bias and label agreement, because JudgeBench has no model fixed effects (verbosity skipped) and no fresh-output design (self-preference skipped). Its labels are binary, objective ground-truth (`A>B` / `B>A`, **no ties**), drawn from reasoning-heavy sources, across two splits: 350 GPT-4o-generated and 270 Claude-3.5-Sonnet-generated pairs.

| Judge | Positional OR (p) | Label agreement | vs. baseline |
|---|---|---|---|
| Claude Haiku 4.5 | 1.072 (0.79) | 26% [23–30%] | Below |
| Llama 3.2 3B | 0.903 (0.54) | 43% [39–47%] | Below |
| Llama 3.1 70B | 0.987 (0.92) | 58% [54–62%] | Not different |
| Qwen3-32B | 1.105 (0.50) | 56% [53–60%] | Not different |
| GPT-4o | 1.049 (0.74) | 63% [59–66%] | Above |

**Positional robustness replicates cleanly for all five judges**, the core positional
finding is externally valid. Agreement is a different story: it does *not* track MT-Bench
agreement, and two judges (Claude Haiku, Llama 3.2 3B) score below the chance baseline on this task. 

---

## Statistical design

- **Why cluster-robust:** MT-Bench rows sharing a prompt are not independent, a judge's behavior on a topic is correlated across model pairs, violating the IID assumption of a naive binomial test. All final models use cluster-robust logistic regression (Huber-White sandwich SEs).
- **Clustering:** `pair_id` (order-invariant `question_id + sorted(model_a, model_b)`) for verbosity / positional / family / self-preference; `question_id` for human agreement.
- **Verbosity:** `length_diff` in 100-word units, with `C(model_a) + C(model_b)` fixed effects to remove the quality-writes-longer confound.
- **Positional:** long-format design (2 rows per swap pair), tests directional bias; flip
  rate reported separately as the noise diagnostic.
- **Agreement metrics:** strict (all rows, penalizes tie under-calling) and Zheng S2
  (non-tie rows only) are both reported, S2 alone hides the tie failure mode.
- **Calibration multiplicity:** Cochran's Q omnibus must precede pairwise McNemar's; exact McNemar's (the handful of baseline tie calls makes the asymptotic approximation unreliable); BH(q=0.10) with Holm-Bonferroni as a sensitivity check.
- **Pre-registration:** n≥400 before reporting any effect; n=400 stratified sample
  (seed=42, fixed); "Explicit tie" committed as the primary variant before data collection, not post-hoc swapped for the winner.

---

## Key references

- Zheng et al. 2023 — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (arXiv:2306.05685)

