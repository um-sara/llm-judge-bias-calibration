"""
Generate fresh MT-Bench responses for self-preference.

For each of 5 models × 80 MT-Bench turn=1 questions × 3 samples, generates a
response at temperature=0.7 (following Zheng et al. for MT-Bench response
generation). Each sample is an independent generation at temperature=0.7, so
the three naturally differ.

Models:
    claude_haiku   — claude-haiku-4-5 via Anthropic API
    llama3b        — llama3.2:3b via local Ollama
    llama70b       — meta-llama/llama-3.1-70b-instruct via OpenRouter
    qwen32b        — qwen/qwen3-32b via OpenRouter (non-thinking mode)
    gpt4o          — openai/gpt-4o via OpenRouter

Output:
    data/fresh_outputs/fresh_outputs_{model}.csv
    Schema: question_id, category, sample, question_text, model, answer

Usage:
    python src/generate_model_outputs.py                    # all models
    python src/generate_model_outputs.py --models claude_haiku llama3b
    python src/generate_model_outputs.py --samples 3          # default
"""

import argparse
import ast
import os
import sys
import time

import anthropic
import ollama
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, os.path.dirname(__file__))
from judge_openrouter import MODEL_IDS, PROVIDER_MAP

load_dotenv()

# MT-Bench category mapping (Zheng et al.)

CATEGORY_MAP = {
    range(81, 91):  "writing",
    range(91, 101): "roleplay",
    range(101, 111): "reasoning",
    range(111, 121): "math",
    range(121, 131): "coding",
    range(131, 141): "extraction",
    range(141, 151): "stem",
    range(151, 161): "humanities",
}

def get_category(question_id: int) -> str:
    for r, cat in CATEGORY_MAP.items():
        if question_id in r:
            return cat
    return "unknown"


# Generation clients

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def generate_anthropic(question: str, model: str = "claude-haiku-4-5") -> str:
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.7,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def generate_ollama(question: str, model: str = "llama3.2:3b") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.7},
    )
    return response["message"]["content"]


def generate_openrouter(question: str, model_key: str) -> str:
    model_id = MODEL_IDS[model_key]
    extra = {
        "provider": {
            "order": PROVIDER_MAP[model_id],
            "allow_fallbacks": False,
        }
    }

    # Qwen3-32B: disable thinking mode
    content = question + " /no_think" if model_id == "qwen/qwen3-32b" else question

    max_retries = 5
    backoff = 10
    for attempt in range(max_retries):
        try:
            response = openrouter_client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                temperature=0.7,
                messages=[{"role": "user", "content": content}],
                extra_body=extra,
            )
            break
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries - 1}...")
            time.sleep(wait)

    if not response.choices:
        return ""
    text = response.choices[0].message.content
    return text or ""


# Model dispatch

MODEL_GENERATORS = {
    "claude_haiku": lambda q: generate_anthropic(q, model="claude-haiku-4-5"),
    "llama3b":      lambda q: generate_ollama(q, model="llama3.2:3b"),
    "llama70b":     lambda q: generate_openrouter(q, model_key="llama70b"),
    "qwen32b":      lambda q: generate_openrouter(q, model_key="qwen32b"),
    "gpt4o":        lambda q: generate_openrouter(q, model_key="gpt4o"),
}

ALL_MODELS = list(MODEL_GENERATORS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Generate fresh MT-Bench responses for self-preference."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Models to generate for (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of independent outputs per question (default: 3)",
    )
    args = parser.parse_args()

    os.makedirs("data/fresh_outputs", exist_ok=True)

    print("Loading MT-Bench questions...")
    df = pd.read_csv("data/mt_bench.csv")
    questions = (
        df[df["turn"] == 1]
        .drop_duplicates("question_id")[["question_id"]]
        .copy()
    )

    # Extract question text from conversation_a (first user turn)
    def extract_question(row):
        conv = df[(df["question_id"] == row["question_id"]) & (df["turn"] == 1)].iloc[0]
        raw = conv["conversation_a"]
        turns = raw if isinstance(raw, list) else ast.literal_eval(raw)
        for turn in turns:
            if turn["role"] == "user":
                return turn["content"]
        return ""

    questions["question_text"] = questions.apply(extract_question, axis=1)
    questions["category"] = questions["question_id"].apply(get_category)
    print(f"Loaded {len(questions)} unique questions across 8 categories.\n")

    for model_key in args.models:
        out_path = f"data/fresh_outputs/fresh_outputs_{model_key}.csv"

        # Resume support: skip already-generated rows
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            done = set(zip(existing["question_id"], existing["sample"]))
            print(f"{model_key}: resuming — {len(existing)} rows already generated.")
        else:
            existing = pd.DataFrame()
            done = set()

        generate_fn = MODEL_GENERATORS[model_key]
        rows = []

        total = len(questions) * args.samples
        completed = len(done)

        print(f"{model_key}: generating {total - completed} remaining outputs "
              f"({len(questions)} questions × {args.samples} samples)...")

        for _, q in questions.iterrows():
            for sample in range(args.samples):
                if (q["question_id"], sample) in done:
                    continue

                try:
                    answer = generate_fn(q["question_text"])
                except Exception as e:
                    print(f"  ERROR q={q['question_id']} sample={sample}: {e}")
                    answer = ""

                rows.append({
                    "question_id":   q["question_id"],
                    "category":      q["category"],
                    "sample":          sample,
                    "question_text": q["question_text"],
                    "model":         model_key,
                    "answer":        answer,
                })
                completed += 1
                print(f"  [{completed}/{total}] q={q['question_id']} sample={sample} "
                      f"({'ok' if answer else 'empty'})")

                # Save incrementally every 20 rows so crashes don't lose progress
                if len(rows) % 20 == 0:
                    batch = pd.DataFrame(rows)
                    combined = pd.concat([existing, batch], ignore_index=True) if not existing.empty else batch
                    combined.to_csv(out_path, index=False)

        if rows:
            batch = pd.DataFrame(rows)
            combined = pd.concat([existing, batch], ignore_index=True) if not existing.empty else batch
            combined.to_csv(out_path, index=False)

        print(f"  Saved to {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
