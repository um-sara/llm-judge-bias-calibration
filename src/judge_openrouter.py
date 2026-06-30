"""
Judge logic using models served via OpenRouter.

Usage (standalone test):
    python src/judge_openrouter.py --model llama70b

Supported model keys:
    llama70b  — meta-llama/llama-3.1-70b-instruct (pinned: Together, DeepInfra)
    qwen32b   — qwen/qwen3-32b                     (pinned: DeepInfra, Chutes)
    gpt4o     — openai/gpt-4o                      (pinned: OpenAI)
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from judge_claude import JUDGE_PROMPT
from api_cache import cached_call

load_dotenv()

# Model registry

MODEL_IDS = {
    "llama70b": "meta-llama/llama-3.1-70b-instruct",
    "qwen32b":  "qwen/qwen3-32b",
    "gpt4o":    "openai/gpt-4o",
}

# Pinned providers per model. allow_fallbacks=False means the call fails
# rather than silently routing to a different provider/quantization.
PROVIDER_MAP = {
    "meta-llama/llama-3.1-70b-instruct": ["Together", "DeepInfra"],
    "qwen/qwen3-32b":                    ["DeepInfra", "Chutes"],
    "openai/gpt-4o":                     ["OpenAI"],
}

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def judge_pair_openrouter(question: str, answer_a: str, answer_b: str, model: str = "llama70b") -> str:
    """
    Ask an OpenRouter-hosted model to pick the better answer.

    Args:
        question:  The question text
        answer_a:  Response A
        answer_b:  Response B
        model:     Model key from MODEL_IDS ("llama70b", "qwen32b", "gpt4o")
                   or a full OpenRouter model ID string

    Returns:
        "A", "B", "tie", or "no answer"
    """
    model_id = MODEL_IDS.get(model, model)

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    extra = {
        "provider": {
            "order": PROVIDER_MAP[model_id],
            "allow_fallbacks": False,
        }
    }

    # Qwen3-32B is a hybrid thinking model. DeepInfra ignores OpenRouter's
    # reasoning:{effort:"none"} flag. Use /no_think in the prompt instead —
    # this is baked into Qwen3's training and works at the model level
    # regardless of provider. max_tokens raised to 50 since thinking tokens
    # count against the budget even when suppressed.
    if model_id == "qwen/qwen3-32b":
        messages = [{"role": "user", "content": prompt + " /no_think"}]
        max_tokens = 50
    else:
        messages = [{"role": "user", "content": prompt}]
        max_tokens = 10

    def _call_api():
        max_retries = 5
        backoff = 10  # seconds; doubles each retry
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=messages,
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
        content = response.choices[0].message.content
        return content if content else ""

    raw = cached_call(
        judge=f"openrouter:{model}",
        model=model_id,
        prompt=messages[0]["content"],
        max_tokens=max_tokens,
        temperature=0.0,
        call_fn=_call_api,
    )

    if not raw:
        return "no answer"

    verdict = raw.strip().upper()
    if "TIE" in verdict:
        return "tie"
    elif verdict.startswith("A"):
        return "A"
    elif verdict.startswith("B"):
        return "B"
    else:
        return "no answer"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test one OpenRouter judge on a sample pair.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_IDS.keys()),
        default="llama70b",
        help="Model to test (default: llama70b)",
    )
    args = parser.parse_args()

    question = "What is the capital of France?"
    answer_a = "The capital of France is Paris."
    answer_b = (
        "France's capital city is Paris. It has been the country's capital since "
        "the 10th century and is home to the Eiffel Tower."
    )

    print(f"Testing OpenRouter judge: {args.model} ({MODEL_IDS[args.model]})")
    print(f"Pinned provider(s): {PROVIDER_MAP[MODEL_IDS[args.model]]}")
    print()

    verdict = judge_pair_openrouter(question, answer_a, answer_b, model=args.model)
    print(f"Question: {question}")
    print(f"Answer A: {answer_a}")
    print(f"Answer B: {answer_b}")
    print(f"Verdict:  {verdict}")
