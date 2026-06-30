"""
Core judge logic: send an A/B pair to Claude and get a verdict.
Start here — get one pair working before scaling up.
"""

import anthropic
import os
from dotenv import load_dotenv

from api_cache import cached_call

load_dotenv()

# Constructed lazily so that importing this module (other judges borrow
# JUDGE_PROMPT from here) does not require an Anthropic key — only actually
# calling the Claude judge does.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client

JUDGE_PROMPT = """You are evaluating two responses to a question. Decide which response is better.

Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Which response is better? Reply with only one of: "A", "B", or "tie".
"""


def judge_pair(question: str, answer_a: str, answer_b: str, model: str = "claude-haiku-4-5") -> str:
    """
    Ask the judge model to pick the better answer.
    Returns "A", "B", or "tie".
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    def _call_api():
        response = _get_client().messages.create(
            model=model,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raw = cached_call(
        judge="claude",
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=0.0,
        call_fn=_call_api,
    )

    verdict = raw.strip().upper()
    # Normalize to A, B, or tie
    if "TIE" in verdict:
        return "tie"
    elif verdict.startswith("A"):
        return "A"
    elif verdict.startswith("B"):
        return "B"
    else:
        return "no answer"


if __name__ == "__main__":
    # Test with one example pair
    question = "What is the capital of France?"
    answer_a = "The capital of France is Paris."
    answer_b = "France's capital city is Paris. It has been the country's capital since the 10th century and is home to the Eiffel Tower."

    print("Testing judge with one pair...")
    verdict = judge_pair(question, answer_a, answer_b)
    print(f"Question: {question}")
    print(f"Answer A: {answer_a}")
    print(f"Answer B: {answer_b}")
    print(f"Judge verdict: {verdict}")
