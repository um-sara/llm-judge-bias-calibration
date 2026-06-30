"""
Core judge logic using a local Ollama model (default: llama3.2:3b).

Drop-in replacement for judge_claude.py — same function signature and return values.
Requires Ollama running locally: `ollama serve`
"""

import ollama
from judge_claude import JUDGE_PROMPT
from api_cache import cached_call


def judge_pair_ollama(question: str, answer_a: str, answer_b: str,
                      model: str = "llama3.2:3b") -> str:
    """
    Ask a local Ollama model to pick the better answer.
    Returns "A", "B", "tie", or "no answer".
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    def _call_api():
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    raw = cached_call(
        judge="ollama",
        model=model,
        prompt=prompt,
        max_tokens=0,
        temperature=0.0,
        call_fn=_call_api,
    )

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
    question = "What is the capital of France?"
    answer_a = "The capital of France is Paris."
    answer_b = "France's capital city is Paris. It has been the country's capital since the 10th century and is home to the Eiffel Tower."

    print("Testing local Ollama judge with one pair...")
    verdict = judge_pair_ollama(question, answer_a, answer_b)
    print(f"Question: {question}")
    print(f"Answer A: {answer_a}")
    print(f"Answer B: {answer_b}")
    print(f"Judge verdict: {verdict}")
