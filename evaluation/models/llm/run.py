"""Zero-shot LLM baseline (owned by Teammate C).

Prompts Google Gemini (gemini-2.0-flash) to return a JSON probability distribution
over the 7 emotion labels for each message.  Temperature 0 for deterministic output.
On a parse failure, retries once, then falls back to a uniform distribution and
counts the fallback.

Records per-message latency and estimated API cost (from token usage).

    pixi run eval-llm
"""

from __future__ import annotations

import json
import os
import re
import time

from evaluation.common import io
from evaluation.common.labels import LABELS, NEUTRAL
from evaluation.models import _common

MODEL_NAME = "llm_zeroshot"
LLM_MODEL_ID = "gemini-2.0-flash"

# Load the prompt template from the committed file.
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")

# Gemini pricing (per 1M tokens, as of Nov 2026 — gemini-2.0-flash).
_INPUT_COST_PER_TOKEN = 0.10 / 1_000_000   # $0.10 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 0.40 / 1_000_000  # $0.40 per 1M output tokens

_fallback_count = 0


def _load_prompt_template() -> str:
    with open(_PROMPT_PATH) as f:
        return f.read()


def _build_prompt(message: dict, template: str) -> str:
    """Format the prompt with the target message and its context window."""
    context_lines = []
    preceding = message.get("preceding_3", [])
    for i, ctx_msg in enumerate(preceding):
        context_lines.append(f"  [context {i + 1}]: {ctx_msg}")

    context_block = "\n".join(context_lines) if context_lines else "  (no preceding context)"
    return template.replace("{context}", context_block).replace("{message}", message["content"])


def _parse_probs(text: str) -> dict[str, float] | None:
    """Extract the JSON probability dict from the LLM response."""
    # Try to find a JSON object in the response.
    match = re.search(r"\{[^{}]+\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    # Validate keys are present (allow case-insensitive and extra keys).
    lower_map = {k.lower(): v for k, v in data.items()}
    probs = {}
    for label in LABELS:
        val = lower_map.get(label)
        if val is None:
            return None
        try:
            probs[label] = float(val)
        except (TypeError, ValueError):
            return None

    return probs


def _call_llm(prompt: str) -> tuple[str, int, int]:
    """Call the Gemini API. Returns (response_text, input_tokens, output_tokens).

    Uses the google-genai SDK if available, otherwise falls back to httpx.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=LLM_MODEL_ID,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return text, input_tokens, output_tokens
    except ImportError:
        pass

    # Fallback: direct REST call via httpx.
    import httpx

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL_ID}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    resp = httpx.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    usage = body.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    return text, input_tokens, output_tokens


def _predict_one(message: dict, template: str) -> tuple[_common.Prediction, float]:
    """Predict one message. Returns ((probs, raw_top), cost_usd)."""
    global _fallback_count
    prompt = _build_prompt(message, template)

    for attempt in range(2):  # try once, retry once on parse failure
        try:
            t0 = time.perf_counter()
            response_text, input_tokens, output_tokens = _call_llm(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            probs = _parse_probs(response_text)
            if probs is not None:
                raw_top = max(probs, key=probs.__getitem__)
                cost = (input_tokens * _INPUT_COST_PER_TOKEN +
                        output_tokens * _OUTPUT_COST_PER_TOKEN)
                return (probs, raw_top), cost
        except Exception:
            pass  # retry

    # Fallback to uniform after 2 failed attempts.
    _fallback_count += 1
    return (_common.uniform(), NEUTRAL), 0.0


def predict_batch(messages: list[dict]) -> list[_common.Prediction]:
    global _fallback_count
    _fallback_count = 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("\033[33m[llm] No GEMINI_API_KEY or GOOGLE_API_KEY set — "
              "returning uniform probs for all messages.\033[0m")
        return [(_common.uniform(), NEUTRAL) for _ in messages]

    template = _load_prompt_template()
    results: list[_common.Prediction] = []
    total_cost = 0.0

    for i, msg in enumerate(messages):
        pred, cost = _predict_one(msg, template)
        results.append(pred)
        total_cost += cost
        if (i + 1) % 50 == 0 or i == len(messages) - 1:
            print(f"  [{MODEL_NAME}] {i + 1}/{len(messages)} done "
                  f"(fallbacks so far: {_fallback_count})")

    print(f"[{MODEL_NAME}] finished — {len(messages)} messages, "
          f"{_fallback_count} fallbacks, ${total_cost:.4f} total cost")
    return results


def main() -> None:
    _common.run(MODEL_NAME, predict_batch, cost_per_msg=0.0)


if __name__ == "__main__":
    main()
