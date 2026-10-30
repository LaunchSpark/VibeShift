"""Zero-shot LLM baseline (owned by Teammate C).

Prompt an LLM to return a probability distribution over the 7 labels for each
message, then record latency and per-message API cost in the prediction file.

STUB STATE: ``predict_batch`` returns a uniform distribution so the pipeline runs
today. Teammate C: replace with real zero-shot calls. Ask the model for JSON with
exactly the `LABELS` keys; `io.normalize_probs` will validate and renormalize.
Set ``cost_per_msg`` from the provider's token pricing so the benchmark's cost
column is meaningful.

    pixi run eval-llm
"""

from __future__ import annotations

from evaluation.models import _common
from evaluation.common.labels import NEUTRAL

MODEL_NAME = "llm_zeroshot"


def predict_batch(texts: list[str]) -> list[_common.Prediction]:
    print("\033[33m[llm] STUB — returning uniform probs. "
          "Teammate C: implement real zero-shot LLM calls here.\033[0m")
    return [(_common.uniform(), NEUTRAL) for _ in texts]


def main() -> None:
    _common.run(MODEL_NAME, predict_batch, cost_per_msg=0.0)


if __name__ == "__main__":
    main()
