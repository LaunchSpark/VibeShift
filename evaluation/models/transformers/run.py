"""Pretrained transformer baselines (owned by Teammate C).

Two models, selected with ``--model``:
  * distilroberta  — j-hartmann/emotion-english-distilroberta-base
  * goemotions     — SamLowe/roberta-base-go_emotions (28 classes mapped to 7)

STUB STATE: ``predict_batch`` returns a uniform distribution so the full pipeline
(`pixi run eval-transformers` → `pixi run eval-report`) runs today on the sample.
Teammate C: replace the body with a real Hugging Face pipeline. `transformers`
and `pytorch` are already in the pixi env. Map each model's native labels onto
`evaluation.common.labels.LABELS`; `io.normalize_probs` handles the renormalize.

    pixi run eval-transformers              # defaults to --model distilroberta
    python -m evaluation.models.transformers.run --model goemotions
"""

from __future__ import annotations

import argparse

from evaluation.models import _common
from evaluation.common.labels import NEUTRAL

# distilroberta uses Ekman's 6 + neutral; goemotions' 28 classes collapse to 7.
# Teammate C: fill these in when wiring up the real models.
GOEMOTIONS_TO_7: dict[str, str] = {}


def predict_batch(texts: list[str]) -> list[_common.Prediction]:
    print("\033[33m[transformers] STUB — returning uniform probs. "
          "Teammate C: implement the real Hugging Face pipeline here.\033[0m")
    return [(_common.uniform(), NEUTRAL) for _ in texts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["distilroberta", "goemotions"], default="distilroberta")
    args = ap.parse_args()
    _common.run(f"transformers_{args.model}", predict_batch, cost_per_msg=0.0)


if __name__ == "__main__":
    main()
