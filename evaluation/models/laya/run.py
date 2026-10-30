"""Laya runner (owned by Lucas) — the core model and a stacking input feature.

Calls the production scorer in ``src/vibeshift/emotions.py`` on every message and
writes ``evaluation/predictions/laya.jsonl``. Two consumers:

* ``eval/evaluate.py`` — Laya is the core model in the benchmark table.
* ``models/stacking/train.py`` — concatenates Laya's 7 probabilities with the
  NRC macro-window vector as the stacking head's input features.

This loads the real checkpoint (slow, network on first run): the 7-label GoEmotions
model pinned by ``MODEL_REVISION`` in ``emotions.py``, so its probabilities already
match the schema labels.

    pixi run eval-laya
"""

from __future__ import annotations

from evaluation.models import _common

MODEL_NAME = "laya"


def predict_batch(texts: list[str]) -> list[_common.Prediction]:
    from vibeshift.emotions import score_batch  # heavy import; keep it lazy

    scores = score_batch(texts)
    return [(s.probabilities, s.emotion) for s in scores]


def main() -> None:
    _common.run(MODEL_NAME, predict_batch, cost_per_msg=0.0)


if __name__ == "__main__":
    main()
