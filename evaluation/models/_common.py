"""Helpers shared by every model runner.

A runner implements ``predict_batch(texts) -> list[(probs, raw_top_label)]`` and
calls ``run(model_name, predict_batch)``. This module handles loading the
messages (real or sample), timing, assembling Contract-2 rows, and writing them.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from evaluation.common import io
from evaluation.common.labels import LABELS

# A runner returns, per message, (probability dict over LABELS, raw argmax label).
Prediction = tuple[dict[str, float], str]
PredictBatch = Callable[[list[str]], list[Prediction]]


def run(model_name: str, predict_batch: PredictBatch, *, cost_per_msg: float = 0.0) -> None:
    """Run a model over every message and write its prediction file."""
    messages = io.load_messages()
    if io.using_sample():
        print(f"[{model_name}] running on the SAMPLE fixture (real dataset not present yet).")

    texts = [m["content"] for m in messages]
    t0 = time.perf_counter()
    preds = predict_batch(texts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if len(preds) != len(messages):
        raise ValueError(f"{model_name} returned {len(preds)} preds for {len(messages)} messages")

    per_msg_ms = elapsed_ms / max(len(messages), 1)
    rows = []
    for msg, (probs, raw_top) in zip(messages, preds):
        rows.append(
            {
                "id": msg["id"],
                "model": model_name,
                "probs": io.normalize_probs(probs),
                "raw_top_label": raw_top,
                "latency_ms": round(per_msg_ms, 3),
                "cost_usd": cost_per_msg,
            }
        )
    io.write_predictions(model_name, rows)


def uniform() -> dict[str, float]:
    """A placeholder uniform distribution — used by not-yet-implemented stubs."""
    return {label: 1.0 / len(LABELS) for label in LABELS}
