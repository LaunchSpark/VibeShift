from __future__ import annotations

import pytest

from evaluation.common.labels import LABELS
from evaluation.w5_model_comparison import evaluate_model


def _probs(label: str) -> dict[str, float]:
    return {key: float(key == label) for key in LABELS}


def test_comparison_reports_common_holdout_metrics_and_cluster_ci():
    labels = [
        {"id": "a", "label": "neutral"},
        {"id": "b", "label": "joy"},
        {"id": "c", "label": "joy"},
        {"id": "d", "label": "neutral"},
    ]
    messages = [
        {"id": "a", "conversation_key": "c1"},
        {"id": "b", "conversation_key": "c2"},
        {"id": "c", "conversation_key": "c3"},
        {"id": "d", "conversation_key": "c4"},
    ]
    split = {"a": "train", "b": "train", "c": "test", "d": "test"}
    predictions = [
        {"id": "a", "probs": _probs("neutral"), "latency_ms": 1.0, "cost_usd": 0},
        {"id": "b", "probs": _probs("joy"), "latency_ms": 1.0, "cost_usd": 0},
        {"id": "c", "probs": _probs("joy"), "latency_ms": 1.0, "cost_usd": 0},
        {"id": "d", "probs": _probs("neutral"), "latency_ms": 1.0, "cost_usd": 0},
    ]
    result = evaluate_model("fixture", labels, messages, split, predictions, iterations=20)
    assert result["test"]["accuracy"] == 1.0
    assert result["test"]["macro_precision"] == 2 / 7
    assert result["test"]["normalized_confusion"]
    assert result["test"]["cluster_bootstrap_95"]["n_groups"] == 2


def test_comparison_rejects_misaligned_predictions():
    labels = [{"id": "a", "label": "neutral"}]
    messages = [{"id": "a", "conversation_key": "c1"}]
    with pytest.raises(ValueError, match="prediction IDs differ"):
        evaluate_model("fixture", labels, messages, {"a": "test"}, [], iterations=1)
