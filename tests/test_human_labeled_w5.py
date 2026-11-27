import json

import pytest

from evaluation.common.labels import LABELS
from evaluation.human_labeled_w5 import (
    _bootstrap_ci,
    _metrics,
    build_grouped_split,
    evaluate,
    gate,
    load_aligned,
    select_tau,
)


def test_w5_text_uses_last_four_preceding_messages_and_marks_target():
    from evaluation.human_labeled_w5 import w5_text

    text = w5_text({"id": "m", "content": "target", "preceding_25": ["older", "one", "two", "three", "four"]})
    assert "older" not in text
    assert all(word in text for word in ("one", "two", "three", "four", "TARGET MESSAGE: target"))


def test_w5_text_handles_short_context():
    from evaluation.human_labeled_w5 import w5_text

    assert w5_text({"id": "m", "content": "target", "preceding_25": []}) == "TARGET MESSAGE: target"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _probabilities(label):
    probs = {name: 0.01 for name in LABELS}
    probs[label] = 1 - 0.01 * (len(LABELS) - 1)
    return probs


def test_load_aligned_uses_stable_ids_and_validates_w5(tmp_path):
    messages_path = tmp_path / "messages.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(messages_path, [{"id": "c1:1", "conversation_key": "c1", "content": "hi", "split": "dev"}])
    _write_jsonl(labels_path, [{"id": "c1:1", "label": "joy", "labeler": "one",
                                "window_size": 5, "queue_position": 0}])

    labels, messages = load_aligned(labels_path, messages_path)

    assert [row["id"] for row in labels] == [row["id"] for row in messages] == ["c1:1"]


def test_load_aligned_rejects_unknown_or_duplicate_ids(tmp_path):
    messages_path = tmp_path / "messages.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(messages_path, [{"id": "c1:1", "conversation_key": "c1", "content": "hi", "split": "dev"}])
    row = {"id": "missing:1", "label": "joy", "window_size": 5, "queue_position": 0}
    _write_jsonl(labels_path, [row])
    with pytest.raises(ValueError, match="no matching message"):
        load_aligned(labels_path, messages_path)

    row["id"] = "c1:1"
    _write_jsonl(labels_path, [row, {**row, "queue_position": 1}])
    with pytest.raises(ValueError, match="duplicate human label id"):
        load_aligned(labels_path, messages_path)


def test_grouped_split_is_reproducible_and_keeps_conversations_together():
    labels, messages = [], []
    for group_index in range(10):
        for message_index in range(2):
            mid = f"c{group_index}:{message_index}"
            labels.append({"id": mid, "label": "neutral" if group_index % 2 else "joy"})
            messages.append({"id": mid, "conversation_key": f"c{group_index}"})

    first = build_grouped_split(labels, messages, seed=19)
    second = build_grouped_split(labels, messages, seed=19)

    assert first == second
    train_groups = {m["conversation_key"] for m in messages if first[m["id"]] == "train"}
    test_groups = {m["conversation_key"] for m in messages if first[m["id"]] == "test"}
    assert not train_groups & test_groups
    assert len([x for x in first.values() if x == "test"]) == 4


def test_tau_gate_matches_repository_rule_and_boundary():
    probs = {"joy": 0.4, "sadness": 0.1, "anger": 0.1, "fear": 0.1,
             "disgust": 0.05, "surprise": 0.05, "neutral": 0.2}
    assert gate(probs, 0.0) == "joy"
    assert gate(probs, 0.400001) == "neutral"
    probs["neutral"] = 0.5
    assert gate(probs, 0.0) == "neutral"


def test_tau_selection_prioritizes_neutral_match_then_performance():
    rows = [
        {"tau": 0.2, "neutral_abs_difference_pct": 0.0, "balanced_accuracy": 0.5, "macro_f1": 0.5},
        {"tau": 0.3, "neutral_abs_difference_pct": 0.0, "balanced_accuracy": 0.7, "macro_f1": 0.6},
        {"tau": 0.4, "neutral_abs_difference_pct": 1.0, "balanced_accuracy": 0.99, "macro_f1": 0.99},
    ]
    assert select_tau(rows)["tau"] == 0.3


def test_metrics_include_all_schema_classes_and_cluster_bootstrap_is_seeded():
    y_true = ["joy", "joy", "neutral", "neutral"]
    y_pred = ["joy", "neutral", "neutral", "neutral"]
    stats = _metrics(y_true, y_pred)
    assert stats["n"] == 4
    assert set(stats["per_class"]) == set(LABELS)
    records = [("c1", y_true[0], y_pred[0]), ("c1", y_true[1], y_pred[1]),
               ("c2", y_true[2], y_pred[2]), ("c2", y_true[3], y_pred[3])]
    first = _bootstrap_ci(records, seed=7, iterations=50)
    second = _bootstrap_ci(records, seed=7, iterations=50)
    assert first == second
    assert first["unit"] == "conversation"
    assert first["n_groups"] == 2


def test_evaluate_writes_machine_readable_report_and_figures(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    labels, messages, predictions = [], [], []
    for i, label in enumerate(LABELS * 10):
        mid = f"conversation-{i}:{i}"
        labels.append({"id": mid, "label": label, "labeler": "test", "window_size": 5,
                       "queue_position": i})
        messages.append({"id": mid, "conversation_key": f"conversation-{i}",
                         "content": "fixture message", "split": "source"})
        predictions.append({"id": mid, "model": "distilroberta", "model_revision": "abc123",
                            "probs": _probabilities(label), "raw_top_label": label,
                            "latency_ms": 1.0, "cost_usd": 0.0})
    _write_jsonl(labels_path, labels)
    _write_jsonl(messages_path, messages)
    _write_jsonl(predictions_path, predictions)
    out = tmp_path / "results"

    results = evaluate(labels_path, messages_path, predictions_path, out,
                       bootstrap_iterations=5)

    assert results["split"]["train_n"] + results["split"]["test_n"] == 70
    assert results["test"]["tau_selected"]["accuracy"] == 1.0
    for name in ("evaluation_results.json", "evaluation_report.md", "tau_calibration_train.csv",
                 "confusion_test_raw.csv", "confusion_test_normalized.csv",
                 "tau_vs_accuracy.png", "tau_vs_neutral_ratio.png", "confusion_test.png",
                 "representative_examples.jsonl"):
        assert (out / name).is_file()
