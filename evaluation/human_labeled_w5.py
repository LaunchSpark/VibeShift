"""Evaluate DistilRoBERTa on the supplied W=5 human labels.

The supplied file has one annotator's labels. The model input is the target
message plus up to four preceding messages, matching the annotator's W=5
context. Outputs are kept separate from shared evaluation contracts.

Commands:
    pixi run eval-human-w5 --mode audit
    pixi run eval-human-w5 --mode predict
    pixi run eval-human-w5 --mode evaluate
    pixi run eval-human-w5 --mode all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from evaluation.common.labels import LABELS, NEUTRAL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "human_labeled_dataset" / "discord_messages_braylon-trail_w5.jsonl"
DEFAULT_MESSAGES = ROOT / "evaluation" / "data" / "messages.jsonl"
OUTPUT = ROOT / "evaluation" / "results" / "human_labeled_w5"
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
MODEL_NAME = "distilroberta"
WINDOW_SIZE = 5
SEED = 4240
LABEL_SET = set(LABELS)


def w5_text(message: dict[str, Any]) -> str:
    """Serialize up to four preceding messages plus a clearly marked target."""
    preceding = message.get("preceding_25", [])
    if not isinstance(preceding, list) or any(not isinstance(text, str) for text in preceding):
        raise ValueError(f"message {message.get('id')}: preceding_25 must be a list of strings")
    context = preceding[-(WINDOW_SIZE - 1):]
    earlier = "\n".join(f"Earlier message {i + 1}: {text}" for i, text in enumerate(context))
    return f"{earlier}\nTARGET MESSAGE: {message['content']}" if earlier else f"TARGET MESSAGE: {message['content']}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict JSONL; malformed nonblank rows fail with a line number."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each row must be a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_aligned(labels_path: Path, messages_path: Path) -> tuple[list[dict], list[dict]]:
    """Validate and align every supplied label to exactly one message by ID.

    Missing/invalid records and duplicate IDs are errors, not silently dropped.
    Messages without a human label are excluded and counted in the audit output.
    """
    labels = read_jsonl(labels_path)
    messages = read_jsonl(messages_path)
    by_id: dict[str, dict] = {}
    for row_no, msg in enumerate(messages, 1):
        mid = msg.get("id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"message row {row_no}: missing nonempty id")
        if mid in by_id:
            raise ValueError(f"duplicate message id: {mid}")
        if not isinstance(msg.get("conversation_key"), str) or not msg["conversation_key"]:
            raise ValueError(f"message {mid}: missing conversation_key")
        if not isinstance(msg.get("content"), str):
            raise ValueError(f"message {mid}: content must be a string")
        by_id[mid] = msg

    seen: set[str] = set()
    aligned_labels, aligned_messages = [], []
    for row_no, label in enumerate(labels, 1):
        mid = label.get("id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"label row {row_no}: missing nonempty id")
        if mid in seen:
            raise ValueError(f"duplicate human label id: {mid}")
        seen.add(mid)
        if mid not in by_id:
            raise ValueError(f"human label id {mid!r} has no matching message")
        if label.get("label") not in LABEL_SET:
            raise ValueError(f"label {mid}: invalid category {label.get('label')!r}")
        if label.get("window_size") != WINDOW_SIZE:
            raise ValueError(
                f"label {mid}: expected W={WINDOW_SIZE}, got {label.get('window_size')!r}"
            )
        if not isinstance(label.get("queue_position"), int):
            raise ValueError(f"label {mid}: queue_position must be an integer")
        if not isinstance(by_id[mid].get("split"), str):
            raise ValueError(f"message {mid}: missing preassigned source split")
        aligned_labels.append(label)
        aligned_messages.append(by_id[mid])
    if not aligned_labels:
        raise ValueError("no human-labeled rows were found")
    if len({r["queue_position"] for r in aligned_labels}) != len(aligned_labels):
        raise ValueError("duplicate queue_position in human labels")
    return aligned_labels, aligned_messages


def build_grouped_split(
    labels: list[dict], messages: list[dict], seed: int = SEED
) -> dict[str, str]:
    """Build a deterministic, class-stratified 80/20 split by conversation.

    A five-fold StratifiedGroupKFold supplies one approximately 20% holdout.
    Its first fold is fixed by seed; no performance metric or model prediction
    is used to choose the fold.  Whole conversations stay in one partition.
    """
    if len(labels) != len(messages):
        raise ValueError("labels and messages must have identical lengths")
    groups = [m["conversation_key"] for m in messages]
    if len(set(groups)) < 5:
        raise ValueError("at least five conversations are needed for an 80/20 group split")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)),
                                               [r["label"] for r in labels], groups))
    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    if train_groups & test_groups:
        raise AssertionError("conversation leakage across train/test split")
    result = {labels[i]["id"]: "train" for i in train_idx}
    result.update({labels[i]["id"]: "test" for i in test_idx})
    return result


def audit(labels_path: Path, messages_path: Path, output: Path = OUTPUT) -> dict:
    labels, messages = load_aligned(labels_path, messages_path)
    split_by_id = build_grouped_split(labels, messages)
    all_counts = Counter(r["label"] for r in labels)
    partitions = {}
    for part in ("train", "test"):
        selected = [r for r in labels if split_by_id[r["id"]] == part]
        selected_messages = [m for m in messages if split_by_id[m["id"]] == part]
        partitions[part] = {
            "n": len(selected),
            "conversations": len({m["conversation_key"] for m in selected_messages}),
            "labels": {label: sum(r["label"] == label for r in selected) for label in LABELS},
            "neutral_ratio": sum(r["label"] == NEUTRAL for r in selected) / len(selected),
        }
    audit_data = {
        "experiment": "human_labeled_w5",
        "human_label_file": str(labels_path.relative_to(ROOT)),
        "message_file": str(messages_path.relative_to(ROOT)),
        "human_label_sha256": _sha256(labels_path),
        "message_file_sha256": _sha256(messages_path),
        "model_id": MODEL_ID,
        "model_revision": None,
        "window_size": WINDOW_SIZE,
        "human_labelers": sorted({r.get("labeler", "unknown") for r in labels}),
        "n_labelers": len({r.get("labeler", "unknown") for r in labels}),
        "n_labeled": len(labels),
        "n_message_rows": len(read_jsonl(messages_path)),
        "n_unlabeled_messages_excluded": len(read_jsonl(messages_path)) - len(labels),
        "n_conversations": len({m["conversation_key"] for m in messages}),
        "all_label_counts": {label: all_counts[label] for label in LABELS},
        "split_method": "first fold of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4240)",
        "seed": SEED,
        "partitions": partitions,
        "input_handling": {
            "incomplete_human_windows": "Included as annotated; at conversation starts the existing queue supplies fewer than five messages.",
            "model_input": "Target plus up to four preceding messages from preceding_25; target explicitly marked.",
            "invalid_records": "Fail fast with row/id context; do not silently discard malformed rows or invalid labels.",
            "missing_human_labels": "Messages without a row in the supplied human label file are excluded and counted above.",
            "conversation_boundaries": "Use precomputed message rows and group by conversation_key; no context is joined across conversations.",
        },
        "known_limitations": [
            "Human ground truth is from one annotator, so inter-annotator agreement cannot be estimated.",
            "The supplied label file contains W=5 annotations; this experiment intentionally differs from GitHub issue #9's W=1/W=4 consensus plan.",
            "The holdout is only about one fifth of 51 labeled conversations; rare-class estimates and their confidence intervals are uncertain.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_audit.json").write_text(json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")
    with (output / "split.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(labels, key=lambda x: x["queue_position"]):
            f.write(json.dumps({"id": r["id"], "queue_position": r["queue_position"],
                                "split": split_by_id[r["id"]]}, ensure_ascii=False) + "\n")
    return audit_data


def _prediction_label(probs: dict[str, float]) -> str:
    return max(LABELS, key=lambda label: probs[label])


def _load_pipeline(revision: str | None = None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import pipeline

    requested_revision = revision or "main"
    try:
        model_path = snapshot_download(MODEL_ID, revision=requested_revision, local_files_only=True)
    except LocalEntryNotFoundError:
        model_path = snapshot_download(MODEL_ID, revision=requested_revision)
    resolved_revision = Path(model_path).name
    classifier = pipeline("text-classification", model=model_path, top_k=None,
                          truncation=True, max_length=512)
    return classifier, resolved_revision


def predict(labels_path: Path, messages_path: Path, output: Path = OUTPUT,
            batch_size: int = 8, revision: str | None = None) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    labels, messages = load_aligned(labels_path, messages_path)
    output.mkdir(parents=True, exist_ok=True)
    classifier, resolved_revision = _load_pipeline(revision)

    predictions: list[dict] = []
    for start in range(0, len(labels), batch_size):
        batch_labels = labels[start : start + batch_size]
        batch_messages = messages[start : start + batch_size]
        texts = [w5_text(m) for m in batch_messages]
        t0 = time.perf_counter()
        raw = classifier(texts, batch_size=batch_size)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if len(raw) != len(batch_labels):
            raise ValueError(f"model returned {len(raw)} rows for {len(batch_labels)} inputs")
        for label_row, result in zip(batch_labels, raw):
            probs = {str(item["label"]).lower(): float(item["score"]) for item in result}
            if set(probs) != LABEL_SET:
                raise ValueError(f"model labels {sorted(probs)} do not match the seven-label schema")
            total = sum(probs.values())
            if not math.isfinite(total) or abs(total - 1.0) > 1e-3:
                raise ValueError(f"model probability sum for {label_row['id']} is {total}")
            predictions.append({
                "id": label_row["id"],
                "model": MODEL_NAME,
                "model_id": MODEL_ID,
                "model_revision": resolved_revision,
                "input_window_size": WINDOW_SIZE,
                "probs": probs,
                "raw_top_label": _prediction_label(probs),
                "latency_ms": round(elapsed_ms / len(batch_labels), 4),
                "cost_usd": 0.0,
            })
    if {r["id"] for r in predictions} != {r["id"] for r in labels}:
        raise ValueError("prediction ids do not exactly match the human-labeled ids")
    pred_path = output / "predictions_distilroberta_w5.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata_path = output / "model_run.json"
    metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": resolved_revision,
        "requested_revision": revision or "main at run time",
        "batch_size": batch_size,
        "max_length": 512,
        "input": "target plus up to four preceding messages; target marked; tokenizer truncates at max_length",
        "n_predictions": len(predictions),
        "prediction_sha256": _sha256(pred_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    audit_path = output / "dataset_audit.json"
    if audit_path.exists():
        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_data["model_revision"] = resolved_revision
        audit_path.write_text(json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")
    return pred_path


def gate(probs: dict[str, float], tau: float) -> str:
    raw_top = _prediction_label(probs)
    if raw_top == NEUTRAL:
        return NEUTRAL
    strongest_emotion = max(probs[label] for label in LABELS if label != NEUTRAL)
    return NEUTRAL if strongest_emotion < tau else raw_top


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    per_class = {
        label: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(LABELS)
    }
    counts = Counter(y_pred)
    n = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    observed = float(np.trace(cm) / n)
    true_margins = cm.sum(axis=1) / n
    pred_margins = cm.sum(axis=0) / n
    expected = float(np.sum(true_margins * pred_margins))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean([recall[i] for i, count in enumerate(support) if count > 0])),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "cohen_kappa": float(kappa),
        "human_neutral_pct": 100 * sum(x == NEUTRAL for x in y_true) / n,
        "predicted_neutral_pct": 100 * counts[NEUTRAL] / n,
        "predicted_distribution_pct": {label: 100 * counts[label] / n for label in LABELS},
        "per_class": per_class,
        "confusion": cm.tolist(),
    }


def _probability_metrics(y_true: list[str], probabilities: list[dict[str, float]]) -> dict:
    """Metrics on the stored, un-thresholded probability vectors."""
    top2 = 0
    brier_total = 0.0
    log_loss_total = 0.0
    for truth, probs in zip(y_true, probabilities):
        ranked = sorted(LABELS, key=lambda label: probs[label], reverse=True)
        top2 += truth in ranked[:2]
        brier_total += sum((probs[label] - float(label == truth)) ** 2 for label in LABELS)
        log_loss_total -= math.log(max(probs[truth], 1e-12))
    n = max(len(y_true), 1)
    return {"top2_accuracy": top2 / n, "brier_score": brier_total / n,
            "log_loss": log_loss_total / n}


def _bootstrap_ci(records: list[tuple[str, str, str]], seed: int, iterations: int) -> dict:
    """Percentile cluster bootstrap: resample conversations, retaining all rows."""
    rng = random.Random(seed)
    group_to_rows: dict[str, list[tuple[str, str]]] = {}
    for group, gold, pred in records:
        group_to_rows.setdefault(group, []).append((gold, pred))
    groups = sorted(group_to_rows)
    if len(groups) < 2:
        return {"unit": "conversation", "n_groups": len(groups), "iterations": 0, "intervals": {}}
    series: dict[str, list[float]] = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
    for _ in range(iterations):
        sample = [rng.choice(groups) for _ in groups]
        pairs = [pair for group in sample for pair in group_to_rows[group]]
        y_true = [p[0] for p in pairs]
        y_pred = [p[1] for p in pairs]
        stats = _metrics(y_true, y_pred)
        for key in series:
            if math.isfinite(stats[key]):
                series[key].append(stats[key])
    intervals = {}
    for key, values in series.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[key] = {"low_95": float(low), "high_95": float(high)}
    return {"unit": "conversation", "n_groups": len(groups), "iterations": iterations,
            "seed": seed, "intervals": intervals}


def _sweep(train: list[dict], prediction_by_id: dict[str, dict], split: dict[str, str]) -> list[dict]:
    train = [r for r in train if split[r["id"]] == "train"]
    y_true = [r["label"] for r in train]
    non_neutral_scores = [
        max(prediction_by_id[r["id"]]["probs"][label] for label in LABELS if label != NEUTRAL)
        for r in train
    ]
    # A 0.01 grid is broad; score breakpoints and their next representable values
    # capture every distinct gated-label state for this finite calibration set.
    taus = {i / 100 for i in range(101)}
    for score in non_neutral_scores:
        taus.add(float(score))
        above = math.nextafter(float(score), math.inf)
        if above <= 1:
            taus.add(above)
    human_neutral = 100 * sum(y == NEUTRAL for y in y_true) / len(y_true)
    table = []
    for tau in sorted(taus):
        y_pred = [gate(prediction_by_id[r["id"]]["probs"], tau) for r in train]
        metrics = _metrics(y_true, y_pred)
        row = {"tau": tau, **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "predicted_neutral_pct")},
               "human_neutral_pct": human_neutral,
               "neutral_abs_difference_pct": abs(metrics["predicted_neutral_pct"] - human_neutral)}
        row.update({f"predicted_{label}_pct": metrics["predicted_distribution_pct"][label] for label in LABELS})
        table.append(row)
    return table


def select_tau(table: list[dict]) -> dict:
    """Prioritize neutral-ratio match, then balanced accuracy, macro-F1, lower τ."""
    if not table:
        raise ValueError("cannot select tau from an empty sweep")
    return min(table, key=lambda row: (
        row["neutral_abs_difference_pct"], -row["balanced_accuracy"],
        -row["macro_f1"], row["tau"],
    ))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(sweep: list[dict], selected_tau: float, train_human_neutral_pct: float,
                  test_stats: dict, output: Path) -> None:
    import altair as alt
    import polars as pl

    data = pl.DataFrame(sweep)
    for column, title, ylabel, filename in (
        ("accuracy", "Calibration accuracy across tau", "Accuracy", "tau_vs_accuracy.png"),
        ("predicted_neutral_pct", "Calibration neutral ratio across tau", "Neutral messages (%)", "tau_vs_neutral_ratio.png"),
    ):
        base = alt.Chart(data).encode(
            x=alt.X("tau:Q", title="Neutral confidence cutoff (tau)"),
            y=alt.Y(f"{column}:Q", title=ylabel),
            tooltip=[alt.Tooltip("tau:Q", format=".4f"), alt.Tooltip(f"{column}:Q", format=".3f")],
        )
        target = (train_human_neutral_pct if column == "predicted_neutral_pct" else None)
        chart = base.mark_line(point=True, color="#2a78d6").properties(
            width=620, height=320, title=title
        )
        rule = alt.Chart(pl.DataFrame({"tau": [selected_tau]})).mark_rule(
            color="#eb6834", strokeDash=[5, 4]
        ).encode(x="tau:Q")
        layers = [chart, rule]
        if target is not None:
            target_df = pl.DataFrame({"pct": [target]})
            layers.append(alt.Chart(target_df).mark_rule(color="#1baf7a", strokeDash=[3, 3]).encode(y="pct:Q"))
        (alt.layer(*layers).configure_view(stroke=None).configure_axis(labelFont="Arial", titleFont="Arial")
         .save(str(output / filename), ppi=180))

    labels = list(LABELS)
    confusion = test_stats["confusion"]
    records = [{"human": labels[i], "predicted": labels[j], "count": confusion[i][j]}
               for i in range(len(labels)) for j in range(len(labels))]
    cm = pl.DataFrame(records)
    heat = alt.Chart(cm).mark_rect().encode(
        x=alt.X("predicted:N", sort=labels, title="Predicted label"),
        y=alt.Y("human:N", sort=labels, title="Human label"),
        color=alt.Color("count:Q", title="Messages"),
        tooltip=["human:N", "predicted:N", "count:Q"],
    ).properties(width=400, height=400, title=f"Held-out confusion matrix at tau={selected_tau:.3f}")
    heat.save(str(output / "confusion_test.png"), ppi=180)


def evaluate(labels_path: Path, messages_path: Path, predictions_path: Path,
             output: Path = OUTPUT, bootstrap_iterations: int = 1000,
             include_text_examples: bool = False) -> dict:
    labels, messages = load_aligned(labels_path, messages_path)
    split = build_grouped_split(labels, messages)
    preds = read_jsonl(predictions_path)
    by_id = {}
    for row in preds:
        mid = row.get("id")
        if mid in by_id:
            raise ValueError(f"duplicate prediction id: {mid}")
        probs = row.get("probs")
        if not isinstance(probs, dict) or set(probs) != LABEL_SET:
            raise ValueError(f"prediction {mid}: probabilities must contain exactly the seven labels")
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probs.values()):
            raise ValueError(f"prediction {mid}: invalid probability value")
        if abs(sum(probs.values()) - 1) > 1e-3:
            raise ValueError(f"prediction {mid}: probabilities do not sum to 1")
        by_id[mid] = row
    label_ids = {r["id"] for r in labels}
    if set(by_id) != label_ids:
        raise ValueError(f"prediction IDs differ: missing={len(label_ids-set(by_id))}, extra={len(set(by_id)-label_ids)}")

    output.mkdir(parents=True, exist_ok=True)
    sweep = _sweep(labels, by_id, split)
    _write_csv(output / "tau_calibration_train.csv", sweep)
    chosen = select_tau(sweep)
    train_labels = [r for r in labels if split[r["id"]] == "train"]
    test_labels = [r for r in labels if split[r["id"]] == "test"]
    y_train = [r["label"] for r in train_labels]
    train_tau0 = _metrics(y_train, [gate(by_id[r["id"]]["probs"], 0.0) for r in train_labels])
    train_selected = _metrics(y_train, [gate(by_id[r["id"]]["probs"], chosen["tau"]) for r in train_labels])
    y_test = [r["label"] for r in test_labels]
    test_results = {}
    test_records_for_ci = []
    message_by_id = {m["id"]: m for m in messages}
    for tau_name, tau in (("tau_0", 0.0), ("tau_selected", chosen["tau"])):
        y_pred = [gate(by_id[r["id"]]["probs"], tau) for r in test_labels]
        stats = _metrics(y_test, y_pred)
        stats.update(_probability_metrics(y_test, [by_id[r["id"]]["probs"] for r in test_labels]))
        stats["mean_latency_ms"] = sum(by_id[r["id"]].get("latency_ms", 0.0) for r in test_labels) / len(test_labels)
        stats["cost_usd_total"] = sum(by_id[r["id"]].get("cost_usd", 0.0) for r in test_labels)
        test_results[tau_name] = stats
        if tau_name == "tau_selected":
            test_records_for_ci = [
                (message_by_id[r["id"]]["conversation_key"], r["label"], pred)
                for r, pred in zip(test_labels, y_pred)
            ]
    selected_test = test_results["tau_selected"]
    selected_test["cluster_bootstrap_95"] = _bootstrap_ci(test_records_for_ci, SEED, bootstrap_iterations)
    majority = Counter(y_train).most_common(1)[0][0]
    majority_test = _metrics(y_test, [majority] * len(y_test))

    confusion_rows = []
    cm = selected_test["confusion"]
    for i, true_label in enumerate(LABELS):
        row_total = sum(cm[i])
        rec = {"human_label": true_label, **{f"pred_{label}": cm[i][j] for j, label in enumerate(LABELS)},
               "support": row_total}
        confusion_rows.append(rec)
    _write_csv(output / "confusion_test_raw.csv", confusion_rows)
    normalized_rows = []
    for i, true_label in enumerate(LABELS):
        row_total = sum(cm[i])
        normalized_rows.append({"human_label": true_label,
                                **{f"pred_{label}": cm[i][j] / row_total if row_total else 0.0
                                   for j, label in enumerate(LABELS)},
                                "support": row_total})
    _write_csv(output / "confusion_test_normalized.csv", normalized_rows)
    _write_csv(output / "classification_test_selected.csv", [
        {"label": label, **selected_test["per_class"][label]} for label in LABELS
    ])

    examples = []
    selected_tau = chosen["tau"]
    correct_examples = []
    for r in test_labels:
        prediction = by_id[r["id"]]
        pred_label = gate(prediction["probs"], selected_tau)
        item = {"id": r["id"], "human_label": r["label"], "predicted_label": pred_label,
                "raw_top_label": prediction["raw_top_label"],
                "max_probability": max(prediction["probs"].values()),
                "max_non_neutral_probability": max(prediction["probs"][l] for l in LABELS if l != NEUTRAL),
                "correct": pred_label == r["label"]}
        if include_text_examples:
            item["content_excerpt"] = message_by_id[r["id"]]["content"][:240]
        (correct_examples if item["correct"] else examples).append(item)
    with (output / "test_errors.jsonl").open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # At most five correct and five incorrect examples, favoring distinct gold labels.
    representative = []
    for pool in (correct_examples, examples):
        selected, used_labels = [], set()
        for label in LABELS:
            match = next((row for row in pool if row["human_label"] == label and row["id"] not in {x["id"] for x in selected}), None)
            if match is not None and label not in used_labels:
                selected.append(match)
                used_labels.add(label)
            if len(selected) == 5:
                break
        if len(selected) < 5:
            selected_ids = {row["id"] for row in selected}
            selected.extend(row for row in pool if row["id"] not in selected_ids and len(selected) < 5)
        representative.extend(selected[:5])
    with (output / "representative_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in representative:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = {
        "model_id": MODEL_ID,
        "model_revision": (preds[0].get("model_revision") if preds else None),
        "window_size": WINDOW_SIZE,
        "seed": SEED,
        "split": {"method": "StratifiedGroupKFold first fold, 5 folds", "train_n": len(train_labels),
                  "test_n": len(test_labels), "train_conversations": len({message_by_id[r["id"]]["conversation_key"] for r in train_labels}),
                  "test_conversations": len({message_by_id[r["id"]]["conversation_key"] for r in test_labels}),
                  "train_class_counts": dict(Counter(y_train)), "test_class_counts": dict(Counter(y_test))},
        "selection_rule": "Minimize absolute difference between predicted and human neutral percentages on train; ties go to higher balanced accuracy, then macro-F1, then lower tau.",
        "selected_tau": chosen["tau"],
        "calibration": {"human_neutral_pct": chosen["human_neutral_pct"],
                        "selected_predicted_neutral_pct": chosen["predicted_neutral_pct"],
                        "absolute_neutral_difference_pct": chosen["neutral_abs_difference_pct"],
                        "tau_0": train_tau0, "selected": train_selected},
        "test": test_results,
        "majority_class_baseline": {"label": majority, **majority_test},
        "bootstrap": "Conversation-cluster percentile bootstrap; test conversations, not individual messages, are resampled.",
        "error_count_at_selected_tau": len(examples),
        "representative_examples_count": len(representative),
        "limitations": ["One human annotator; no inter-annotator agreement estimate.",
                        "Only the last four available preceding messages and target are included; model tokenization may truncate long windows.",
                        "Rare labels may have very small held-out support; interpret per-class scores and intervals cautiously."],
    }
    (output / "evaluation_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _plot_outputs(sweep, chosen["tau"], chosen["human_neutral_pct"], selected_test, output)
    _write_report(results, output)
    return results


def _write_report(results: dict, output: Path) -> None:
    test = results["test"]
    ci = test["tau_selected"]["cluster_bootstrap_95"]["intervals"]
    lines = [
        "# W=5 Human-Labeled Evaluation",
        "",
        f"Model: `{results['model_id']}` (revision `{results.get('model_revision')}`).",
        "",
        "This is an evaluation of one annotator's labels. The annotator and classifier both receive the target plus up to four preceding messages (fewer at conversation start); the target is explicitly marked in model input.",
        "",
        f"Split: {results['split']['train_n']} training/calibration rows from {results['split']['train_conversations']} conversations; {results['split']['test_n']} held-out test rows from {results['split']['test_conversations']} conversations. Conversation groups do not cross partitions.",
        "",
        f"Selected tau: **{results['selected_tau']:.6f}**. Selection minimizes neutral-ratio difference on train; ties prefer higher balanced accuracy, then Macro-F1, then lower tau. Human neutral rate: {results['calibration']['human_neutral_pct']:.2f}%; model rate at selected tau: {results['calibration']['selected_predicted_neutral_pct']:.2f}%.",
        "",
        "## Held-out results",
        "",
              "| Setting | Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 | Top-2 | Brier | Log loss | Kappa | Human neutral | Predicted neutral | Mean ms/message | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting, label in (("tau_0", "tau = 0"), ("tau_selected", "selected tau")):
        m = test[setting]
        lines.append(f"| {label} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['top2_accuracy']:.3f} | {m['brier_score']:.3f} | {m['log_loss']:.3f} | {m['cohen_kappa']:.3f} | {m['human_neutral_pct']:.2f}% | {m['predicted_neutral_pct']:.2f}% | {m['mean_latency_ms']:.2f} | ${m['cost_usd_total']:.4f} |")
    lines += ["", "Majority-class baseline: **{}** (test accuracy {:.3f}).".format(
        results["majority_class_baseline"]["label"], results["majority_class_baseline"]["accuracy"]), "",
              "Conversation-cluster bootstrap 95% intervals at selected tau:", ""]
    for key, interval in ci.items():
        lines.append(f"- {key.replace('_', ' ')}: [{interval['low_95']:.3f}, {interval['high_95']:.3f}]")
    base = results["majority_class_baseline"]
    selected = test["tau_selected"]
    lines += ["", f"The selected tau improves held-out Macro-F1 over the neutral-only majority baseline ({selected['macro_f1']:.3f} vs {base['macro_f1']:.3f}) but has lower accuracy ({selected['accuracy']:.3f} vs {base['accuracy']:.3f}). This is the expected tradeoff from matching a minority-aware objective instead of predicting neutral for every message. Top-2, Brier, and log loss use the stored raw probabilities, so they do not change when tau changes.",
              "", "### Class counts by partition", "",
              "| Label | Train/calibration | Held-out test |", "|---|---:|---:|"]
    for label in LABELS:
        lines.append(f"| {label} | {results['split']['train_class_counts'].get(label, 0)} | {results['split']['test_class_counts'].get(label, 0)} |")
    confusion = selected["confusion"]
    error_pairs = sorted(
        ((confusion[i][j], LABELS[i], LABELS[j]) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j),
        reverse=True,
    )
    lines += ["", "### Largest class confusions at selected tau", ""]
    for count, human_label, predicted_label in [row for row in error_pairs if row[0]][:5]:
        lines.append(f"- Human **{human_label}** predicted as **{predicted_label}**: {count}")
    lines += ["", "## Per-class results at selected tau", "",
              "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        m = test["tau_selected"]["per_class"][label]
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "## Caveats", ""]
    lines.extend(f"- {text}" for text in results["limitations"])
    lines += ["", "The calibration curves and held-out confusion matrices are in this folder. `test_errors.jsonl` lists held-out mistakes, and `representative_examples.jsonl` includes correct and incorrect examples, by message ID. Message text is included only when the run uses `--include-text-examples`.", ""]
    (output / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("audit", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--revision", help="Hugging Face commit or branch; resolved commit is recorded")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--include-text-examples", action="store_true",
                        help="include short message excerpts in the local error file")
    args = parser.parse_args()
    if args.mode in ("audit", "all"):
        info = audit(args.labels, args.messages, args.output)
        print(f"audited {info['n_labeled']} labels; train={info['partitions']['train']['n']}, test={info['partitions']['test']['n']}")
    pred_path = args.output / "predictions_distilroberta_w5.jsonl"
    if args.mode in ("predict", "all"):
        pred_path = predict(args.labels, args.messages, args.output, args.batch_size, args.revision)
        print(f"wrote predictions to {pred_path}")
    if args.mode in ("evaluate", "all"):
        results = evaluate(args.labels, args.messages, pred_path, args.output,
                           args.bootstrap_iterations, args.include_text_examples)
        print(f"selected tau={results['selected_tau']:.6f}; report={args.output / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
