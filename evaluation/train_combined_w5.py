"""Train and compare models on combined human and Laya weak labels (W=5)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from evaluation.common.labels import LABELS, NEUTRAL
from evaluation.human_labeled_w5 import (
    DEFAULT_LABELS,
    DEFAULT_MESSAGES,
    SEED,
    build_grouped_split,
    read_jsonl,
    w5_text,
)
from evaluation.w5_model_comparison import evaluate_model
from vibeshift.emotions import CONTEXT_INSTRUCTIONS, EMOTION_CRITERIA, MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = ROOT / "data" / "synthetic_emotion_labels_w5.csv"
RUN_ROOT = ROOT / "operations" / "data" / "combined_w5_20261126"
RESULTS = ROOT / "evaluation" / "results" / "combined_w5_20261126"
BASELINE_PATH = ROOT / "evaluation" / "results" / "bot_laya_w5" / "predictions_bot_laya_w5.jsonl"
QUESTIONS = {"emotion_tracker": {
    "type": "choice",
    "instructions": CONTEXT_INSTRUCTIONS,
    "criteria": EMOTION_CRITERIA,
}}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_data() -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, str], dict[str, str]]:
    human_labels = read_jsonl(DEFAULT_LABELS)
    messages = read_jsonl(DEFAULT_MESSAGES)
    message_by_id = {str(row["id"]): row for row in messages}
    aligned_human_messages = [message_by_id[str(row["id"])] for row in human_labels]
    human_split = build_grouped_split(human_labels, aligned_human_messages)

    with SYNTHETIC_CSV.open("r", newline="", encoding="utf-8") as stream:
        machine_rows = list(csv.DictReader(stream))
    machine_labels = [{"id": row["id"], "label": row["label"], "source": "machine"}
                      for row in machine_rows]
    human_labels = [{**row, "source": "human"} for row in human_labels]
    machine_ids = {row["id"] for row in machine_labels}
    human_ids = {row["id"] for row in human_labels}
    if human_ids & machine_ids:
        raise ValueError("human and machine datasets contain duplicate message IDs")
    missing = (human_ids | machine_ids) - set(message_by_id)
    if missing:
        raise ValueError(f"missing W=5 message/context rows: {sorted(missing)[:5]}")
    if set(row["label"] for row in machine_labels) - set(LABELS):
        raise ValueError("machine CSV contains labels outside the seven-category schema")

    machine_messages = [message_by_id[row["id"]] for row in machine_labels]
    machine_groups = [row["conversation_key"] for row in machine_messages]
    from sklearn.model_selection import StratifiedGroupKFold
    machine_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(machine_splitter.split(
        np.zeros(len(machine_labels)), [row["label"] for row in machine_labels], machine_groups
    ))
    machine_split = {machine_labels[i]["id"]: "train" for i in train_idx}
    machine_split.update({machine_labels[i]["id"]: "test" for i in test_idx})

    all_labels = human_labels + machine_labels
    split = {**human_split, **machine_split}
    if set(split) != {row["id"] for row in all_labels}:
        raise ValueError("split is missing combined dataset rows")
    conversation_by_id = {mid: message_by_id[mid]["conversation_key"] for mid in split}
    train_conversations = {conversation_by_id[mid] for mid, part in split.items() if part == "train"}
    test_conversations = {conversation_by_id[mid] for mid, part in split.items() if part == "test"}
    if train_conversations & test_conversations:
        raise ValueError("conversation leakage in combined train/test split")
    if {message_by_id[row["id"]]["conversation_key"] for row in human_labels} & {
        message_by_id[row["id"]]["conversation_key"] for row in machine_labels
    }:
        raise ValueError("human and machine datasets reuse a conversation")

    return all_labels, messages, machine_rows, human_labels, split, conversation_by_id


def _sample_weights(labels: list[dict]) -> np.ndarray:
    """Balance both label classes and the two data sources for text classifiers."""
    counts = Counter(row["label"] for row in labels)
    source_counts = Counter(row["source"] for row in labels)
    n = len(labels)
    k = len(LABELS)
    weights = np.array([
        (n / (k * counts[row["label"]])) * (n / (2 * source_counts[row["source"]]))
        for row in labels
    ], dtype=np.float32)
    return weights / weights.mean()


def _probability_rows(name: str, ids: list[str], probs: np.ndarray,
                      latency_ms: float, model_id: str) -> list[dict]:
    rows = []
    for message_id, values in zip(ids, probs):
        vector = {label: 0.0 for label in LABELS}
        for i, label in enumerate(LABELS):
            vector[label] = float(values[i])
        total = sum(vector.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"{name}: invalid probabilities for {message_id}")
        vector = {label: value / total for label, value in vector.items()}
        rows.append({"id": message_id, "model": name, "model_id": model_id,
                     "model_revision": "combined-w5-20261126", "probs": vector,
                     "raw_top_label": max(LABELS, key=vector.__getitem__),
                     "latency_ms": latency_ms, "cost_usd": 0.0,
                     "input_window_size": 5})
    return rows


def _train_text_models(train_rows: list[dict], selected_rows: list[dict],
                       texts: dict[str, str], output: Path, epochs: int) -> tuple[dict[str, list[dict]], dict]:
    import joblib
    import torch
    from torch import nn
    from torch.nn import functional as F

    output.mkdir(parents=True, exist_ok=True)
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                                 max_features=20000, sublinear_tf=True)),
    ])
    x_train = vectorizer.fit_transform([texts[row["id"]] for row in train_rows])
    x_selected = vectorizer.transform([texts[row["id"]] for row in selected_rows])
    y_train = [row["label"] for row in train_rows]
    fit_weights = _sample_weights(train_rows)
    selected_ids = [row["id"] for row in selected_rows]
    predictions: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}
    joblib.dump(vectorizer, output / "tfidf_features.joblib")

    svm = LinearSVC(C=1.0, random_state=SEED)
    started = time.perf_counter()
    svm.fit(x_train, y_train, sample_weight=fit_weights)
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    decision = svm.decision_function(x_selected)
    decision = decision - decision.max(axis=1, keepdims=True)
    probability = np.exp(decision)
    probability /= probability.sum(axis=1, keepdims=True)
    timings["linear_svm"] = fit_time
    predictions["linear_svm"] = _probability_rows(
        "linear_svm", selected_ids, probability, (time.perf_counter() - started) * 1000 / len(selected_ids),
        "TFIDF + class/source-weighted LinearSVC",
    )
    joblib.dump(svm, output / "linear_svm.joblib")
    _write_jsonl(RESULTS / "predictions_linear_svm.jsonl", predictions["linear_svm"])
    print(f"Linear SVM trained in {fit_time:.1f}s", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    x_train_dense = x_train.toarray().astype(np.float32, copy=False)
    x_selected_dense = x_selected.toarray().astype(np.float32, copy=False)
    x_train_t = torch.tensor(x_train_dense, dtype=torch.float32, device=device)
    x_selected_t = torch.tensor(x_selected_dense, dtype=torch.float32, device=device)
    y_idx = torch.tensor([LABELS.index(label) for label in y_train], dtype=torch.long, device=device)
    weight_t = torch.tensor(fit_weights, dtype=torch.float32, device=device)
    models = {
        "logistic_regression": nn.Linear(x_train_t.shape[1], len(LABELS)).to(device),
        "neural_net_mlp": nn.Sequential(
            nn.Linear(x_train_t.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, len(LABELS)),
        ).to(device),
    }
    for name, net in models.items():
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        started = time.perf_counter()
        net.train()
        for epoch in range(epochs):
            order = torch.randperm(len(x_train_t), device=device)
            for start in range(0, len(order), 64):
                idx = order[start:start + 64]
                optimizer.zero_grad(set_to_none=True)
                losses = F.cross_entropy(net(x_train_t[idx]), y_idx[idx], reduction="none")
                loss = (losses * weight_t[idx]).mean()
                loss.backward()
                optimizer.step()
        fit_time = time.perf_counter() - started
        net.eval()
        started = time.perf_counter()
        with torch.inference_mode():
            probability = torch.cat([
                torch.softmax(net(x_selected_t[i:i + 64]), dim=1)
                for i in range(0, len(x_selected_t), 64)
            ]).cpu().numpy()
        timings[name] = fit_time
        architecture = "multinomial_logistic_linear_head" if name == "logistic_regression" else "128-unit MLP"
        predictions[name] = _probability_rows(
            name, selected_ids, probability,
            (time.perf_counter() - started) * 1000 / len(selected_ids),
            f"TFIDF + source/class-weighted {architecture}",
        )
        _write_jsonl(RESULTS / f"predictions_{name}.jsonl", predictions[name])
        torch.save({"state_dict": net.cpu().state_dict(), "input_dim": int(x_train.shape[1]),
                    "labels": LABELS, "epochs": epochs, "architecture": architecture},
                   output / f"{name}.pt")
        print(f"{name} trained in {fit_time:.1f}s on {device}", flush=True)
        del net
    del x_train_t, x_selected_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return predictions, {"fit_seconds": timings, "torch_device": device,
                         "features": int(x_train.shape[1]),
                         "feature_fit_rows": len(train_rows),
                         "source_class_sample_weighting": True}


def _baseline_rows(human_labels: list[dict], machine_rows: list[dict],
                   selected_ids: set[str]) -> list[dict]:
    human = {row["id"]: row for row in read_jsonl(BASELINE_PATH)}
    machine = {row["id"]: row for row in machine_rows}
    rows = []
    for message_id in selected_ids:
        if message_id in human:
            source = human[message_id]
            probs = source["probs"]
            latency = float(source.get("latency_ms", 0.0))
        else:
            source = machine.get(message_id)
            if source is None:
                raise ValueError(f"missing pretrained bot probabilities for {message_id}")
            probs = {label: float(source[f"prob_{label}"]) for label in LABELS}
            latency = 0.0
        rows.append({"id": message_id, "model": "pretrained_bot_laya",
                     "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                     "probs": probs, "raw_top_label": max(LABELS, key=probs.__getitem__),
                     "latency_ms": latency, "cost_usd": 0.0, "input_window_size": 5})
    return rows


def _train_laya(train_rows: list[dict], selected_rows: list[dict], texts: dict[str, str],
                output: Path, epochs: int, batch_size: int,
                inference_batch_size: int,
                prediction_path: Path | None = None) -> tuple[list[dict], dict]:
    import laya
    import laya.train
    import torch
    from huggingface_hub import snapshot_download

    checkpoint_dir = output / "learned_behavior_laya"
    train_path = output / "combined_train_w5.jsonl"
    train_records = [{"state": {"text": texts[row["id"]]}, "questions": QUESTIONS,
                      "expected": {"emotion_tracker": row["label"]}}
                     for row in train_rows]
    _write_jsonl(train_path, train_records)
    summary_path = output / "training_summary.json"
    if not (checkpoint_dir / "questions.json").exists():
        base_path = snapshot_download("LaunchSpark/laya-emotion-tracker", revision=MODEL_REVISION)
        config = laya.train.TrainConfig(
            epochs=epochs, micro_batch=batch_size, grad_accum=max(1, 64 // batch_size),
            encoder_lr=3e-5, loss="soft-ce", freeze_encoder=True,
            calib_frac=0.0, calib_seed=SEED, seed=SEED, log_every=10,
        )
        print(f"Fine-tuning Laya on all {len(train_rows)} combined training rows for {epochs} epochs", flush=True)
        summary = laya.train.finetune(
            data=str(train_path), model_dir=base_path, output_dir=str(checkpoint_dir),
            config=config, device="auto",
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print(f"Reusing completed combined Laya checkpoint {checkpoint_dir}", flush=True)

    agent = laya.Agent(str(checkpoint_dir))
    pred_path = prediction_path or RESULTS / "predictions_learned_behavior_laya_calibration_test.jsonl"
    expected_ids = {row["id"] for row in selected_rows}
    existing = {row["id"]: row for row in read_jsonl(pred_path)} if pred_path.exists() else {}
    if set(existing) - expected_ids:
        raise ValueError("partial Laya predictions contain unexpected IDs")
    rows = list(existing.values())
    todo = [row for row in selected_rows if row["id"] not in existing]
    for start in range(0, len(todo), inference_batch_size):
        batch = todo[start:start + inference_batch_size]
        started = time.perf_counter()
        results = agent.decide_batch([texts[row["id"]] for row in batch],
                                     questions=QUESTIONS, return_details=True)
        elapsed = time.perf_counter() - started
        for label_row, result in zip(batch, results):
            values = result.values["emotion_tracker"]["probabilities"]
            probs = {label: float(values.get(label, 0.0)) for label in LABELS}
            total = sum(probs.values())
            if total <= 0:
                raise ValueError(f"learned Laya returned empty probabilities for {label_row['id']}")
            probs = {label: value / total for label, value in probs.items()}
            rows.append({"id": label_row["id"], "model": "learned_behavior_laya",
                         "model_id": MODEL_ID + "+combined-w5-supervised",
                         "model_revision": "local-combined-w5-20261126",
                         "probs": probs, "raw_top_label": max(LABELS, key=probs.__getitem__),
                         "latency_ms": elapsed * 1000 / len(batch), "cost_usd": 0.0,
                         "input_window_size": 5})
        _write_jsonl(pred_path, rows)
        print(f"Learned Laya scored {min(start + len(batch), len(todo))}/{len(todo)} selected rows",
              flush=True)
    del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "checkpoint": checkpoint_dir.relative_to(ROOT).as_posix(), "epochs": epochs,
                  "gradient_training_rows": len(train_rows), "calibration_fraction": 0.0,
                  "freeze_encoder": True, "loss": "soft-ce",
                  "training_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.inference_batch_size < 1 or args.mlp_epochs < 1:
        parser.error("epoch and batch-size values must be positive")

    all_labels, messages, machine_rows, human_labels, split, conversation_by_id = _load_data()
    message_by_id = {row["id"]: row for row in messages}
    texts = {row["id"]: w5_text(message_by_id[row["id"]]) for row in all_labels}
    train_rows = [row for row in all_labels if split[row["id"]] == "train"]
    test_rows = [row for row in all_labels if split[row["id"]] == "test"]
    # Preserve the full 80% training partition for fitting. Score a deterministic,
    # stratified 10% slice of it to select tau, plus every held-out test row.
    calibration_rows = []
    for source, seed_offset in (("human", 11), ("machine", 29)):
        source_rows = [row for row in train_rows if row["source"] == source]
        rng = random.Random(SEED + seed_offset)
        for label in LABELS:
            class_rows = [row for row in source_rows if row["label"] == label]
            if class_rows:
                count = min(len(class_rows), max(1, round(len(class_rows) * 0.10)))
                calibration_rows.extend(rng.sample(class_rows, count))
    evaluation_rows = calibration_rows + test_rows
    selected_rows = evaluation_rows
    evaluation_split = {row["id"]: "train" for row in calibration_rows}
    evaluation_split.update({row["id"]: "test" for row in test_rows})
    human_train = [row for row in human_labels if split[row["id"]] == "train"]
    human_test = [row for row in human_labels if split[row["id"]] == "test"]
    machine_labels = [row for row in all_labels if row["source"] == "machine"]
    machine_train = [row for row in machine_labels if split[row["id"]] == "train"]
    machine_test = [row for row in machine_labels if split[row["id"]] == "test"]

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    split_record = {
        "seed": SEED,
        "human_split": "StratifiedGroupKFold 5-fold, first fold; group=conversation",
        "machine_split": "StratifiedGroupKFold 5-fold, first fold; group=conversation",
        "train_test_conversation_overlap": sorted(
            {conversation_by_id[mid] for mid, value in split.items() if value == "train"}
            & {conversation_by_id[mid] for mid, value in split.items() if value == "test"}
        ),
        "human_train_rows": len(human_train), "human_test_rows": len(human_test),
        "machine_train_rows": len(machine_train), "machine_test_rows": len(machine_test),
        "tau_calibration_rows": len(calibration_rows),
        "tau_calibration_human_rows": sum(row["source"] == "human" for row in calibration_rows),
        "tau_calibration_machine_rows": sum(row["source"] == "machine" for row in calibration_rows),
        "tau_calibration_note": "stratified sample from training conversations; all 80% train rows were used for fitting",
        "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "test_label_counts": dict(Counter(row["label"] for row in test_rows)),
        "machine_label_counts_test": dict(Counter(row["label"] for row in machine_test)),
        "human_test_label_counts": dict(Counter(row["label"] for row in human_test)),
        "human_machine_conversation_overlap": [],
    }
    (RESULTS / "split.json").write_text(json.dumps(split_record, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(RUN_ROOT / "combined_train_w5.jsonl", [
        {"id": row["id"], "source": row["source"], "label": row["label"],
         "text": texts[row["id"]]} for row in train_rows
    ])
    _write_jsonl(RUN_ROOT / "combined_test_w5.jsonl", [
        {"id": row["id"], "source": row["source"], "label": row["label"],
         "text": texts[row["id"]]} for row in test_rows
    ])
    print(json.dumps({"split": split_record}, indent=2), flush=True)

    text_predictions, text_metadata = _train_text_models(
        train_rows, selected_rows, texts, RUN_ROOT / "models", args.mlp_epochs
    )

    baseline = _baseline_rows(human_labels, machine_rows, {row["id"] for row in selected_rows})
    _write_jsonl(RESULTS / "predictions_pretrained_bot_laya.jsonl", baseline)
    laya_predictions, laya_metadata = _train_laya(
        train_rows, selected_rows, texts, RUN_ROOT / "laya", args.epochs,
        args.batch_size, args.inference_batch_size
    )

    models = {"pretrained_bot_laya": baseline, **text_predictions,
              "learned_behavior_laya": laya_predictions}
    metadata = {"pretrained_bot_laya": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
                **{name: {"model_id": name, **text_metadata} for name in text_predictions},
                "learned_behavior_laya": laya_metadata}

    partitions = {
        "combined": (evaluation_rows, evaluation_split),
        "human_only": ([row for row in evaluation_rows if row["source"] == "human"],
                       {row["id"]: evaluation_split[row["id"]] for row in evaluation_rows if row["source"] == "human"}),
        "machine_only": ([row for row in evaluation_rows if row["source"] == "machine"],
                         {row["id"]: evaluation_split[row["id"]] for row in evaluation_rows if row["source"] == "machine"}),
    }
    report: dict[str, Any] = {
        "protocol": split_record,
        "weak_label_warning": (
            "Machine-held-out labels were generated by the same pretrained bot Laya checkpoint. "
            "Machine-only metrics measure agreement with pseudo-labels, not independent accuracy. "
            "Human-only held-out metrics are the primary quality estimate."
        ),
        "model_metadata": metadata,
        "results": {},
    }
    for model_name, prediction_rows in models.items():
        prediction_by_id = {row["id"]: row for row in prediction_rows}
        report["results"][model_name] = {}
        for partition_name, (partition_labels, partition_split) in partitions.items():
            subset_ids = {row["id"] for row in partition_labels}
            subset_predictions = [prediction_by_id[mid] for mid in subset_ids]
            result = evaluate_model(
                model_name, partition_labels, [message_by_id[mid] for mid in subset_ids],
                partition_split, subset_predictions, args.bootstrap_iterations,
            )
            result.pop("tau_sweep", None)
            result.pop("test_confusion", None)
            report["results"][model_name][partition_name] = result
        print(f"Evaluated {model_name}", flush=True)

    report["run"] = {
        "run_directory": RUN_ROOT.relative_to(ROOT).as_posix(),
        "results_directory": RESULTS.relative_to(ROOT).as_posix(),
        "human_training_rows": len(human_train),
        "machine_training_rows": len(machine_train),
        "combined_training_rows": len(train_rows),
        "combined_test_rows": len(test_rows),
        "laya_epochs": args.epochs,
        "mlp_epochs": args.mlp_epochs,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    print("CHAT_ONLY_COMBINED_W5_REPORT=" + json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
