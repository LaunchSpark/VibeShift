"""Train and evaluate the PPMI semantic-space model on combined W=5 data."""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.common.labels import LABELS
from evaluation.human_labeled_w5 import (
    DEFAULT_LABELS,
    DEFAULT_MESSAGES,
    SEED,
    build_grouped_split,
    read_jsonl,
    w5_text,
)
from evaluation.w5_model_comparison import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
MACHINE_CSV = ROOT / "data" / "synthetic_emotion_labels_w5.csv"
OUT = ROOT / "evaluation" / "results" / "combined_w5_20261126_semantic"
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+", re.I)


def _load() -> tuple[list[dict], list[dict], dict[str, str]]:
    human = [{**row, "source": "human"} for row in read_jsonl(DEFAULT_LABELS)]
    messages = read_jsonl(DEFAULT_MESSAGES)
    by_id = {row["id"]: row for row in messages}
    human_split = build_grouped_split(human, [by_id[row["id"]] for row in human])
    with MACHINE_CSV.open("r", newline="", encoding="utf-8") as stream:
        machine_source = list(csv.DictReader(stream))
    machine = [{"id": row["id"], "label": row["label"], "source": "machine"}
               for row in machine_source]
    machine_messages = [by_id[row["id"]] for row in machine]
    groups = [row["conversation_key"] for row in machine_messages]
    from sklearn.model_selection import StratifiedGroupKFold
    train_idx, test_idx = next(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    ).split(np.zeros(len(machine)), [row["label"] for row in machine], groups))
    split = dict(human_split)
    split.update({machine[i]["id"]: "train" for i in train_idx})
    split.update({machine[i]["id"]: "test" for i in test_idx})
    if {row["id"] for row in human} & {row["id"] for row in machine}:
        raise ValueError("human and machine label IDs overlap")
    if len(split) != len(human) + len(machine):
        raise ValueError("combined split is incomplete")
    conv_by_id = {row["id"]: by_id[row["id"]]["conversation_key"] for row in human + machine}
    train_convs = {conv_by_id[mid] for mid, part in split.items() if part == "train"}
    test_convs = {conv_by_id[mid] for mid, part in split.items() if part == "test"}
    if train_convs & test_convs:
        raise ValueError("conversation leakage between training and test data")
    human_convs = {conv_by_id[row["id"]] for row in human}
    machine_convs = {conv_by_id[row["id"]] for row in machine}
    if human_convs & machine_convs:
        raise ValueError("machine and human datasets share conversations")
    return human + machine, messages, split


def _calibration_rows(train_rows: list[dict]) -> list[dict]:
    calibration = []
    for source, offset in (("human", 11), ("machine", 29)):
        rng = random.Random(SEED + offset)
        source_rows = [row for row in train_rows if row["source"] == source]
        for label in LABELS:
            members = [row for row in source_rows if row["label"] == label]
            if members:
                count = min(len(members), max(1, round(0.10 * len(members))))
                calibration.extend(rng.sample(members, count))
    return calibration


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    started = time.perf_counter()
    labels, messages, split = _load()
    by_id = {row["id"]: row for row in messages}
    texts = {row["id"]: w5_text(by_id[row["id"]]) for row in labels}
    train_rows = [row for row in labels if split[row["id"]] == "train"]
    test_rows = [row for row in labels if split[row["id"]] == "test"]
    calibration = _calibration_rows(train_rows)
    evaluation_rows = calibration + test_rows
    evaluation_split = {row["id"]: "train" for row in calibration}
    evaluation_split.update({row["id"]: "test" for row in test_rows})

    tokenized = {mid: TOKEN.findall(text.lower()) for mid, text in texts.items()}
    train_ids = {row["id"] for row in train_rows}
    frequency = Counter(token for mid in train_ids for token in tokenized[mid])
    vocab = [token for token, count in frequency.most_common(800) if count >= 2]
    word_index = {token: i for i, token in enumerate(vocab)}
    co = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for mid in train_ids:
        sequence = [word_index[token] for token in tokenized[mid] if token in word_index]
        for pos, word in enumerate(sequence):
            for offset in range(1, 5):
                if pos + offset >= len(sequence):
                    break
                other = sequence[pos + offset]
                weight = 1.0 / offset
                co[word, other] += weight
                co[other, word] += weight

    total = float(co.sum())
    row_mass = co.sum(axis=1)
    col_mass = co.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(
            np.log(np.maximum(co * total, 1e-12) /
                   np.maximum(np.outer(row_mass, col_mass), 1e-12)), 0.0
        )
    np.fill_diagonal(ppmi, 0.0)
    vectors = ppmi / np.maximum(np.linalg.norm(ppmi, axis=1, keepdims=True), 1e-12)

    message_vectors = {}
    for mid, tokens in tokenized.items():
        ids = [word_index[token] for token in tokens if token in word_index]
        if ids:
            vector = vectors[ids].mean(axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
        else:
            vector = np.zeros(len(vocab), dtype=np.float32)
        message_vectors[mid] = vector.astype(np.float32, copy=False)

    prototypes = []
    for label in LABELS:
        ids = [row["id"] for row in train_rows if row["label"] == label]
        if not ids:
            raise ValueError(f"training split contains no {label} examples")
        centroid = np.stack([message_vectors[mid] for mid in ids]).mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        prototypes.append(centroid)
    prototypes = np.stack(prototypes).astype(np.float32)
    ids = [row["id"] for row in labels]
    all_vectors = np.stack([message_vectors[mid] for mid in ids])
    # Elementwise reductions avoid a Windows BLAS crash seen for this small
    # matrix multiply in the local environment.
    cosines = np.stack([np.sum(all_vectors * prototype, axis=1)
                        for prototype in prototypes], axis=1)
    logits = cosines / 0.08
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    predictions = []
    for message_id, probability in zip(ids, probabilities):
        probs = {label: float(probability[i]) for i, label in enumerate(LABELS)}
        predictions.append({
            "id": message_id,
            "model": "semantic_space_combined",
            "model_id": "w5-cooccurrence-ppmi-emotion-prototypes-combined",
            "model_revision": f"ppmi-{len(vocab)}d-combined-w5-v1",
            "probs": probs,
            "raw_top_label": max(LABELS, key=probs.__getitem__),
            "latency_ms": 0.0,
            "cost_usd": 0.0,
            "input_window_size": 5,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT / "predictions_semantic_space.jsonl", predictions)
    (OUT / "vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "semantic_space_model.npz", word_vectors=vectors, prototypes=prototypes)

    partitions = {
        "combined": [row for row in evaluation_rows],
        "human_only": [row for row in evaluation_rows if row["source"] == "human"],
        "machine_only": [row for row in evaluation_rows if row["source"] == "machine"],
    }
    metrics = {}
    prediction_by_id = {row["id"]: row for row in predictions}
    for name, partition_rows in partitions.items():
        sub_split = {row["id"]: evaluation_split[row["id"]] for row in partition_rows}
        sub_predictions = [prediction_by_id[row["id"]] for row in partition_rows]
        result = evaluate_model(
            "semantic_space_combined", partition_rows,
            [by_id[row["id"]] for row in partition_rows], sub_split,
            sub_predictions, 1000,
        )
        result.pop("tau_sweep", None)
        result.pop("test_confusion", None)
        metrics[name] = result

    split_record = {
        "train_human_rows": sum(row["source"] == "human" for row in train_rows),
        "test_human_rows": sum(row["source"] == "human" for row in test_rows),
        "train_machine_rows": sum(row["source"] == "machine" for row in train_rows),
        "test_machine_rows": sum(row["source"] == "machine" for row in test_rows),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "calibration_rows": len(calibration),
        "calibration_is_training_in_sample": True,
        "conversation_grouped_split_seed": SEED,
        "train_test_conversation_overlap": [],
        "vocabulary_fit_on_train_only": True,
        "prototypes_fit_on_train_only": True,
    }
    report = {
        "method": "W=5 token co-occurrence, PPMI word vectors, cosine similarity to seven emotion centroids",
        "protocol": split_record,
        "vocabulary_size": len(vocab),
        "dimensions": len(vocab),
        "fit_and_prediction_seconds": time.perf_counter() - started,
        "results": metrics,
        "machine_test_warning": "Machine labels were created by pretrained bot Laya; machine-only results are teacher-label agreement, not independent accuracy.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        "protocol": split_record,
        "vocabulary_size": len(vocab),
        "elapsed_seconds": report["fit_and_prediction_seconds"],
        "results": {
            name: {
                "tau": result["selected_tau"],
                "n": result["test"]["n"],
                "accuracy": result["test"]["accuracy"],
                "balanced_accuracy": result["test"]["balanced_accuracy"],
                "macro_f1": result["test"]["macro_f1"],
                "weighted_f1": result["test"]["weighted_f1"],
                "kappa": result["test"]["cohen_kappa"],
                "neutral_human_pct": result["test"]["human_neutral_pct"],
                "neutral_predicted_pct": result["test"]["predicted_neutral_pct"],
                "macro_f1_ci_95": result["test"]["cluster_bootstrap_95"]["intervals"]["macro_f1"],
                "per_class": result["test"]["per_class"],
            } for name, result in metrics.items()
        },
    }
    print("CHAT_ONLY_SEMANTIC_SPACE=" + json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
