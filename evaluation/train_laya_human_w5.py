"""Fine-tune the bot's Laya path on the W=5 train split and evaluate held-out chats.

The conversation-grouped split is shared with ``evaluation.w5_model_comparison``.
All training/calibration rows come from its 80% train partition; the 20% test
partition is scored only after training has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from evaluation.common.labels import LABELS
from evaluation.human_labeled_w5 import (
    DEFAULT_LABELS,
    DEFAULT_MESSAGES,
    SEED,
    build_grouped_split,
    load_aligned,
    read_jsonl,
    w5_text,
)
from evaluation.w5_model_comparison import evaluate_model, write_report
from vibeshift.emotions import (
    CONTEXT_INSTRUCTIONS,
    EMOTION_CRITERIA,
    MODEL_ID,
    MODEL_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "operations" / "data" / "laya_human_w5"
BASELINE_RESULTS = ROOT / "evaluation" / "results" / "bot_laya_w5"
DEFAULT_OUTPUT = RUN_DIR / "results"
TRAIN_MODEL_ID = MODEL_ID.replace("+w5-context-v1", "+w5-supervised-v1")
MODEL_NAME = "learned_laya"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_artifacts(result: dict, output: Path) -> None:
    model = result["model"]
    with (output / f"tau_sweep_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["tau_sweep"][0]))
        writer.writeheader()
        writer.writerows(result["tau_sweep"])
    for suffix, matrix in (("raw", result["test_confusion"]),
                           ("normalized", result["test"]["normalized_confusion"])):
        with (output / f"confusion_{model}_{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["human_label", *[f"pred_{label}" for label in LABELS]]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for label, row in zip(LABELS, matrix):
                writer.writerow({"human_label": label,
                                 **{f"pred_{name}": value for name, value in zip(LABELS, row)}})
    with (output / f"classification_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["label", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in LABELS:
            writer.writerow({"label": label, **result["test_classification"][label]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    labels, messages = load_aligned(args.labels, args.messages)
    split = build_grouped_split(labels, messages)
    message_by_id = {message["id"]: message for message in messages}
    train_labels = [row for row in labels if split[row["id"]] == "train"]
    test_labels = [row for row in labels if split[row["id"]] == "test"]
    train_conversations = {message_by_id[row["id"]]["conversation_key"] for row in train_labels}
    test_conversations = {message_by_id[row["id"]]["conversation_key"] for row in test_labels}
    if train_conversations & test_conversations:
        raise ValueError("conversation leakage between train and held-out test partitions")

    import laya
    import laya.train
    from huggingface_hub import snapshot_download

    args.run_dir.mkdir(parents=True, exist_ok=True)
    questions = {"emotion_tracker": {
        "type": "choice",
        "instructions": CONTEXT_INSTRUCTIONS,
        "criteria": EMOTION_CRITERIA,
    }}
    train_file = args.run_dir / "train_w5.jsonl"
    _write_jsonl(train_file, [
        {"state": {"text": w5_text(message_by_id[row["id"]])},
         "questions": questions,
         "expected": {"emotion_tracker": row["label"]}}
        for row in train_labels
    ])

    checkpoint_dir = args.run_dir / "learned_behavior_laya"
    summary_file = args.run_dir / "training_summary.json"
    if not (checkpoint_dir / "questions.json").exists():
        base_dir = snapshot_download("LaunchSpark/laya-emotion-tracker", revision=MODEL_REVISION)
        config = laya.train.TrainConfig(
            epochs=args.epochs,
            micro_batch=args.batch_size,
            grad_accum=max(1, 64 // args.batch_size),
            encoder_lr=3e-5,
            loss="soft-ce",
            freeze_encoder=True,
            calib_seed=SEED,
            seed=SEED,
            log_every=10,
        )
        print(f"Training on {len(train_labels)} W=5 labels; held-out rows: {len(test_labels)}", flush=True)
        print(f"Base checkpoint: LaunchSpark/laya-emotion-tracker@{MODEL_REVISION}", flush=True)
        summary = laya.train.finetune(
            data=str(train_file),
            model_dir=base_dir,
            output_dir=str(checkpoint_dir),
            config=config,
            device="auto",
        )
        summary_file.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        print(f"Reusing locally trained checkpoint at {checkpoint_dir}", flush=True)

    agent = laya.Agent(str(checkpoint_dir))
    prediction_path = args.output / f"predictions_{MODEL_NAME}_w5.jsonl"
    all_messages = messages
    predictions = []
    for start in range(0, len(all_messages), args.batch_size):
        batch = all_messages[start : start + args.batch_size]
        texts = [w5_text(message) for message in batch]
        started = time.perf_counter()
        outputs = agent.decide_batch(texts, questions=questions, return_details=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        for message, output in zip(batch, outputs):
            answer = output.values["emotion_tracker"]
            probs = {label: float(answer["probabilities"].get(label, 0.0)) for label in LABELS}
            total = sum(probs.values())
            if total <= 0:
                raise ValueError(f"trained Laya returned invalid probabilities for {message['id']}")
            probs = {label: value / total for label, value in probs.items()}
            predictions.append({
                "id": message["id"], "model": MODEL_NAME,
                "model_id": TRAIN_MODEL_ID,
                "model_revision": f"local-finetune-from-{MODEL_REVISION}",
                "probs": probs,
                "raw_top_label": max(LABELS, key=probs.__getitem__),
                "latency_ms": elapsed_ms / max(len(batch), 1),
                "cost_usd": 0.0, "input_window_size": 5,
            })
        _write_jsonl(prediction_path, predictions)
        print(f"[{MODEL_NAME}] predicted {len(predictions)}/{len(all_messages)}", flush=True)

    baseline_path = BASELINE_RESULTS / "predictions_bot_laya_w5.jsonl"
    baseline_rows = read_jsonl(baseline_path)
    baseline_result = evaluate_model(
        "bot_laya", labels, messages, split, baseline_rows, args.bootstrap_iterations
    )
    learned_result = evaluate_model(
        MODEL_NAME, labels, messages, split, predictions, args.bootstrap_iterations
    )
    def report_row(result: dict, display: str) -> dict:
        metrics = result["test"]
        return {
            "model": display,
            "tau_selected_on_train": result["selected_tau"],
            "test_n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "top2_accuracy": metrics["top2_accuracy"],
            "brier_score": metrics["brier_score"],
            "log_loss": metrics["log_loss"],
            "cohen_kappa": metrics["cohen_kappa"],
            "human_neutral_pct": metrics["human_neutral_pct"],
            "predicted_neutral_pct": metrics["predicted_neutral_pct"],
            "mean_latency_ms": metrics["mean_latency_ms"],
            "bootstrap_95": metrics["cluster_bootstrap_95"]["intervals"],
            "per_class": metrics["per_class"],
        }

    chat_report = {
        "protocol": {
            "window": "W=5 (target plus up to four preceding messages)",
            "train_messages_including_calibration": len(train_labels),
            "gradient_training_messages": summary.get("train_items"),
            "training_calibration_messages": summary.get("calibration_items"),
            "held_out_test_messages": len(test_labels),
            "train_conversations": len(train_conversations),
            "test_conversations": len(test_conversations),
            "loss": "soft cross-entropy",
            "epochs": args.epochs,
            "encoder_frozen": True,
            "base_checkpoint": f"LaunchSpark/laya-emotion-tracker@{MODEL_REVISION}",
            "model_checkpoint_saved_locally": str(checkpoint_dir),
            "prediction_file_saved_locally": str(prediction_path),
        },
        "results": [
            report_row(baseline_result, "Pretrained Laya"),
            report_row(learned_result, "Learned Behavior Laya"),
        ],
    }
    print("CHAT_ONLY_W5_REPORT=" + json.dumps(chat_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
