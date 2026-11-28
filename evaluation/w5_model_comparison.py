"""Run and compare emotion models on the supplied W=5 labeled conversations.

All model inputs contain the target and up to four preceding messages. Threshold
selection is performed independently for each model on the training/calibration
partition; every reported final score uses the same held-out conversation groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

from evaluation.common.labels import LABELS
from evaluation.human_labeled_w5 import (
    DEFAULT_LABELS,
    DEFAULT_MESSAGES,
    OUTPUT as DEFAULT_OUTPUT,
    SEED,
    _bootstrap_ci,
    _metrics,
    _probability_metrics,
    _sweep,
    build_grouped_split,
    gate,
    load_aligned,
    read_jsonl,
    select_tau,
    w5_text,
)

ROOT = Path(__file__).resolve().parents[1]
LAYA_MODEL = "convaiinnovations/laya"
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LAYA_LOCAL_DIR = ROOT / ".local_models" / "convaiinnovations-laya"
LLM_MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_REVISION_DEFAULT = "7ae557604adf67be50417f59c2c2f167def9a775"
LLM_LOCAL_DIR = ROOT / ".local_models" / "qwen2.5-0.5b-instruct"
DISPLAY_NAMES = {
    "bot_laya": "Pretrained Laya",
    "learned_laya": "Learned Behavior Laya",
}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_or_predict(model: str, labels: list[dict], messages: list[dict], output: Path,
                     batch_size: int, llm_model: str, labels_path: Path,
                     messages_path: Path) -> tuple[list[dict], dict]:
    outpath = output / f"predictions_{model}_w5.jsonl"
    bot_laya_identity = None
    if model == "bot_laya":
        from vibeshift.emotions import MODEL_ID, MODEL_REVISION
        bot_laya_identity = (MODEL_ID, MODEL_REVISION)
    if outpath.exists():
        rows = read_jsonl(outpath)
        if (len(rows) == len(labels) and {r.get("id") for r in rows} == {r["id"] for r in labels}
                and all(r.get("input_window_size") == 5 for r in rows)
                and (bot_laya_identity is None or all(
                    r.get("model_id") == bot_laya_identity[0]
                    and r.get("model_revision") == bot_laya_identity[1] for r in rows
                ))):
            if model == "distilroberta":
                metadata_path = output / "raw_predictions" / model / "model_run.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"model_id": "j-hartmann/emotion-english-distilroberta-base"}
            elif model == "laya":
                metadata = {"model_id": LAYA_MODEL, "revision": LAYA_REVISION,
                            "window": "target + up to four preceding", "reused": True}
            elif model == "bot_laya":
                metadata = {"model_id": bot_laya_identity[0], "revision": bot_laya_identity[1],
                            "window": "target + up to four preceding", "reused": True}
            elif model == "nrc_emolex":
                metadata = {"lexicon": "NRC EmoLex v0.92", "window": "target + up to four preceding", "cost_usd": 0.0}
            elif model == "local_llm":
                metadata = {"model_id": llm_model,
                            "revision": LLM_REVISION_DEFAULT if llm_model == LLM_MODEL_DEFAULT else "recorded in predictions",
                            "device": "cpu", "window": "target + up to four preceding",
                            "api_cost_usd": 0.0,
                            "output_scores": "next-token logits over seven answer codes, normalized to seven choices"}
            else:
                metadata = {"reused": True}
            return rows, metadata
        if (model == "laya" and len(rows) == len(labels)
                and {r.get("id") for r in rows} == {r["id"] for r in labels}
                and all(r.get("model_id") == LAYA_MODEL and r.get("model_revision") == LAYA_REVISION
                        for r in rows)):
            # Earlier resumable batches from this same W=5 runner predate the
            # input_window_size metadata field; the model/revision and exact IDs
            # establish which run produced these rows.
            for row in rows:
                row["input_window_size"] = 5
            _write_jsonl(outpath, rows)
            return rows, {"model_id": LAYA_MODEL, "revision": LAYA_REVISION,
                          "window": "target + up to four preceding", "reused": True}

    if model == "distilroberta":
        from evaluation.human_labeled_w5 import predict
        # Run into a private model directory then use the standard prediction records.
        model_dir = output / "raw_predictions" / model
        path = predict(labels_path, messages_path, model_dir, batch_size=batch_size)
        rows = read_jsonl(path)
        _write_jsonl(outpath, rows)
        metadata = json.loads((model_dir / "model_run.json").read_text(encoding="utf-8"))
        return rows, metadata

    if model == "nrc_emolex":
        from evaluation.models.nrc_lexicon.run import _counts, _prediction, load_lexicon
        lexicon = load_lexicon()
        rows = []
        for message in messages:
            started = time.perf_counter()
            counts = _counts(w5_text(message), lexicon)
            probs, top = _prediction(counts)
            rows.append({"id": message["id"], "model": model, "model_id": "NRC-Emotion-Lexicon-Wordlevel-v0.92",
                         "probs": probs, "raw_top_label": top,
                         "latency_ms": (time.perf_counter() - started) * 1000, "cost_usd": 0.0,
                         "input_window_size": 5})
        _write_jsonl(outpath, rows)
        return rows, {"lexicon": "NRC EmoLex v0.92", "window": "target + up to four preceding", "cost_usd": 0.0}

    if model == "bot_laya":
        from vibeshift.emotions import score_batch

        states = [w5_text(message) for message in messages]
        ordered_ids = [message["id"] for message in messages]
        existing_by_id = {}
        if outpath.exists():
            partial = read_jsonl(outpath)
            expected = set(ordered_ids)
            for row in partial:
                if (row.get("id") not in expected or row.get("id") in existing_by_id
                        or row.get("model_id") != bot_laya_identity[0]
                        or row.get("model_revision") != bot_laya_identity[1]
                        or row.get("input_window_size") != 5):
                    raise ValueError(f"{outpath} contains invalid or incompatible partial predictions")
                existing_by_id[row["id"]] = row
        todo = [(message, state) for message, state in zip(messages, states)
                if message["id"] not in existing_by_id]
        rows = [existing_by_id[mid] for mid in ordered_ids if mid in existing_by_id]
        if rows:
            print(f"[{model}] resuming from {len(rows)}/{len(messages)} saved predictions", flush=True)
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            started = time.perf_counter()
            scores = score_batch([state for _, state in batch])
            elapsed_ms = (time.perf_counter() - started) * 1000
            if len(scores) != len(batch):
                raise ValueError(f"bot Laya returned {len(scores)} scores for {len(batch)} inputs")
            for (message, _), score in zip(batch, scores):
                probs = {label: float(score.probabilities.get(label, 0.0)) for label in LABELS}
                total = sum(probs.values())
                if not math.isfinite(total) or total <= 0:
                    raise ValueError(f"bot Laya returned invalid probabilities for {message['id']}")
                probs = {label: value / total for label, value in probs.items()}
                existing_by_id[message["id"]] = {
                    "id": message["id"], "model": model, "model_id": bot_laya_identity[0],
                    "model_revision": bot_laya_identity[1], "probs": probs,
                    "raw_top_label": max(LABELS, key=probs.__getitem__),
                    "latency_ms": elapsed_ms / max(len(batch), 1), "cost_usd": 0.0,
                    "input_window_size": 5,
                }
            rows = [existing_by_id[mid] for mid in ordered_ids if mid in existing_by_id]
            _write_jsonl(outpath, rows)
            print(f"[{model}] predicted {len(rows)}/{len(states)}; saved {len(rows)}",
                  flush=True)
        return rows, {"model_id": bot_laya_identity[0], "revision": bot_laya_identity[1],
                      "window": "target + up to four preceding", "scoring_path": "bot production score_batch"}

    if model == "laya":
        import laya
        from huggingface_hub import snapshot_download
        criteria = {
            "joy": "Happiness, amusement, excitement, gratitude, affection, or positive enjoyment.",
            "sadness": "Sadness, grief, disappointment, loneliness, or emotional distress.",
            "anger": "Anger, irritation, frustration, hostility, or outrage.",
            "fear": "Fear, anxiety, worry, nervousness, or threat.",
            "disgust": "Disgust, revulsion, contempt, or strong aversion.",
            "surprise": "Surprise, astonishment, or unexpectedness without a stronger other emotion.",
            "neutral": "No clear emotion is expressed by the target message in its context.",
        }
        questions = {"emotion": {"type": "choice", "instructions":
                      "Classify the emotion expressed by the TARGET MESSAGE, using preceding messages only as context. Choose exactly one best category.",
                      "criteria": criteria}}
        LAYA_LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(LAYA_MODEL, revision=LAYA_REVISION,
                                       local_dir=str(LAYA_LOCAL_DIR))
        agent = laya.load(model_path)
        states = [w5_text(m) for m in messages]
        existing = {r["id"]: r for r in read_jsonl(outpath)} if outpath.exists() else {}
        predictions = [existing[m["id"]] for m in messages if m["id"] in existing]
        done = set(existing)
        todo = [(m, s) for m, s in zip(messages, states) if m["id"] not in done]
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            started = time.perf_counter()
            raw = agent.predict_batch([item[1] for item in batch], questions, batch_size=batch_size)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if len(raw) != len(batch):
                raise ValueError(f"Laya returned {len(raw)} outputs for {len(batch)} inputs")
            for (msg, _), result in zip(batch, raw):
                answer = result["answers"]["emotion"]
                probs = {label: float(answer["probabilities"].get(label, 0.0)) for label in LABELS}
                total = sum(probs.values())
                if not math.isfinite(total) or total <= 0:
                    raise ValueError(f"Laya returned invalid probabilities for {msg['id']}")
                probs = {label: value / total for label, value in probs.items()}
                predictions.append({"id": msg["id"], "model": model, "model_id": LAYA_MODEL,
                                    "model_revision": LAYA_REVISION, "probs": probs,
                                    "raw_top_label": max(LABELS, key=probs.__getitem__),
                                    "latency_ms": elapsed_ms / max(len(batch), 1), "cost_usd": 0.0,
                                    "input_window_size": 5})
            _write_jsonl(outpath, predictions)
            print(f"[{model}] predicted {min(start + len(batch), len(todo))}/{len(todo)} remaining; saved {len(predictions)} total",
                  flush=True)
        _write_jsonl(outpath, predictions)
        del agent
        return predictions, {"model_id": LAYA_MODEL, "revision": LAYA_REVISION,
                             "batch_size": batch_size, "window": "target + up to four preceding",
                             "probabilities": "native seven-way choice scores; no Discord soft-probability calibration"}

    if model == "local_llm":
        import torch
        from huggingface_hub import model_info, snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LLM_LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        revision = LLM_REVISION_DEFAULT if llm_model == LLM_MODEL_DEFAULT else model_info(llm_model).sha
        local_path = snapshot_download(llm_model, revision=revision, local_dir=str(LLM_LOCAL_DIR))
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_obj = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.float32)
        model_obj.to(device).eval()
        codes = list("ABCDEFG")
        code_ids = [tokenizer.encode(code, add_special_tokens=False) for code in codes]
        if any(len(tokens) != 1 for tokens in code_ids):
            raise ValueError("the local LLM tokenizer must represent A-G as single tokens for logit scoring")
        code_ids = [tokens[0] for tokens in code_ids]
        existing = {r["id"]: r for r in read_jsonl(outpath)} if outpath.exists() else {}
        rows = [existing[m["id"]] for m in messages if m["id"] in existing]
        todo = [m for m in messages if m["id"] not in existing]
        descriptions = {
            "joy": "happiness, enjoyment, excitement, gratitude, or affection",
            "sadness": "sadness, grief, disappointment, loneliness, or distress",
            "anger": "anger, irritation, frustration, hostility, or outrage",
            "fear": "fear, anxiety, worry, nervousness, or threat",
            "disgust": "disgust, revulsion, contempt, or strong aversion",
            "surprise": "surprise or astonishment without a stronger other emotion",
            "neutral": "no clear emotion expressed by the target message",
        }
        import hashlib
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            label_orders, prompts = [], []
            for message in batch:
                offset = int(hashlib.sha256(message["id"].encode()).hexdigest()[:8], 16) % len(LABELS)
                ordered = list(LABELS[offset:]) + list(LABELS[:offset])
                label_orders.append(ordered)
                options = "\n".join(f"{codes[i]}: {label} — {descriptions[label]}" for i, label in enumerate(ordered))
                user = ("Classify only the TARGET MESSAGE, using the preceding messages as context. "
                        "Choose the best one category. The options are:\n" + options +
                        "\nOutput only its single letter code.\n\n" + w5_text(message))
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a zero-shot emotion classification model."},
                     {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096,
                               padding=True).to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                last_indices = inputs["attention_mask"].sum(dim=1) - 1
                hidden = model_obj.model(input_ids=inputs["input_ids"],
                                         attention_mask=inputs["attention_mask"]).last_hidden_state
                last_hidden = hidden[torch.arange(len(batch), device=hidden.device), last_indices]
                last_logits = model_obj.get_output_embeddings()(last_hidden)
                choice_probs = torch.softmax(last_logits[:, code_ids].float(), dim=-1).cpu().tolist()
            elapsed_ms = (time.perf_counter() - started) * 1000
            for i, message in enumerate(batch):
                probs = {label: 0.0 for label in LABELS}
                for j, label in enumerate(label_orders[i]):
                    probs[label] = float(choice_probs[i][j])
                rows.append({"id": message["id"], "model": model, "model_id": llm_model,
                             "model_revision": revision, "probs": probs,
                             "raw_top_label": max(LABELS, key=probs.__getitem__),
                             "latency_ms": elapsed_ms / len(batch), "cost_usd": 0.0,
                             "input_window_size": 5,
                             "input_tokens": int(inputs["attention_mask"][i].sum().item()),
                             "output_tokens": 1})
            _write_jsonl(outpath, rows)
            print(f"[{model}] scored {min(start + len(batch), len(todo))}/{len(todo)} remaining; saved {len(rows)} total", flush=True)
        del model_obj
        return rows, {"model_id": llm_model, "revision": revision, "device": device,
                      "window": "target + up to four preceding", "api_cost_usd": 0.0,
                      "output_scores": "next-token logits normalized across seven single-token choice codes; label order rotated deterministically per message"}
    raise ValueError(f"unknown model: {model}")


def evaluate_model(model: str, labels: list[dict], messages: list[dict], split: dict[str, str],
                   rows: list[dict], iterations: int) -> dict:
    by_id = {r["id"]: r for r in rows}
    expected = {row["id"] for row in labels}
    if set(by_id) != expected:
        raise ValueError(f"{model}: prediction IDs differ from labels")
    for mid, row in by_id.items():
        probs = row.get("probs", {})
        if set(probs) != set(LABELS) or any(not math.isfinite(float(p)) or float(p) < 0 for p in probs.values()):
            raise ValueError(f"{model}: invalid probability vector for {mid}")
        if abs(sum(float(p) for p in probs.values()) - 1) > 1e-3:
            raise ValueError(f"{model}: probabilities for {mid} do not sum to 1")
    sweep = _sweep(labels, by_id, split)
    selected = select_tau(sweep)
    test = [r for r in labels if split[r["id"]] == "test"]
    y_true = [r["label"] for r in test]
    y_pred = [gate(by_id[r["id"]]["probs"], selected["tau"]) for r in test]
    probs = [by_id[r["id"]]["probs"] for r in test]
    hard = _metrics(y_true, y_pred)
    hard.update(_probability_metrics(y_true, probs))
    hard["mean_latency_ms"] = sum(float(by_id[r["id"]].get("latency_ms", 0)) for r in test) / len(test)
    hard["cost_usd_total"] = sum(float(by_id[r["id"]].get("cost_usd", 0)) for r in test)
    message_by_id = {m["id"]: m for m in messages}
    records = [(message_by_id[r["id"]]["conversation_key"], r["label"], pred)
               for r, pred in zip(test, y_pred)]
    hard["cluster_bootstrap_95"] = _bootstrap_ci(records, SEED, iterations)
    hard["majority_class_baseline"] = _metrics(y_true,
        [Counter(r["label"] for r in labels if split[r["id"]] == "train").most_common(1)[0][0]] * len(test))
    hard["normalized_confusion"] = [
        [float(count) / sum(row) if sum(row) else 0.0 for count in row]
        for row in hard["confusion"]
    ]
    return {"model": model, "selected_tau": selected["tau"],
            "calibration_neutral_match_pct_points": selected["neutral_abs_difference_pct"],
            "calibration_human_neutral_pct": selected["human_neutral_pct"],
            "calibration_predicted_neutral_pct": selected["predicted_neutral_pct"],
            "test": hard, "tau_sweep": sweep,
            "test_classification": hard["per_class"], "test_confusion": hard["confusion"]}


def write_report(results: list[dict], model_metadata: dict, output: Path, split: dict,
                 messages: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_comparison.json").write_text(json.dumps({
        "window_size": 5, "input_description": "target plus up to four preceding messages, target explicitly marked",
        "split": {"train_n": sum(v == "train" for v in split.values()),
                  "test_n": sum(v == "test" for v in split.values()),
                  "method": "StratifiedGroupKFold first of five folds; random_state=4240; groups=conversation"},
        "models": results, "model_metadata": model_metadata}, indent=2) + "\n", encoding="utf-8")
    lines = ["# W=5 Model Comparison", "",
        "All systems received the target message and up to four preceding messages. Whole conversations were held out by the same deterministic split. Each model's tau was selected on training/calibration data by neutral-rate matching, with balanced accuracy and Macro-F1 tie breaks.", "",
        "Weighted F1 is the support-weighted average of per-class F1. Because neutral is about 78% of the held-out labels, it is strongly influenced by neutral performance. Macro-F1 gives each class equal weight; balanced accuracy averages class recall. Consider all three under this class imbalance.", "",
        f"Train/calibration: {sum(v == 'train' for v in split.values())}; held-out test: {sum(v == 'test' for v in split.values())}; test conversations: {len({m['conversation_key'] for m in messages if split.get(m['id']) == 'test'})}.", "",
        "## Held-out test summary", "",
        "| Model | Tau | Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 | Top-2 accuracy | Brier (↓) | Log loss (↓) | Kappa | Neutral human / predicted | Mean latency (ms) | Test cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        m = result["test"]
        lines.append(f"| {display_name(result['model'])} | {result['selected_tau']:.4f} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['top2_accuracy']:.3f} | {m['brier_score']:.3f} | {m['log_loss']:.3f} | {m['cohen_kappa']:.3f} | {m['human_neutral_pct']:.2f}% / {m['predicted_neutral_pct']:.2f}% | {m['mean_latency_ms']:.2f} | ${m['cost_usd_total']:.5f} |")
    lines += ["", "## Majority-class reference", "",
              "The majority-class baseline always predicts the most frequent training label (neutral). Its metrics on the same held-out rows:", "",
              "| Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 |", "|---:|---:|---:|---:|"]
    baseline = results[0]["test"]["majority_class_baseline"] if results else {}
    lines.append(f"| {baseline.get('accuracy', 0):.3f} | {baseline.get('balanced_accuracy', 0):.3f} | {baseline.get('macro_f1', 0):.3f} | {baseline.get('weighted_f1', 0):.3f} |")
    lines += ["", "## Training-only tau calibration", "",
              "Tau is selected per model using training/calibration messages only. The selection criterion minimizes the absolute neutral-rate difference; ties prefer higher balanced accuracy, then Macro-F1, then lower tau.", "",
              "| Model | Tau | Human neutral | Predicted neutral | Absolute gap (percentage points) |", "|---|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {display_name(result['model'])} | {result['selected_tau']:.6f} | {result['calibration_human_neutral_pct']:.2f}% | {result['calibration_predicted_neutral_pct']:.2f}% | {result['calibration_neutral_match_pct_points']:.3f} |")
    lines += ["", "## Exact model identities", "",
              "| Runner | Model / resource | Revision or version | Device / notes |", "|---|---|---|---|"]
    for model_name, info in model_metadata.items():
        lines.append(f"| {display_name(model_name)} | {info.get('model_id', info.get('lexicon', model_name))} | {info.get('revision', info.get('resolved_revision', 'local resource'))} | {info.get('device', info.get('window', ''))} |")
    lines += ["",
              "## Per-class and averaged held-out metrics", ""]
    for result in results:
        m = result["test"]
        lines += [f"### {display_name(result['model'])}", "",
                  f"Macro average: precision {m['macro_precision']:.3f}, recall {m['macro_recall']:.3f}, F1 {m['macro_f1']:.3f}. Weighted average: precision {m['weighted_precision']:.3f}, recall {m['weighted_recall']:.3f}, F1 {m['weighted_f1']:.3f}.", "",
                  "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label in LABELS:
            item = result["test_classification"][label]
            lines.append(f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |")
        matrix_header = "| Human \\ Predicted | " + " | ".join(LABELS) + " |"
        matrix_rule = "|---|" + "---:|" * len(LABELS)
        lines += ["", "Raw confusion matrix (rows=human, columns=predicted):", "", matrix_header, matrix_rule]
        for label, row in zip(LABELS, result["test_confusion"]):
            lines.append("| " + label + " | " + " | ".join(str(n) for n in row) + " |")
        lines += ["", "Row-normalized confusion matrix (each human-label row sums to 1 where supported):", "",
                  matrix_header, matrix_rule]
        for label, row in zip(LABELS, result["test"]["normalized_confusion"]):
            lines.append("| " + label + " | " + " | ".join(f"{n:.3f}" for n in row) + " |")
        confusion = result["test_confusion"]
        errors = sorted(((confusion[i][j], LABELS[i], LABELS[j]) for i in range(len(LABELS))
                         for j in range(len(LABELS)) if i != j), reverse=True)
        lines += ["", "Largest off-diagonal errors: " + "; ".join(
            f"{gold} → {pred} ({count})" for count, gold, pred in [e for e in errors if e[0]][:5]) + ".",
            f"Conversation-cluster bootstrap 95% intervals: `{json.dumps(result['test']['cluster_bootstrap_95']['intervals'], sort_keys=True)}`.",
                  f"Tau sweep: `tau_sweep_{result['model']}.csv`.", ""]
    if results:
        best = max(results, key=lambda row: row["test"]["macro_f1"])
        lines += ["## Highest held-out Macro-F1", "",
                  f"Among completed models, **{display_name(best['model'])}** has the highest Macro-F1 ({best['test']['macro_f1']:.3f}); accuracy {best['test']['accuracy']:.3f}, balanced accuracy {best['test']['balanced_accuracy']:.3f}. This is a ranking on a small single-annotator holdout, not proof of general superiority.", ""]
    lines += ["## Method and limitations", "",
        "Gold labels are from one annotator, so these metrics measure agreement with that annotator rather than consensus truth. There are only 214 test messages from 10 conversations and only one fear example and three anger examples; rare-class metrics and confidence intervals are unstable. W=5 windows overlap; conversation-level splitting prevents overlapping context from crossing the train/test boundary. Models receive serialized context with the target marked; tokenizer truncation may discard older context for transformer models.", "",
        "The NRC baseline counts dictionary word/emotion matches and has no learned language understanding. When a window has no recognized emotion words, it emits a hard neutral-only distribution; its zero probabilities for other classes produce the large held-out log loss (19.068). Laya, the zero-shot LLM, and transformer probabilities may not be calibrated on Discord. The local language model reports relative next-token scores; these are not guaranteed to be calibrated probabilities. EmoLex remains local and is not included in repository outputs.", "",
        "## Artifacts", "",
        "- `model_comparison.json`: complete numeric metrics and metadata.",
        "- `predictions_*_w5.jsonl`: probabilities and run metadata per message.",
        "- `tau_sweep_*.csv`: calibration scores across tau values.",
        "- `confusion_*_raw.csv`, `confusion_*_normalized.csv`, `classification_*.csv`: per-model held-out matrices and class reports.", ""]
    (output / "model_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default="all", help="comma list: distilroberta,laya,bot_laya,nrc_emolex,local_llm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--llm-model", default=LLM_MODEL_DEFAULT)
    args = parser.parse_args()
    labels, messages = load_aligned(args.labels, args.messages)
    split = build_grouped_split(labels, messages)
    models = (["distilroberta", "laya", "nrc_emolex", "local_llm"]
              if args.models.strip().lower() == "all"
              else [m.strip() for m in args.models.split(",") if m.strip()])
    metadata, results = {}, []
    for model in models:
        rows, metadata[model] = _load_or_predict(model, labels, messages, args.output, args.batch_size,
                                                   args.llm_model, args.labels, args.messages)
        result = evaluate_model(model, labels, messages, split, rows, args.bootstrap_iterations)
        results.append(result)
        (args.output / f"tau_sweep_{model}.csv").parent.mkdir(parents=True, exist_ok=True)
        with (args.output / f"tau_sweep_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["tau_sweep"][0]))
            writer.writeheader()
            writer.writerows(result["tau_sweep"])
        for suffix, matrix in (("raw", result["test_confusion"]),
                               ("normalized", result["test"]["normalized_confusion"])):
            path = args.output / f"confusion_{model}_{suffix}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["human_label", *[f"pred_{label}" for label in LABELS]]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for label, row in zip(LABELS, matrix):
                    writer.writerow({"human_label": label, **{f"pred_{name}": value for name, value in zip(LABELS, row)}})
        with (args.output / f"classification_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["label", "precision", "recall", "f1", "support"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for label in LABELS:
                writer.writerow({"label": label, **result["test_classification"][label]})
    write_report(results, metadata, args.output, split, messages)
    print(f"wrote comparison report to {args.output / 'model_comparison_report.md'}")


if __name__ == "__main__":
    main()
