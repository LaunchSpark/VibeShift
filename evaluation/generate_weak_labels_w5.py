"""Create an auditable weak-label CSV from conversations outside the human set.

The integrated bot Laya model scores W=5 context. Explicit first-person emotion
phrases override the model category; a tau gate calibrated to the human training
neutral prior assigns neutral to lower-confidence non-neutral predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.human_labeled_w5 import read_jsonl, w5_text
from evaluation.common.labels import LABELS, NEUTRAL
from vibeshift.emotions import MODEL_ID, MODEL_REVISION, score_batch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "discord_messages.csv"
MESSAGES_JSONL = ROOT / "evaluation" / "data" / "messages.jsonl"
HUMAN_LABELS = ROOT / "human_labeled_dataset" / "discord_messages_braylon-trail_w5.jsonl"
AUDIT_JSON = ROOT / "evaluation" / "results" / "human_labeled_w5" / "dataset_audit.json"
OUTPUT_CSV = ROOT / "data" / "synthetic_emotion_labels_w5.csv"
LABEL_ORDER = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

# Deliberately narrow overrides: these are first-person expressions, not general
# keyword hits (for example, a message merely mentioning "fear" is not enough).
PHRASES: dict[str, re.Pattern[str]] = {
    "joy": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(happy|glad|joyful|delighted|excited|cheerful|pleased|thrilled)\b", re.I,
    ),
    "sadness": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(sad|unhappy|depressed|heartbroken|miserable|down|lonely|disappointed)\b", re.I,
    ),
    "anger": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(angry|mad|furious|annoyed|irritated|pissed|frustrated)\b", re.I,
    ),
    "fear": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(afraid|scared|fearful|terrified|anxious|worried|nervous)\b", re.I,
    ),
    "disgust": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(disgusted|grossed\s+out|revolted|sickened)\b", re.I,
    ),
    "surprise": re.compile(
        r"\b(?:i\s+am|i'm|im|i\s+feel|i'm\s+feeling|i\s+am\s+feeling)\s+"
        r"(?:so\s+|very\s+|really\s+|pretty\s+|quite\s+)?"
        r"(surprised|shocked|amazed|astonished)\b", re.I,
    ),
}


def _read_source() -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"conversation_key", "conversation_message_number", "author", "message"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{SOURCE_CSV} is empty or missing columns {sorted(required)}")
    return rows


def _explicit_label(text: str, probs: dict[str, float]) -> tuple[str | None, str]:
    hits = [(label, match.group(0)) for label, pattern in PHRASES.items()
            if (match := pattern.search(text))]
    if not hits:
        return None, ""
    # If a message states more than one explicit feeling, choose the stated
    # feeling with the highest model probability; label order breaks ties.
    hits.sort(key=lambda item: (-probs.get(item[0], 0.0), LABEL_ORDER.index(item[0])))
    return hits[0]


def _tau_for_target(rows: list[dict[str, Any]], target_neutral_ratio: float) -> tuple[float, int]:
    target_emotional = round(len(rows) * (1.0 - target_neutral_ratio))
    explicit_count = sum(row["explicit_label"] is not None for row in rows)
    needed_from_model = max(0, target_emotional - explicit_count)
    candidates = sorted(
        (row["max_emotion_probability"] for row in rows
         if row["explicit_label"] is None and row["raw_prediction"] != NEUTRAL),
        reverse=True,
    )
    if needed_from_model == 0:
        return 1.000001, target_emotional
    if needed_from_model > len(candidates):
        # There are not enough non-neutral model winners to meet the prior
        # without overriding the model's top class, so use every eligible one.
        return min(candidates, default=1.0), explicit_count + len(candidates)
    cutoff = candidates[needed_from_model - 1]
    return cutoff, target_emotional


def generate(batch_size: int = 8) -> dict[str, Any]:
    source_rows = _read_source()
    human = read_jsonl(HUMAN_LABELS)
    human_ids = {str(row["id"]) for row in human}
    used_conversations = {message_id.rsplit(":", 1)[0] for message_id in human_ids}
    messages = read_jsonl(MESSAGES_JSONL)
    by_id = {str(message["id"]): message for message in messages}

    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        conversation = str(row["conversation_key"])
        number = str(row["conversation_message_number"])
        message_id = f"{conversation}:{number}"
        if conversation in used_conversations:
            continue
        if message_id in seen:
            raise ValueError(f"duplicate source message id: {message_id}")
        seen.add(message_id)
        message = by_id.get(message_id)
        if message is None:
            raise ValueError(f"source CSV id {message_id} missing from W=5 context file")
        # The source CSV stores embedded line breaks as the two literal
        # characters ``\\n``; the W=5 JSONL has decoded them to newlines.
        source_text = row["message"].replace("\\n", "\n")
        if message.get("content", "") != source_text:
            raise ValueError(f"source text mismatch for {message_id}")
        eligible.append({"id": message_id, "source": {**row, "message": source_text}, "message": message})

    if not eligible:
        raise ValueError("no conversations remain after excluding the human-labeled set")

    scores_by_id: dict[str, Any] = {}
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start:start + batch_size]
        scores = score_batch([w5_text(item["message"]) for item in batch])
        if len(scores) != len(batch):
            raise ValueError("Laya returned a different number of scores than inputs")
        for item, score in zip(batch, scores):
            probs = {label: float(score.probabilities.get(label, 0.0)) for label in LABELS}
            total = sum(probs.values())
            if total <= 0 or not (0.999 <= total <= 1.001):
                if total <= 0:
                    raise ValueError(f"invalid Laya probabilities for {item['id']}")
                probs = {label: value / total for label, value in probs.items()}
            scores_by_id[item["id"]] = probs
        print(f"Scored {min(start + len(batch), len(eligible))}/{len(eligible)}", flush=True)

    prepared = []
    for item in eligible:
        probs = scores_by_id[item["id"]]
        raw = max(LABEL_ORDER, key=probs.__getitem__)
        explicit, phrase = _explicit_label(item["source"]["message"], probs)
        prepared.append({
            **item,
            "probs": probs,
            "raw_prediction": raw,
            "max_emotion_probability": max(probs[label] for label in LABEL_ORDER if label != NEUTRAL),
            "explicit_label": explicit,
            "matched_phrase": phrase,
        })

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    train_counts = audit["partitions"]["train"]["labels"]
    target_neutral = train_counts[NEUTRAL] / audit["partitions"]["train"]["n"]
    tau, target_emotional_count = _tau_for_target(prepared, target_neutral)

    for row in prepared:
        probs = row["probs"]
        if row["explicit_label"] is not None:
            final_label = row["explicit_label"]
            source = "explicit_first_person_emotion_phrase"
        elif row["raw_prediction"] == NEUTRAL or row["max_emotion_probability"] < tau:
            final_label = NEUTRAL
            source = "laya_tau_neutral_gate"
        else:
            final_label = row["raw_prediction"]
            source = "laya_tau_emotion_prediction"
        row["final_label"] = final_label
        row["label_source"] = source
        row["label_probability"] = probs[final_label]

    output_rows = []
    for row in prepared:
        source_row = row["source"]
        message = row["message"]
        output = {
            "id": row["id"],
            "conversation_key": source_row["conversation_key"],
            "conversation_message_number": source_row["conversation_message_number"],
            "author": source_row["author"],
            "context_previous_4": " | ".join(message.get("preceding_25", [])[-4:]),
            "message": source_row["message"],
            "label": row["final_label"],
            "label_source": row["label_source"],
            "matched_explicit_phrase": row["matched_phrase"],
            "raw_model_prediction": row["raw_prediction"],
            "model_probability_for_label": f"{row['label_probability']:.8f}",
            "model_confidence": f"{max(row['probs'].values()):.8f}",
            "tau": f"{tau:.8f}",
            "target_neutral_ratio_from_human_train": f"{target_neutral:.8f}",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        output.update({f"prob_{label}": f"{row['probs'][label]:.8f}" for label in LABEL_ORDER})
        output_rows.append(output)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    labels = Counter(row["final_label"] for row in prepared)
    explicit_counts = Counter(row["explicit_label"] for row in prepared if row["explicit_label"])
    return {
        "output": str(OUTPUT_CSV),
        "rows": len(output_rows),
        "conversations": len({row["source"]["conversation_key"] for row in prepared}),
        "human_labeled_conversations_excluded": len(used_conversations),
        "overlap_conversations": sorted({row["source"]["conversation_key"] for row in prepared} & used_conversations),
        "tau": tau,
        "target_neutral_ratio": target_neutral,
        "actual_neutral_ratio": labels[NEUTRAL] / len(prepared),
        "label_counts": dict(labels),
        "explicit_phrase_counts": dict(explicit_counts),
        "target_emotional_count": target_emotional_count,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    print(json.dumps(generate(args.batch_size), indent=2), flush=True)


if __name__ == "__main__":
    main()
