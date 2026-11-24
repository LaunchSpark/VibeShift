"""Error analysis helper for Teammate B (#64).

Extracts and categorizes representative misclassifications from test predictions
against gold labels, separating them into:
1. short replies (len(content.split()) <= 3)
2. slang/emoji/all-caps
3. context-dependent (gold W=1 != gold W=4)
4. ambiguous / high disagreement (entropy or low consensus margin)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from evaluation.common import io
from evaluation.common.labels import NEUTRAL
from evaluation.eval import metrics

REPORT_DIR = io.ROOT / "eval" / "report"

_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|:[a-zA-Z0-9_]+:|(?:\b(?:lol|lmao|idk|smh|tbh|bruh|wtf|gg|rip|afaik|imo|omg)\b)",
    re.IGNORECASE,
)


def categorize_error(
    msg: dict,
    gold_label: str,
    pred_label: str,
    gold_w1_label: str | None = None,
    soft_probs: dict[str, float] | None = None,
) -> str:
    """Assign an error category to a misclassification."""
    content = msg.get("content", "").strip()
    words = content.split()

    # 1. Context-dependent: human label changed when context was provided
    if gold_w1_label and gold_w1_label != gold_label:
        return "context-dependent (W=1 vs W=4 gold differs)"

    # 2. Ambiguous: soft_probs show close disagreement among labelers
    if soft_probs:
        sorted_probs = sorted(soft_probs.values(), reverse=True)
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.25:
            return "ambiguous (labeler disagreement)"

    # 3. Short replies
    if len(words) <= 3:
        return "short reply (<= 3 words)"

    # 4. Slang / emoji / shouting
    if _EMOJI_PATTERN.search(content) or (len(content) > 4 and content.isupper()):
        return "slang / emoji / shouting"

    return "general semantic confusion"


def extract_misclassifications(
    model: str,
    split: str = "test",
    window: int | None = None,
    tau: float = 0.0,
    limit_per_category: int = 3,
) -> dict[str, list[dict]]:
    """Pull representative misclassifications grouped by category."""
    messages = {m["id"]: m for m in io.load_messages()}
    windows = io.consensus_windows()
    target_window = window
    if target_window is None:
        if 4 in windows:
            target_window = 4
        elif windows:
            target_window = min(windows)

    gold_target = io.load_consensus(window=target_window)
    gold_w1 = {}
    if io.consensus_path(1).exists():
        try:
            gold_w1 = io.load_consensus(window=1)
        except Exception:
            gold_w1 = {}
    predictions = io.load_predictions(model)

    test_ids = {m["id"] for m in messages.values() if m.get("split") == split or split == "all"}
    errors_by_category: dict[str, list[dict]] = {}

    for row in predictions:
        mid = row["id"]
        if mid not in test_ids or mid not in gold_target:
            continue

        gold = gold_target[mid]["label"]
        pred = metrics.gated_label(row["probs"], row["raw_top_label"], tau)
        if pred == gold:
            continue

        msg = messages.get(mid, {})
        w1_gold = gold_w1.get(mid, {}).get("label") if gold_w1 else None
        soft = gold_target[mid].get("soft_probs")

        cat = categorize_error(msg, gold, pred, w1_gold, soft)
        if cat not in errors_by_category:
            errors_by_category[cat] = []

        if len(errors_by_category[cat]) < limit_per_category:
            errors_by_category[cat].append({
                "id": mid,
                "content": msg.get("content", ""),
                "preceding_3": msg.get("preceding_3", []),
                "gold_label": gold,
                "pred_label": pred,
                "gold_w1": w1_gold,
                "model_probs": row["probs"],
                "category": cat,
            })

    return errors_by_category


def generate_error_analysis_markdown(
    errors_by_category: dict[str, list[dict]],
    model_name: str,
    output_path: Path,
) -> str:
    """Format categorized errors into error_analysis.md."""
    lines = [
        f"# Qualitative Error Analysis: {model_name}",
        "",
        "This report analyzes representative test misclassifications categorized by root cause, ",
        "highlighting the challenges of Discord conversational emotion detection.",
        "",
    ]

    total_examples = sum(len(items) for items in errors_by_category.values())
    lines.append(f"Analyzed **{total_examples}** representative test errors across **{len(errors_by_category)}** categories.\n")

    for cat, examples in sorted(errors_by_category.items()):
        lines.append(f"## Category: {cat.title()}")
        lines.append("")
        for ex in examples:
            ctx_str = " | ".join(ex["preceding_3"]) if ex["preceding_3"] else "*(none)*"
            lines.append(f"- **Message ID:** `{ex['id']}`")
            lines.append(f"  - **Context:** \"{ctx_str}\"")
            lines.append(f"  - **Target Message:** \"{ex['content']}\"")
            lines.append(f"  - **True Gold Label:** `{ex['gold_label']}` (W=1: `{ex.get('gold_w1') or 'N/A'}`)")
            lines.append(f"  - **Model Prediction:** `{ex['pred_label']}`")
            lines.append(f"  - **Diagnosis:** Misclassified because cues depend on {cat.lower()}.")
            lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya", help="model prediction name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--window", type=int, help="window size to score against (defaults to 4 or available consensus)")
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()

    if not io.prediction_path(args.model).exists():
        print(f"Prediction file predictions/{args.model}.jsonl not found.")
        return

    errors = extract_misclassifications(args.model, split=args.split, window=args.window, tau=args.tau)
    out_file = REPORT_DIR / "error_analysis.md"
    generate_error_analysis_markdown(errors, args.model, out_file)
    print(f"Generated error analysis in {out_file.relative_to(io.ROOT.parent)}")


if __name__ == "__main__":
    main()
