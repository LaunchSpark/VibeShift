"""Score every model in ``predictions/`` against the gold consensus labels.

Produces the final report table (Macro-F1 overall + on context-sensitive
messages, Top-2 accuracy, multi-class Brier,
log-loss, JSD, latency, cost) and the report charts: a model comparison, plus a
confusion matrix, label-distribution comparison, and τ sweep per model. All
metrics are schema-ordered (see ``metrics.py``); charts are Altair, rendered to
PNG headlessly by vl-convert-python. For interactive drill-down, use
``pixi run eval-notebook``.

Run:
    pixi run eval-report                  # default τ (VIBESHIFT_TAU or 0.0), test split
    python -m evaluation.eval.evaluate --tau 0.3 --split test --window 5

"Context-sensitive" messages are those whose gold label differs between the
smallest and largest window size labeled (``io.context_sensitive_ids``); the
column is NaN until gold exists for two window sizes.
"""

from __future__ import annotations

import argparse
import json

from sklearn.metrics import precision_recall_fscore_support

from evaluation.common import io
from evaluation.common.labels import LABELS
from evaluation.eval import charts, frames, metrics

REPORT_DIR = io.ROOT / "eval" / "report"
PPI = 200


def evaluate_model(
    rows: list[dict], gold: dict[str, dict], tau: float, context_ids: set[str] | None = None
) -> dict:
    """Metrics for one model over the messages it and the gold set share."""
    rows = [r for r in rows if r["id"] in gold]
    ids = [r["id"] for r in rows]
    probs_list = [r["probs"] for r in rows]
    gold_labels = [gold[i]["label"] for i in ids]
    pred_labels = [metrics.gated_label(r["probs"], r["raw_top_label"], tau) for r in rows]

    def sliced(keep: set[str] | None) -> float:
        if keep is None:
            return float("nan")
        g = [gl for gl, i in zip(gold_labels, ids) if i in keep]
        p = [pl for pl, i in zip(pred_labels, ids) if i in keep]
        return metrics.macro_f1(g, p) if g else float("nan")

    # JSD: predicted distribution vs gold, two ways.
    pred_mean = metrics.mean_distribution(probs_list)
    gold_mix = metrics.label_mix(gold_labels)
    gated_mix = metrics.label_mix(pred_labels)

    return {
        "n": len(rows),
        "macro_f1": metrics.macro_f1(gold_labels, pred_labels),
        "macro_f1_context": sliced(context_ids),
        "top2_acc": metrics.top_k_accuracy(gold_labels, probs_list, k=2),
        "brier": metrics.brier_score(gold_labels, probs_list),
        "log_loss": metrics.log_loss(gold_labels, probs_list),
        "jsd_mean_probs_vs_gold": metrics.jensen_shannon(pred_mean, gold_mix),
        "jsd_gated_counts_vs_gold": metrics.jensen_shannon(gated_mix, gold_mix),
        "latency_ms": sum(r["latency_ms"] for r in rows) / max(len(rows), 1),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
        "per_class": {
            label: {"precision": float(p), "recall": float(r)}
            for label, p, r in zip(
                LABELS,
                *precision_recall_fscore_support(
                    gold_labels, pred_labels, labels=list(LABELS), zero_division=0,
                )[:2],
            )
        },
    }


_COLUMNS = [
    ("macro_f1", "Macro-F1"),
    ("macro_f1_context", "Macro-F1 (ctx-sensitive)"),
    ("top2_acc", "Top-2"),
    ("brier", "Brier"),
    ("log_loss", "LogLoss"),
    ("jsd_mean_probs_vs_gold", "JSD probs"),
    ("jsd_gated_counts_vs_gold", "JSD gated"),
    ("latency_ms", "ms/msg"),
    ("cost_usd", "$ total"),
]


def _table(results: dict[str, dict]) -> str:
    head = "| Model | n | " + " | ".join(h for _, h in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 2)
    lines = [head, sep]
    for name, r in results.items():
        cells = []
        for key, _ in _COLUMNS:
            v = r[key]
            cells.append(f"{v:.2f}" if key in ("latency_ms", "cost_usd") else f"{v:.3f}")
        lines.append(f"| {name} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_class_table(results: dict[str, dict]) -> str:
    lines = ["| Model | Label | Precision | Recall |", "|---|---|---:|---:|"]
    for model, result in results.items():
        for label in LABELS:
            values = result["per_class"][label]
            lines.append(f"| {model} | {label} | {values['precision']:.3f} | {values['recall']:.3f} |")
    return "\n".join(lines)


def _write_charts(
    preds: dict[str, list[dict]], results: dict[str, dict],
    gold: dict[str, dict], messages: list[dict], tau: float,
) -> None:
    charts.plot_model_comparison(frames.metrics_frame(results)).save(
        str(REPORT_DIR / "model_comparison.png"), ppi=PPI)
    for name, rows in preds.items():
        charts.plot_confusion_matrix(
            frames.message_frame(rows, gold, messages, tau), title=f"{name} — τ = {tau:.2f}"
        ).save(str(REPORT_DIR / f"confusion_{name}.png"), ppi=PPI)
        charts.plot_distribution_comparison(
            frames.distribution_frame(rows, gold, tau), title=f"{name} — label distribution vs gold"
        ).save(str(REPORT_DIR / f"distribution_{name}.png"), ppi=PPI)
        charts.plot_tau_sweep(
            frames.sweep_table(rows, gold), tau=tau, title=f"{name} — τ sweep"
        ).save(str(REPORT_DIR / f"tau_sweep_{name}.png"), ppi=PPI)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score prediction files against gold labels.")
    ap.add_argument("--tau", type=float, help="also score at this explicit τ")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--window", type=int, help="window size whose gold labels to score against")
    args = ap.parse_args()

    messages = io.load_messages()
    available_windows = io.consensus_windows()
    if args.window is not None:
        gold_sets = [(args.window, io.load_consensus(window=args.window))]
    elif 4 in available_windows:
        gold_sets = [(4, io.load_consensus(window=4))]
        if 1 in available_windows:
            gold_sets.append((1, io.load_consensus(window=1)))
    elif len(available_windows) == 1:
        gold_sets = [(available_windows[0], io.load_consensus(window=available_windows[0]))]
    else:
        gold_sets = [(None, io.load_consensus())]

    gold_sets = [(window, frames.filter_split(messages, gold, args.split)) for window, gold in gold_sets]
    if not any(gold for _, gold in gold_sets):
        raise SystemExit(
            f"no gold labels for the messages in messages.jsonl (split={args.split}). "
            "Label, export, then run `pixi run label-consensus --window W`."
        )
    context_ids = io.context_sensitive_ids()

    models = io.list_prediction_models()
    if not models:
        raise SystemExit(
            "no prediction files in evaluation/predictions/ — run a model first, "
            "e.g. `pixi run eval-transformers` or `pixi run eval-nrc`."
        )

    preds = {name: io.load_predictions(name) for name in models}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tau_star_path = REPORT_DIR / "tau_star.json"
    tau_star = json.loads(tau_star_path.read_text(encoding="utf-8")) if tau_star_path.exists() else {}
    tau_values = [("τ = 0", 0.0)]
    if tau_star:
        tau_values.append(("τ = τ* (selected on dev)", None))
    if args.tau is not None and args.tau != 0.0:
        tau_values.append((f"τ = {args.tau:g} (requested)", args.tau))

    sections = ["# Evaluation report", "", f"split=`{args.split}` · primary gold is W=4 when available", ""]
    primary_results = primary_gold = None
    for window, gold in gold_sets:
        if not gold:
            continue
        window_name = f"W={window}" if window is not None else "sample/legacy gold"
        sections.extend([f"## {window_name}", "", f"Gold messages: {len(gold)}", ""])
        for tau_name, explicit_tau in tau_values:
            results = {
                name: evaluate_model(
                    rows, gold,
                    float(tau_star.get(name, 0.0)) if explicit_tau is None else explicit_tau,
                    context_ids,
                )
                for name, rows in preds.items()
            }
            sections.extend([f"### {tau_name}", "", _table(results), "", "Per-class precision and recall", "",
                             _per_class_table(results), ""])
            if window == 4 or (window is None and primary_results is None):
                if tau_name == "τ = 0":
                    primary_results, primary_gold = results, gold
        print(f"Scored {len(models)} model(s) on {window_name}, split={args.split}.")

    (REPORT_DIR / "metrics.md").write_text("\n".join(sections), encoding="utf-8")
    if primary_results is not None and primary_gold is not None:
        _write_charts(preds, primary_results, primary_gold, messages, 0.0)
    print(f"Wrote table and charts -> {REPORT_DIR.relative_to(io.ROOT.parent)}/")


if __name__ == "__main__":
    main()
