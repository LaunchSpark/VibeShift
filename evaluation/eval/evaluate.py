"""Score every model in ``predictions/`` against the gold consensus labels.

Produces the final report table (Macro-F1 overall + sliced by
``is_context_dependent`` / ``is_sarcastic``, Top-2 accuracy, multi-class Brier,
log-loss, JSD, latency, cost) and the report charts: a model comparison, plus a
confusion matrix, label-distribution comparison, and τ sweep per model. All
metrics are schema-ordered (see ``metrics.py``); charts are Altair, rendered to
PNG headlessly by vl-convert-python. For interactive drill-down, use
``pixi run eval-notebook``.

Run:
    pixi run eval-report                  # default τ (VIBESHIFT_TAU or 0.0), test split
    python -m evaluation.eval.evaluate --tau 0.3 --split test
"""

from __future__ import annotations

import argparse
import os

from evaluation.common import io
from evaluation.eval import charts, frames, metrics

REPORT_DIR = io.ROOT / "eval" / "report"
PPI = 200


def _default_tau() -> float:
    try:
        return float(os.environ.get("VIBESHIFT_TAU", "0.0"))
    except ValueError:
        return 0.0


def evaluate_model(rows: list[dict], gold: dict[str, dict], tau: float) -> dict:
    """Metrics for one model over the messages it and the gold set share."""
    rows = [r for r in rows if r["id"] in gold]
    ids = [r["id"] for r in rows]
    probs_list = [r["probs"] for r in rows]
    gold_labels = [gold[i]["label"] for i in ids]
    pred_labels = [metrics.gated_label(r["probs"], r["raw_top_label"], tau) for r in rows]

    def sliced(flag: str) -> float:
        g = [gl for gl, i in zip(gold_labels, ids) if gold[i].get(flag)]
        p = [pl for pl, i in zip(pred_labels, ids) if gold[i].get(flag)]
        return metrics.macro_f1(g, p) if g else float("nan")

    # JSD: predicted distribution vs gold, two ways.
    pred_mean = metrics.mean_distribution(probs_list)
    gold_mix = metrics.label_mix(gold_labels)
    gated_mix = metrics.label_mix(pred_labels)

    return {
        "n": len(rows),
        "macro_f1": metrics.macro_f1(gold_labels, pred_labels),
        "macro_f1_context": sliced("is_context_dependent"),
        "macro_f1_sarcastic": sliced("is_sarcastic"),
        "top2_acc": metrics.top_k_accuracy(gold_labels, probs_list, k=2),
        "brier": metrics.brier_score(gold_labels, probs_list),
        "log_loss": metrics.log_loss(gold_labels, probs_list),
        "jsd_mean_probs_vs_gold": metrics.jensen_shannon(pred_mean, gold_mix),
        "jsd_gated_counts_vs_gold": metrics.jensen_shannon(gated_mix, gold_mix),
        "latency_ms": sum(r["latency_ms"] for r in rows) / max(len(rows), 1),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
    }


_COLUMNS = [
    ("macro_f1", "Macro-F1"),
    ("macro_f1_context", "Macro-F1 (ctx)"),
    ("macro_f1_sarcastic", "Macro-F1 (sarc)"),
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
    ap.add_argument("--tau", type=float, default=_default_tau(), help="τ cutoff (default VIBESHIFT_TAU or 0.0)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = ap.parse_args()

    messages = io.load_messages()
    gold = frames.filter_split(messages, io.load_consensus(), args.split)

    models = io.list_prediction_models()
    if not models:
        raise SystemExit(
            "no prediction files in evaluation/predictions/ — run a model first, "
            "e.g. `pixi run eval-transformers` or `pixi run eval-nrc`."
        )

    print(f"Scoring {len(models)} model(s) on split={args.split}, τ={args.tau}, "
          f"{len(gold)} gold messages.\n")
    preds = {name: io.load_predictions(name) for name in models}
    results = {name: evaluate_model(rows, gold, args.tau) for name, rows in preds.items()}

    table = _table(results)
    print(table + "\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.md").write_text(
        f"# Evaluation report\n\nsplit=`{args.split}`, τ=`{args.tau}`, "
        f"gold messages=`{len(gold)}`\n\n{table}\n"
    )
    _write_charts(preds, results, gold, messages, args.tau)
    print(f"Wrote table and charts -> {REPORT_DIR.relative_to(io.ROOT.parent)}/")


if __name__ == "__main__":
    main()
