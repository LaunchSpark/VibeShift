"""Score every model in ``predictions/`` against the gold consensus labels.

Produces the final report table (Macro-F1 overall + on context-sensitive
messages, Top-2 accuracy, multi-class Brier,
log-loss, JSD, latency, cost) and the report charts: a model comparison, plus a
confusion matrix, label-distribution comparison, and τ sweep per model. All
metrics are schema-ordered (see ``metrics.py``); charts are Altair, rendered to
PNG headlessly by vl-convert-python. For interactive drill-down, use
``pixi run eval-notebook``.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from sklearn.metrics import precision_recall_fscore_support

from evaluation.common import io
from evaluation.common.labels import LABELS
from evaluation.eval import charts, frames, metrics

REPORT_DIR = io.ROOT / "eval" / "report"
PPI = 200


def _default_tau() -> float:
    try:
        return float(os.environ.get("VIBESHIFT_TAU", "0.0"))
    except ValueError:
        return 0.0


def _load_tau_stars() -> dict[str, float]:
    path = REPORT_DIR / "tau_star.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


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

    # Bootstrap 95% Confidence Intervals for headline metrics
    f1_low, f1_high, _, f1_small = metrics.bootstrap_ci(
        list(zip(gold_labels, pred_labels)),
        lambda pairs: metrics.macro_f1([p[0] for p in pairs], [p[1] for p in pairs])
    )

    top2_low, top2_high, _, top2_small = metrics.bootstrap_ci(
        list(zip(gold_labels, probs_list)),
        lambda pairs: metrics.top_k_accuracy([p[0] for p in pairs], [p[1] for p in pairs], k=2)
    )

    if context_ids is not None:
        ctx_pairs = [(gl, pl) for gl, pl, i in zip(gold_labels, pred_labels, ids) if i in context_ids]
        ctx_low, ctx_high, ctx_n, ctx_small = metrics.bootstrap_ci(
            ctx_pairs,
            lambda pairs: metrics.macro_f1([p[0] for p in pairs], [p[1] for p in pairs])
        )
    else:
        ctx_low, ctx_high, ctx_n, ctx_small = float("nan"), float("nan"), 0, True

    precision, recall, _, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=list(LABELS), zero_division=0
    )
    per_class = {
        label: {"precision": float(p), "recall": float(r)}
        for label, p, r in zip(LABELS, precision, recall)
    }

    return {
        "n": len(rows),
        "tau": tau,
        "macro_f1": metrics.macro_f1(gold_labels, pred_labels),
        "macro_f1_ci": (f1_low, f1_high, f1_small),
        "macro_f1_context": sliced(context_ids),
        "macro_f1_context_ci": (ctx_low, ctx_high, ctx_n, ctx_small),
        "top2_acc": metrics.top_k_accuracy(gold_labels, probs_list, k=2),
        "top2_acc_ci": (top2_low, top2_high, top2_small),
        "brier": metrics.brier_score(gold_labels, probs_list),
        "log_loss": metrics.log_loss(gold_labels, probs_list),
        "jsd_mean_probs_vs_gold": metrics.jensen_shannon(pred_mean, gold_mix),
        "jsd_gated_counts_vs_gold": metrics.jensen_shannon(gated_mix, gold_mix),
        "latency_ms": sum(r.get("latency_ms", 0.0) for r in rows) / max(len(rows), 1),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
        "per_class": per_class,
    }


_COLUMNS = [
    ("macro_f1", "Macro-F1 (95% CI)"),
    ("macro_f1_context", "Macro-F1 ctx (95% CI)"),
    ("top2_acc", "Top-2 (95% CI)"),
    ("brier", "Brier"),
    ("log_loss", "LogLoss"),
    ("jsd_mean_probs_vs_gold", "JSD probs"),
    ("jsd_gated_counts_vs_gold", "JSD gated"),
    ("latency_ms", "ms/msg"),
    ("cost_usd", "$ total"),
]


def _format_cell(r: dict, key: str) -> str:
    if key == "macro_f1":
        low, high, small = r.get("macro_f1_ci", (float("nan"), float("nan"), False))
        flag = " *" if small else ""
        return f"{r['macro_f1']:.3f} [{low:.2f}, {high:.2f}]{flag}"
    if key == "macro_f1_context":
        v = r.get("macro_f1_context", float("nan"))
        if math.isnan(v):
            return "NaN"
        low, high, n, small = r.get("macro_f1_context_ci", (float("nan"), float("nan"), 0, False))
        flag = f" (n={n})*" if small else f" (n={n})"
        return f"{v:.3f} [{low:.2f}, {high:.2f}]{flag}"
    if key == "top2_acc":
        low, high, small = r.get("top2_acc_ci", (float("nan"), float("nan"), False))
        flag = " *" if small else ""
        return f"{r['top2_acc']:.3f} [{low:.2f}, {high:.2f}]{flag}"
    if key in ("latency_ms", "cost_usd"):
        return f"{r[key]:.2f}"
    return f"{r[key]:.3f}"


def _table(results: dict[str, dict]) -> str:
    head = "| Model | τ | n | " + " | ".join(h for _, h in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 3)
    lines = [head, sep]
    for name, r in results.items():
        cells = [_format_cell(r, key) for key, _ in _COLUMNS]
        tau_val = f"{r.get('tau', 0.0):.2f}"
        lines.append(f"| {name} | {tau_val} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_class_table(results: dict[str, dict]) -> str:
    headers = ["Model", "Metric", *LABELS]
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "---|" * len(headers)
    lines = [head, sep]
    for name, result in results.items():
        precisions = [f"{result['per_class'][label]['precision']:.3f}" for label in LABELS]
        recalls = [f"{result['per_class'][label]['recall']:.3f}" for label in LABELS]
        lines.append(f"| {name} | Precision | " + " | ".join(precisions) + " |")
        lines.append(f"| {name} | Recall | " + " | ".join(recalls) + " |")
    return "\n".join(lines)


def _write_charts(
    preds: dict[str, list[dict]], results: dict[str, dict],
    gold: dict[str, dict], messages: list[dict], tau: float,
) -> None:
    charts.save_chart(
        charts.plot_model_comparison(frames.metrics_frame(results)),
        REPORT_DIR / "model_comparison.png",
        ppi=PPI,
    )
    for name, rows in preds.items():
        charts.save_chart(
            charts.plot_confusion_matrix(
                frames.message_frame(rows, gold, messages, tau), title=f"{name} — τ = {tau:.2f}"
            ),
            REPORT_DIR / f"confusion_{name}.png",
            ppi=PPI,
        )
        charts.save_chart(
            charts.plot_distribution_comparison(
                frames.distribution_frame(rows, gold, tau), title=f"{name} — label distribution vs gold"
            ),
            REPORT_DIR / f"distribution_{name}.png",
            ppi=PPI,
        )
        charts.save_chart(
            charts.plot_tau_sweep(
                frames.sweep_table(rows, gold), tau=tau, title=f"{name} — τ sweep"
            ),
            REPORT_DIR / f"tau_sweep_{name}.png",
            ppi=PPI,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score prediction files against gold labels.")
    ap.add_argument("--tau", type=float, default=_default_tau(), help="τ cutoff (default VIBESHIFT_TAU or 0.0)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--window", type=int, help="window size whose gold labels to score against")
    ap.add_argument("--with-tau-star", action="store_true", default=True,
                    help="also score each model at its optimal τ* from tau_star.json")
    args = ap.parse_args()

    messages = io.load_messages()
    windows = [args.window] if args.window is not None else (io.consensus_windows() or [None])
    gold_sets = [(w, frames.filter_split(messages, io.load_consensus(window=w), args.split)) for w in windows]
    gold_sets = [(w, g) for w, g in gold_sets if g]
    if not gold_sets:
        raise SystemExit(
            f"no gold labels for the messages in messages.jsonl (split={args.split})."
        )

    context_ids = io.context_sensitive_ids()
    models = io.list_prediction_models()
    if not models:
        raise SystemExit(
            "no prediction files in evaluation/predictions/ — run a model first."
        )

    preds = {name: io.load_predictions(name) for name in models}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tau_stars = _load_tau_stars() if args.with_tau_star else {}

    sections = ["# Evaluation report", "", f"split=`{args.split}`", ""]
    primary_results = primary_gold = None

    for window, gold in gold_sets:
        w_title = f"Window W={window}" if window is not None else "Sample Gold"
        sections.extend([f"## {w_title}", "", f"Gold messages: {len(gold)}", ""])

        results = {}
        for name, rows in preds.items():
            results[f"{name} (τ={args.tau:.2f})"] = evaluate_model(rows, gold, args.tau, context_ids)
            if args.with_tau_star and name in tau_stars:
                t_star = float(tau_stars[name])
                if abs(t_star - args.tau) > 1e-4:
                    results[f"{name} (τ*={t_star:.2f})"] = evaluate_model(rows, gold, t_star, context_ids)

        sections.extend([_table(results), "", "### Per-class precision and recall", "", _per_class_table(results), ""])
        if primary_results is None:
            primary_results, primary_gold = results, gold
        print(f"Scored {len(models)} model(s) on {w_title}, split={args.split}.")

    (REPORT_DIR / "metrics.md").write_text("\n".join(sections), encoding="utf-8")
    if primary_results is not None and primary_gold is not None:
        try:
            _write_charts(preds, primary_results, primary_gold, messages, args.tau)
        except Exception as e:
            print(f"Chart render notice: {e}")
    print(f"Wrote table and metrics -> {REPORT_DIR.relative_to(io.ROOT.parent)}/")


if __name__ == "__main__":
    main()
