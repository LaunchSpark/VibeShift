"""Shape prediction files into polars DataFrames for the charts and the notebook.

``eval-report``, ``eval-tau``, and the explorer notebook all build their tables
here, so a chart in the notebook and the PNG in the report come from the same data.
"""

from __future__ import annotations

import polars as pl

from evaluation.common.labels import LABELS, NEUTRAL
from evaluation.eval import metrics

# τ grid for the sweep chart and table: 0.15, 0.20, ..., 0.85.
TAU_GRID = tuple(round(0.15 + 0.05 * i, 2) for i in range(15))


def filter_split(messages: list[dict], gold: dict[str, dict], split: str) -> dict[str, dict]:
    """Gold rows restricted to one split ('dev', 'test', or 'all')."""
    if split == "all":
        return gold
    keep = {m["id"] for m in messages if m.get("split") == split}
    return {i: g for i, g in gold.items() if i in keep}


def _aligned(rows: list[dict], gold: dict[str, dict]) -> list[dict]:
    return [r for r in rows if r["id"] in gold]


def message_frame(
    rows: list[dict], gold: dict[str, dict], messages: list[dict], tau: float
) -> pl.DataFrame:
    """One row per gold message a model predicted, with its τ-gated label.

    This is the confusion matrix's data: clicking a cell filters these rows by
    ``true_label`` / ``pred_label`` to show the exact messages behind it.
    """
    by_id = {m["id"]: m for m in messages}
    records = []
    for r in _aligned(rows, gold):
        g, m = gold[r["id"]], by_id.get(r["id"], {})
        pred = metrics.gated_label(r["probs"], r["raw_top_label"], tau)
        records.append(
            {
                "id": r["id"],
                "content": m.get("content", ""),
                "preceding_3": " | ".join(m.get("preceding_3", [])),
                "true_label": g["label"],
                "pred_label": pred,
                "correct": pred == g["label"],
                "is_sarcastic": bool(g.get("is_sarcastic")),
                "is_context_dependent": bool(g.get("is_context_dependent")),
            }
        )
    return pl.DataFrame(records, schema={
        "id": pl.String, "content": pl.String, "preceding_3": pl.String,
        "true_label": pl.String, "pred_label": pl.String, "correct": pl.Boolean,
        "is_sarcastic": pl.Boolean, "is_context_dependent": pl.Boolean,
    })


_COMPARISON_METRICS = {
    "macro_f1": "Macro-F1",
    "top2_acc": "Top-2 accuracy",
    "brier": "Brier (lower is better)",
}


def metrics_frame(results: dict[str, dict]) -> pl.DataFrame:
    """Long table (model, metric, value) of the headline metrics per model."""
    return pl.DataFrame(
        [
            {"model": model, "metric": label, "value": r[key]}
            for model, r in results.items()
            for key, label in _COMPARISON_METRICS.items()
        ]
    )


def sweep_table(
    rows: list[dict], gold: dict[str, dict], taus: tuple[float, ...] = TAU_GRID
) -> pl.DataFrame:
    """Macro-F1, JSD(τ-gated counts vs gold), and % neutral at each τ."""
    rows = _aligned(rows, gold)
    gold_labels = [gold[r["id"]]["label"] for r in rows]
    gold_mix = metrics.label_mix(gold_labels)
    records = []
    for tau in taus:
        preds = [metrics.gated_label(r["probs"], r["raw_top_label"], tau) for r in rows]
        records.append(
            {
                "tau": float(tau),
                "macro_f1": metrics.macro_f1(gold_labels, preds),
                "jsd_gated": metrics.jensen_shannon(metrics.label_mix(preds), gold_mix),
                "pct_neutral": sum(p == NEUTRAL for p in preds) / max(len(preds), 1),
            }
        )
    return pl.DataFrame(records)


def distribution_frame(rows: list[dict], gold: dict[str, dict], tau: float) -> pl.DataFrame:
    """Gold label mix vs mean predicted probabilities vs τ-gated counts.

    Source names carry each distribution's JSD to the gold mix, so the chart
    legend shows how much the τ gate closes the gap.
    """
    rows = _aligned(rows, gold)
    gold_mix = metrics.label_mix([gold[r["id"]]["label"] for r in rows])
    mean_probs = metrics.mean_distribution([r["probs"] for r in rows])
    gated = metrics.label_mix(
        [metrics.gated_label(r["probs"], r["raw_top_label"], tau) for r in rows]
    )
    sources = {
        "Gold labels": gold_mix,
        f"Mean probabilities (JSD {metrics.jensen_shannon(mean_probs, gold_mix):.3f})": mean_probs,
        f"τ-gated counts (JSD {metrics.jensen_shannon(gated, gold_mix):.3f})": gated,
    }
    return pl.DataFrame(
        [
            {"label": label, "source": source, "share": dist[label]}
            for source, dist in sources.items()
            for label in LABELS
        ]
    )
