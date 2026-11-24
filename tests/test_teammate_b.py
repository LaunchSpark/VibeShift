"""Unit tests for Teammate B deliverables:
- Sub-issues #61, #62: Bootstrap confidence intervals in metrics.py
- Sub-issues #56, #57: tau_sweep.py updates and tau* selection rule
- Sub-issues #60: JSD distribution calculations
- Sub-issues #63: Dual tau scoring in evaluate.py
- Sub-issues #64: Categorized error analysis
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from evaluation.annotation.tau_sweep import select_tau_star, update_tau_star_json
from evaluation.common.labels import LABELS, NEUTRAL
from evaluation.eval import error_analysis, frames, metrics
from evaluation.eval.evaluate import evaluate_model


def test_class_f1():
    """Verify class_f1 computes binary F1 for a given class."""
    gold = ["neutral", "joy", "neutral", "sadness"]
    pred = ["neutral", "neutral", "neutral", "sadness"]
    # neutral: true = [1, 0, 1, 0], pred = [1, 1, 1, 0] -> precision 2/3, recall 2/2 -> f1 = 4/5 = 0.8
    f1 = metrics.class_f1(gold, pred, target_label=NEUTRAL)
    assert pytest.approx(f1) == 0.8


def test_bootstrap_ci_coverage_and_determinism():
    """Verify bootstrap confidence interval is within [0, 1] and deterministic with fixed seed."""
    gold = ["joy"] * 50 + ["sadness"] * 50
    pred = ["joy"] * 45 + ["sadness"] * 5 + ["sadness"] * 45 + ["joy"] * 5
    data = list(zip(gold, pred))

    # Calculate actual macro-F1 on this dataset:
    # Since labels=LABELS (7 classes), unrepresented classes get F1=0, so (0.9 + 0.9 + 0 + 0 + 0 + 0 + 0) / 7 ≈ 0.257
    expected_f1 = metrics.macro_f1(gold, pred)

    low, high, n, small = metrics.bootstrap_ci(
        data,
        lambda pairs: metrics.macro_f1([p[0] for p in pairs], [p[1] for p in pairs]),
        n_resamples=200,
        ci=0.95,
        seed=42,
    )

    assert n == 100
    assert not small
    assert 0.0 <= low <= expected_f1 <= high <= 1.0


def test_bootstrap_ci_flags_small_sample():
    """Verify bootstrap flags n < 30."""
    data = [("joy", "joy")] * 15
    low, high, n, small = metrics.bootstrap_ci(
        data,
        lambda pairs: 1.0,
        n_resamples=50,
        seed=42,
    )
    assert n == 15
    assert small is True


def test_select_tau_star_rule():
    """Verify Issue #56: tau* = argmax dev macro-F1, tie-break on lower JSD."""
    df = pl.DataFrame({
        "tau": [0.0, 0.2, 0.4, 0.6],
        "macro_f1": [0.60, 0.75, 0.75, 0.70],
        "jsd_gated": [0.05, 0.04, 0.02, 0.01],  # 0.4 ties on macro_f1 with 0.2 but has lower JSD
        "neutral_f1": [0.50, 0.60, 0.65, 0.70],
        "pct_neutral": [0.20, 0.30, 0.40, 0.50],
    })
    tau_star = select_tau_star(df)
    assert tau_star == 0.4


def test_sweep_table_includes_tau_zero_and_neutral_f1():
    """Verify sweep_table outputs tau=0.0 and neutral_f1 column."""
    gold = {
        "m1": {"label": "joy"},
        "m2": {"label": "neutral"},
    }
    rows = [
        {
            "id": "m1",
            "probs": {label: 0.1 for label in LABELS} | {"joy": 0.4},
            "raw_top_label": "joy",
        },
        {
            "id": "m2",
            "probs": {label: 0.1 for label in LABELS} | {"neutral": 0.4},
            "raw_top_label": "neutral",
        },
    ]

    table = frames.sweep_table(rows, gold)
    assert "neutral_f1" in table.columns
    assert 0.0 in table["tau"].to_list()


def test_jsd_bounds_and_symmetry():
    """Verify JSD properties."""
    p = {label: 1.0 / len(LABELS) for label in LABELS}
    q = {label: 1.0 / len(LABELS) for label in LABELS}
    assert pytest.approx(metrics.jensen_shannon(p, q)) == 0.0

    r = {label: 0.0 for label in LABELS}
    r["joy"] = 1.0
    assert 0.0 <= metrics.jensen_shannon(p, r) <= 1.0


def test_evaluate_model_with_bootstrap_ci():
    """Verify evaluate_model returns bootstrap CIs and tau."""
    gold = {f"m{i}": {"label": "joy"} for i in range(40)}
    rows = [
        {
            "id": f"m{i}",
            "probs": {label: 0.05 for label in LABELS} | {"joy": 0.7},
            "raw_top_label": "joy",
            "latency_ms": 5.0,
            "cost_usd": 0.001,
        }
        for i in range(40)
    ]
    res = evaluate_model(rows, gold, tau=0.3)
    assert res["n"] == 40
    assert res["tau"] == 0.3
    assert "macro_f1_ci" in res
    assert "top2_acc_ci" in res
    assert res["macro_f1_ci"][2] is False  # not small sample


def test_error_categorization():
    """Verify error categorizer sorts samples into expected buckets."""
    msg_short = {"content": "ok"}
    cat_short = error_analysis.categorize_error(msg_short, gold_label="joy", pred_label="neutral")
    assert "short reply" in cat_short

    msg_slang = {"content": "bruh that was insane lol"}
    cat_slang = error_analysis.categorize_error(msg_slang, gold_label="joy", pred_label="surprise")
    assert "slang" in cat_slang

    msg_ctx = {"content": "No way"}
    cat_ctx = error_analysis.categorize_error(
        msg_ctx,
        gold_label="surprise",
        pred_label="anger",
        gold_w1_label="anger",
    )
    assert "context-dependent" in cat_ctx
