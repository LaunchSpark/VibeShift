"""Schema-ordered classification and distribution metrics.

Implemented fresh here (there was no prior ``finetune_laya_emotions.py`` to port
from). Every function indexes probabilities in canonical ``LABELS`` order, so a
model's prediction file and the gold labels always line up on the same axes.

Metrics
-------
* ``log_loss``        — multi-class cross-entropy of gold vs predicted probs
* ``brier_score``     — multi-class Brier (mean squared error vs the one-hot gold)
* ``top_k_accuracy``  — gold label among the model's top-k probs (Top-2 by default)
* ``macro_f1``        — unweighted mean per-class F1 (scikit-learn, labels=LABELS)
* ``jensen_shannon``  — JSD between two distributions over LABELS

The τ rule (``gated_label``) is shared by the report and the τ sweep.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score

from evaluation.common.labels import INDEX, LABELS, NEUTRAL
from evaluation.common.io import normalize_probs, probs_vector

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# the τ rule (see schema.md — identical in evaluate.py and tau_sweep.py)
# --------------------------------------------------------------------------- #
def gated_label(probs: dict[str, float], raw_top_label: str, tau: float) -> str:
    """Apply the universal τ cutoff to one prediction.

    Predict ``neutral`` if the raw top label is neutral or the strongest
    non-neutral probability is below τ; otherwise keep the raw top label.
    τ = 0.0 reduces to plain argmax.
    """
    if raw_top_label == NEUTRAL:
        return NEUTRAL
    non_neutral = max(p for e, p in probs.items() if e != NEUTRAL)
    if non_neutral < tau:
        return NEUTRAL
    return raw_top_label


# --------------------------------------------------------------------------- #
# probabilistic metrics (gold = hard one-hot label)
# --------------------------------------------------------------------------- #
def _onehot(label: str) -> np.ndarray:
    v = np.zeros(len(LABELS))
    v[INDEX[label]] = 1.0
    return v


def log_loss(gold_labels: list[str], probs_list: list[dict[str, float]]) -> float:
    total = 0.0
    for label, probs in zip(gold_labels, probs_list):
        p = probs_vector(normalize_probs(probs))[INDEX[label]]
        total += -math.log(max(p, _EPS))
    return total / len(gold_labels)


def brier_score(gold_labels: list[str], probs_list: list[dict[str, float]]) -> float:
    total = 0.0
    for label, probs in zip(gold_labels, probs_list):
        p = np.array(probs_vector(normalize_probs(probs)))
        total += float(np.sum((p - _onehot(label)) ** 2))
    return total / len(gold_labels)


def top_k_accuracy(gold_labels: list[str], probs_list: list[dict[str, float]], k: int = 2) -> float:
    hits = 0
    for label, probs in zip(gold_labels, probs_list):
        order = sorted(LABELS, key=lambda e: probs.get(e, 0.0), reverse=True)
        if label in order[:k]:
            hits += 1
    return hits / len(gold_labels)


def macro_f1(gold_labels: list[str], pred_labels: list[str]) -> float:
    if not gold_labels:
        return float("nan")
    return float(
        f1_score(gold_labels, pred_labels, labels=list(LABELS), average="macro", zero_division=0)
    )


# --------------------------------------------------------------------------- #
# distribution metrics
# --------------------------------------------------------------------------- #
def jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    """JSD (base-2, in [0, 1]) between two distributions over LABELS."""
    pv = np.array(probs_vector(normalize_probs(p)))
    qv = np.array(probs_vector(normalize_probs(q)))
    m = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], _EPS))))

    return 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)


def label_mix(labels: list[str]) -> dict[str, float]:
    """Empirical label-frequency distribution over LABELS."""
    counts = {label: 0.0 for label in LABELS}
    for label in labels:
        counts[label] += 1.0
    n = max(len(labels), 1)
    return {label: c / n for label, c in counts.items()}


def mean_distribution(probs_list: list[dict[str, float]]) -> dict[str, float]:
    """Average predicted distribution over a set of messages."""
    acc = {label: 0.0 for label in LABELS}
    for probs in probs_list:
        for label, p in normalize_probs(probs).items():
            acc[label] += p
    n = max(len(probs_list), 1)
    return {label: v / n for label, v in acc.items()}
