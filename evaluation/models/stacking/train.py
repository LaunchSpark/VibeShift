"""Train the Laya ⊕ NRC macro stacking head and write dev OOF + test rows.

The head uses Laya's seven probabilities concatenated with NRC's seven
``preceding_25`` probabilities. Hyperparameters are selected using dev only
(C in 0.01, 0.1, 1, 10; balanced and unweighted classes), then the chosen model
is refit on all dev for test scoring. Dev predictions use out-of-fold fits. For
the tiny sample fixture where stratification is impossible, deterministic
K-folds still provide out-of-fold predictions and the default C=1/unweighted
head is used because CV cannot be estimated reliably.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.common import io
from evaluation.common.io import normalize_probs, prediction_path, probs_vector
from evaluation.common.labels import LABELS
from evaluation.eval.metrics import macro_f1

MODEL_NAME = "stacking"
DEFAULT_FEATURES = ["laya", "nrc_macro"]
C_GRID = (0.01, 0.1, 1.0, 10.0)
RANDOM_STATE = 42


def _feature_rows(ids: list[str], feature_models: list[str]) -> tuple[np.ndarray, dict[str, dict[str, dict]]]:
    by_model: dict[str, dict[str, dict]] = {}
    for name in feature_models:
        if not prediction_path(name).exists():
            io._warn(f"stacking: feature model '{name}' has no predictions file — skipping it. "
                     "Run its `eval-*` task to include it.")
            continue
        by_model[name] = {row["id"]: row for row in io.load_predictions(name)}
    if not by_model:
        raise SystemExit("stacking: no feature models available — run eval-laya / eval-nrc first.")
    rows = []
    for message_id in ids:
        vector: list[float] = []
        for name in feature_models:
            if name in by_model:
                vector.extend(probs_vector(normalize_probs(by_model[name].get(message_id, {}).get("probs", {}))))
        rows.append(vector)
    return np.asarray(rows, dtype=float), by_model


def _folds(y: list[str]):
    counts = Counter(y)
    min_count = min(counts.values(), default=0)
    n_splits = min(5, len(y))
    if n_splits < 2:
        return [], False
    if min_count >= 2:
        n_splits = min(n_splits, min_count)
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y)), True
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)))), False


def _classifier(c: float, balanced: bool) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced" if balanced else None, max_iter=2000)


def _probabilities(model: LogisticRegression, x: np.ndarray) -> list[dict[str, float]]:
    matrix = model.predict_proba(x)
    class_index = {label: index for index, label in enumerate(model.classes_)}
    return [normalize_probs({label: float(row[class_index[label]]) if label in class_index else 0.0
                             for label in LABELS}) for row in matrix]


def _select_hyperparameters(x: np.ndarray, y: list[str], folds, stratified: bool) -> tuple[float, bool]:
    if not stratified or len(set(y)) < 2 or not folds:
        return 1.0, False
    best: tuple[float, float, bool] | None = None
    for c in C_GRID:
        for balanced in (False, True):
            truth: list[str] = []
            predicted: list[str] = []
            for train_index, valid_index in folds:
                model = _classifier(c, balanced).fit(x[train_index], [y[i] for i in train_index])
                probs = _probabilities(model, x[valid_index])
                predicted.extend(max(LABELS, key=p.__getitem__) for p in probs)
                truth.extend(y[i] for i in valid_index)
            score = macro_f1(truth, predicted)
            candidate = (score, -c, not balanced)
            if best is None or candidate > best:
                best = candidate
                chosen = (c, balanced)
    return chosen if best is not None else (1.0, False)


def _load_gold(window: int | None):
    if window is not None:
        return io.load_consensus(window=window)
    # Prefer the primary W=4 gold once it exists; otherwise use the sample or
    # the sole available consensus file for local development.
    if 4 in io.consensus_windows():
        return io.load_consensus(window=4)
    return io.load_consensus()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--window", type=int, help="gold window size (defaults to W=4 when available)")
    args = parser.parse_args()

    messages = io.load_messages()
    gold = _load_gold(args.window)
    dev = [m["id"] for m in messages if m.get("split") == "dev" and m["id"] in gold]
    test = [m["id"] for m in messages if m.get("split") == "test" and m["id"] in gold]
    if not dev or not test:
        raise SystemExit("stacking: need both dev and test gold messages to train/score.")

    x_dev, by_model = _feature_rows(dev, args.features)
    x_test, _ = _feature_rows(test, args.features)
    y_dev = [gold[message_id]["label"] for message_id in dev]
    folds, stratified = _folds(y_dev)
    c, balanced = _select_hyperparameters(x_dev, y_dev, folds, stratified)
    if stratified:
        print(f"[stacking] dev-only CV selected C={c:g}, class_weight={'balanced' if balanced else 'none'}, "
              f"folds={len(folds)}")
    else:
        print("[stacking] dev labels are too sparse for stratified CV; using C=1, class_weight=None")

    # A deterministic fallback is used when fewer than two dev examples exist.
    if len(set(y_dev)) < 2:
        raise SystemExit("stacking: dev needs at least two distinct gold labels to fit logistic regression.")

    oof: dict[str, tuple[dict[str, float], float]] = {}
    if not folds:
        folds = list(KFold(n_splits=min(2, len(dev)), shuffle=True, random_state=RANDOM_STATE).split(x_dev))
    for train_index, valid_index in folds:
        model = _classifier(c, balanced).fit(x_dev[train_index], [y_dev[i] for i in train_index])
        start = time.perf_counter()
        prob_rows = _probabilities(model, x_dev[valid_index])
        elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(valid_index), 1)
        for index, probs in zip(valid_index, prob_rows):
            oof[dev[index]] = (probs, elapsed_ms)

    final_model = _classifier(c, balanced).fit(x_dev, y_dev)
    start = time.perf_counter()
    test_probs = _probabilities(final_model, x_test)
    test_head_ms = (time.perf_counter() - start) * 1000 / max(len(test), 1)

    rows = []
    for ids, predictions, head_ms in ((dev, [oof[mid][0] for mid in dev], None),
                                      (test, test_probs, test_head_ms)):
        for index, (message_id, probs) in enumerate(zip(ids, predictions)):
            feature_latency = sum(by_model[name].get(message_id, {}).get("latency_ms", 0.0)
                                  for name in args.features if name in by_model)
            feature_cost = sum(by_model[name].get(message_id, {}).get("cost_usd", 0.0)
                               for name in args.features if name in by_model)
            model_ms = oof[message_id][1] if head_ms is None else head_ms
            rows.append({
                "id": message_id,
                "model": MODEL_NAME,
                "probs": probs,
                "raw_top_label": max(LABELS, key=probs.__getitem__),
                "latency_ms": round(feature_latency + model_ms, 4),
                "cost_usd": feature_cost,
            })
    io.write_predictions(MODEL_NAME, rows)
    print(f"[stacking] wrote {len(dev)} dev OOF + {len(test)} test rows; "
          f"features={args.features}")


if __name__ == "__main__":
    main()
