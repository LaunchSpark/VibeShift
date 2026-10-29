"""Lightweight stacking head (owned by Teammate D).

Trains a ``LogisticRegression`` on the 75-message ``dev`` split and scores the
225-message ``test`` split. Input features per message are the concatenation of
the feature models' 7-class probability vectors — by default **Laya** (local
sequence probabilities) and **nrc_macro** (the M=25 macro-window mood vector).

Prerequisite prediction files (run these first):
    pixi run eval-laya        # -> predictions/laya.jsonl   (Lucas / heavy model)
    pixi run eval-nrc --role macro   # -> predictions/nrc_macro.jsonl

If a feature model's file is missing, it is skipped with a warning so D can
prototype on whatever is available today (e.g. nrc_macro alone on the sample).

    pixi run eval-stack
    python -m evaluation.models.stacking.train --features laya nrc_macro
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluation.common import io
from evaluation.common.labels import LABELS
from evaluation.common.io import prediction_path, probs_vector, normalize_probs

MODEL_NAME = "stacking"
DEFAULT_FEATURES = ["laya", "nrc_macro"]


def _feature_matrix(ids: list[str], feature_models: list[str]) -> np.ndarray:
    """Per-id feature vector: each present model's 7 probs, concatenated."""
    by_model = {}
    for name in feature_models:
        if not prediction_path(name).exists():
            io._warn(f"stacking: feature model '{name}' has no predictions file — skipping it. "
                     f"Run its `eval-*` task to include it.")
            continue
        by_model[name] = {r["id"]: r["probs"] for r in io.load_predictions(name)}
    if not by_model:
        raise SystemExit("stacking: no feature models available — run eval-laya / eval-nrc first.")

    rows = []
    for i in ids:
        vec: list[float] = []
        for name in feature_models:
            if name in by_model:
                vec.extend(probs_vector(normalize_probs(by_model[name].get(i, {}))))
        rows.append(vec)
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = ap.parse_args()

    messages = io.load_messages()
    gold = io.load_consensus()
    dev = [m["id"] for m in messages if m.get("split") == "dev" and m["id"] in gold]
    test = [m["id"] for m in messages if m.get("split") == "test" and m["id"] in gold]
    if not dev or not test:
        raise SystemExit("stacking: need both dev and test gold messages to train/score.")

    x_dev = _feature_matrix(dev, args.features)
    x_test = _feature_matrix(test, args.features)
    y_dev = [gold[i]["label"] for i in dev]

    # multinomial is the lbfgs default (the explicit multi_class kwarg was removed in sklearn 1.7).
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_dev, y_dev)

    proba = clf.predict_proba(x_test)  # columns follow clf.classes_
    class_index = {c: j for j, c in enumerate(clf.classes_)}

    rows = []
    for i, p in zip(test, proba):
        probs = {label: float(p[class_index[label]]) if label in class_index else 0.0
                 for label in LABELS}
        probs = normalize_probs(probs)
        rows.append(
            {
                "id": i,
                "model": MODEL_NAME,
                "probs": probs,
                "raw_top_label": max(probs, key=probs.get),
                "latency_ms": 0.0,
                "cost_usd": 0.0,
            }
        )
    io.write_predictions(MODEL_NAME, rows)
    print(f"[stacking] trained on {len(dev)} dev, scored {len(test)} test, "
          f"features={args.features}")


if __name__ == "__main__":
    main()
