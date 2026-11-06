"""Inter-annotator agreement for one window size.

Reads every labeler's ``annotations/labels/*_w{W}.jsonl`` and reports Cohen's κ
for each pair of labelers (on the messages both labeled) and Fleiss' κ across
everyone (on the messages all of them labeled), then lists the messages where
labelers disagree — the adjudication worklist for ``label-consensus``.

    pixi run eval-kappa --window 5
"""

from __future__ import annotations

import argparse
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

from evaluation.common import io
from evaluation.common.labels import LABELS


def fleiss_kappa(counts: list[list[int]]) -> float:
    """Fleiss' κ. Rows are items, columns categories; every row sums to the same rater count."""
    n_items = len(counts)
    raters = sum(counts[0])
    total = n_items * raters
    p_cat = [sum(row[j] for row in counts) / total for j in range(len(counts[0]))]
    p_item = [(sum(c * c for c in row) - raters) / (raters * (raters - 1)) for row in counts]
    p_bar = sum(p_item) / n_items
    p_e = sum(p * p for p in p_cat)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def pairwise_cohen(by_labeler: dict[str, dict[str, str]]) -> dict[tuple[str, str], float]:
    """Cohen's κ for every pair of labelers, on the ids both labeled."""
    out = {}
    for a, b in combinations(sorted(by_labeler), 2):
        ids = sorted(set(by_labeler[a]) & set(by_labeler[b]))
        if ids:
            out[(a, b)] = cohen_kappa_score(
                [by_labeler[a][i] for i in ids], [by_labeler[b][i] for i in ids], labels=list(LABELS)
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, required=True, help="window size to measure")
    args = ap.parse_args()

    by_labeler = io.load_label_files(args.window)
    if len(by_labeler) < 2:
        raise SystemExit(
            f"need label files from at least 2 labelers at window size {args.window} "
            f"(found {sorted(by_labeler) or 'none'} in {io.LABEL_FILES})."
        )

    print(f"window size {args.window}")
    for (a, b), kappa in pairwise_cohen(by_labeler).items():
        shared = len(set(by_labeler[a]) & set(by_labeler[b]))
        print(f"  Cohen's κ {a} vs {b}: {kappa:.3f}  ({shared} shared messages)")

    shared = sorted(set.intersection(*(set(v) for v in by_labeler.values())))
    if not shared:
        raise SystemExit("no message has been labeled by every labeler yet.")
    table = [[sum(labels[i] == k for labels in by_labeler.values()) for k in LABELS] for i in shared]
    print(f"  Fleiss' κ ({len(by_labeler)} labelers, {len(shared)} messages): {fleiss_kappa(table):.3f}")

    disagree = [i for i in shared if len({labels[i] for labels in by_labeler.values()}) > 1]
    print(f"\n{len(disagree)} messages with disagreement:")
    for i in disagree:
        print(f"  {i}: " + ", ".join(f"{lab}={labels[i]}" for lab, labels in sorted(by_labeler.items())))


if __name__ == "__main__":
    main()
