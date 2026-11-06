"""Build gold labels for one window size from everyone's exported label files.

    pixi run label-consensus --window 5

Reads ``annotations/labels/*_w{W}.jsonl`` and keeps the messages *every* labeler
has labeled (the shared prefix of the queue). The gold label is the strict
majority; a tie is resolved by ``annotations/overrides_w{W}.json``
(``{"<id>": "<label>"}``, committed) or listed for adjudication and left out.
``soft_probs`` is the share of labelers who chose each label, before any override.
Writes ``annotations/consensus_w{W}.jsonl`` (Contract 1B).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from evaluation.common import io
from evaluation.common.labels import LABELS


def build_consensus(
    by_labeler: dict[str, dict[str, str]], overrides: dict[str, str]
) -> tuple[list[dict], list[str]]:
    """(gold rows, unresolved tie ids) over the ids every labeler has labeled."""
    if not by_labeler:
        return [], []
    shared = set.intersection(*(set(labels) for labels in by_labeler.values()))
    rows, ties = [], []
    for mid in sorted(shared):
        votes = Counter(labels[mid] for labels in by_labeler.values())
        n = sum(votes.values())
        ranked = votes.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            label = ranked[0][0]
        elif mid in overrides:
            label = overrides[mid]
        else:
            ties.append(mid)
            continue
        rows.append({
            "id": mid,
            "label": label,
            "soft_probs": {k: votes.get(k, 0) / n for k in LABELS},
        })
    return rows, ties


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, required=True, help="window size to build gold for")
    args = ap.parse_args()

    by_labeler = io.load_label_files(args.window)
    if len(by_labeler) < 2:
        raise SystemExit(
            f"need label files from at least 2 labelers at window size {args.window} "
            f"(found {sorted(by_labeler) or 'none'} in {io.LABEL_FILES})."
        )
    ov_path = io.overrides_path(args.window)
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    bad = {k: v for k, v in overrides.items() if v not in LABELS}
    if bad:
        raise SystemExit(f"{ov_path.name} has labels outside LABELS: {bad}")

    rows, ties = build_consensus(by_labeler, overrides)
    out = io.consensus_path(args.window)
    io.write_jsonl(out, rows)
    counts = ", ".join(f"{k}={len(v)}" for k, v in sorted(by_labeler.items()))
    print(f"window {args.window}: labelers {counts}; {len(rows)} gold rows -> {out.name}")
    if ties:
        print(f"\n{len(ties)} ties to adjudicate — add them to {ov_path.name} as "
              '{"<id>": "<label>"} and re-run:')
        for mid in ties:
            votes = {lab: labels[mid] for lab, labels in sorted(by_labeler.items())}
            print(f"  {mid}: {votes}")


if __name__ == "__main__":
    main()
