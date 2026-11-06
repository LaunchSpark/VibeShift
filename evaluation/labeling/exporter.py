"""Export your labels from local PocketBase so the team can merge them.

    pixi run label-export

Writes one file per (labeler, window size) to
``evaluation/data/annotations/labels/{labeler}_w{W}.jsonl``, sorted by queue
position. Commit your files: each person only writes their own, so pulls never
conflict, and the consensus step lines everyone up by message id.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import httpx

from evaluation.common import io

from .importer import DEFAULT_PB, default_pb_url, fetch_all

_FIELDS = "message_id,labeler,window_size,label,queue_position,algorithm_version"


def export_labels(pb_url: str, out_dir: Path = io.LABEL_FILES) -> list[Path]:
    with httpx.Client(base_url=pb_url.rstrip("/"), timeout=30) as client:
        records = fetch_all(client, "labels", _FIELDS)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        size = int(r["window_size"])
        groups[(r["labeler"], size)].append({
            "id": r["message_id"],
            "labeler": r["labeler"],
            "window_size": size,
            "label": r["label"],
            "queue_position": int(r["queue_position"]),
            "algorithm_version": int(r["algorithm_version"]),
        })
    paths = []
    for (labeler, size), rows in sorted(groups.items()):
        path = out_dir / f"{labeler}_w{size}.jsonl"
        io.write_jsonl(path, sorted(rows, key=lambda r: r["queue_position"]))
        paths.append(path)
        print(f"[label-export] {len(rows):>5} labels -> {path.relative_to(io.ROOT.parent) if path.is_relative_to(io.ROOT.parent) else path}")
    if not paths:
        print("[label-export] no labels in PocketBase yet.")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pb", default=default_pb_url(), help=f"PocketBase URL (default {DEFAULT_PB})")
    args = ap.parse_args()
    try:
        export_labels(args.pb)
    except httpx.ConnectError:
        raise SystemExit(f"could not reach PocketBase at {args.pb} — start it with `pixi run pb`.") from None


if __name__ == "__main__":
    main()
