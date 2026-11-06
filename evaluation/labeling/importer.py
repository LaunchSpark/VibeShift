"""Import the labeling dataset: check the CSV, build the queue, sync PocketBase.

    pixi run label-import                 # everyone: check + sync
    pixi run label-import --new-manifest  # once, by whoever owns the dataset

Steps: verify ``data/discord_messages.csv`` against the committed manifest
(refusing *before* touching PocketBase if it differs), write
``evaluation/data/messages.jsonl`` (Contract 1A), then upsert every message into
the local ``label_messages`` collection. Safe to re-run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

from evaluation.common import io

from .dataset import CSV_PATH, load_conversations, parse_id
from .manifest import (
    MANIFEST_PATH, ManifestMismatch, check_manifest, manifest_for, read_manifest, write_manifest,
)
from .queue import ALGORITHM_VERSION, DEFAULT_SEED, assign_splits, build_queue, min_overlap_distance
from .rows import build_rows

DEFAULT_PB = "http://127.0.0.1:8090"
_FIELDS = ("message_id", "conversation_key", "message_number", "author", "content",
           "queue_position", "algorithm_version")


def default_pb_url() -> str:
    return os.environ.get("VIBESHIFT_PB_URL", DEFAULT_PB)


def fetch_all(client: httpx.Client, collection: str, fields: str = "", filter: str = "") -> list[dict]:
    """Every record of a collection, paged 500 at a time."""
    items, page = [], 1
    while True:
        params = {"perPage": 500, "page": page, "skipTotal": 1}
        if fields:
            params["fields"] = fields
        if filter:
            params["filter"] = filter
        r = client.get(f"/api/collections/{collection}/records", params=params)
        r.raise_for_status()
        batch = r.json()["items"]
        items.extend(batch)
        if len(batch) < 500:
            return items
        page += 1


def _sync(client: httpx.Client, records: list[dict]) -> None:
    existing = {r["message_id"]: r for r in fetch_all(client, "label_messages", "id," + ",".join(_FIELDS))}
    wanted = {r["message_id"] for r in records}
    stale = [r for mid, r in existing.items() if mid not in wanted]
    changed = [(existing[r["message_id"]], r) for r in records
               if r["message_id"] in existing
               and any(existing[r["message_id"]].get(k) != r[k] for k in _FIELDS)]
    moved = [old for old, new in changed if old["queue_position"] != new["queue_position"]]

    # Saved labels record their queue position, so reordering or dropping messages
    # after labeling has started would point everyone's progress at the wrong windows.
    if (stale or moved) and fetch_all(client, "labels", "id")[:1]:
        raise SystemExit(
            f"labels already exist in PocketBase, but this import would move {len(moved)} and "
            f"remove {len(stale)} messages in the queue. Export them first (`pixi run label-export`); "
            "changing the order invalidates existing progress."
        )

    url = "/api/collections/label_messages/records"
    for r in stale:
        client.delete(f"{url}/{r['id']}").raise_for_status()
    # Park changed rows on positions nothing else holds (below every current value,
    # including rows left negative by an interrupted run), then write the real values,
    # so the unique queue_position index never collides.
    base = min([0, *(r["queue_position"] for r in existing.values())]) - 1
    for i, (old, _) in enumerate(changed):
        client.patch(f"{url}/{old['id']}", json={"queue_position": base - i}).raise_for_status()
    for old, new in changed:
        client.patch(f"{url}/{old['id']}", json=new).raise_for_status()
    for r in records:
        if r["message_id"] not in existing:
            client.post(url, json=r).raise_for_status()


def run_import(
    pb_url: str,
    csv_path: Path = CSV_PATH,
    manifest_path: Path = MANIFEST_PATH,
    messages_path: Path = io.MESSAGES,
    seed: int | None = None,
    new_manifest: bool = False,
) -> int:
    convs = load_conversations(csv_path)
    expected = read_manifest(manifest_path)
    if seed is None:
        seed = expected.seed if expected else DEFAULT_SEED
    actual = manifest_for(csv_path, convs, seed)

    if new_manifest:
        write_manifest(actual, manifest_path)
        print(f"[label-import] wrote new manifest {manifest_path} — commit it so teammates match.")
    elif expected is None:
        raise SystemExit(
            f"no label manifest at {manifest_path}. The dataset owner runs "
            "`pixi run label-import --new-manifest` once and commits it; teammates then pull."
        )
    else:
        try:
            check_manifest(expected, actual)
        except ManifestMismatch as err:
            raise SystemExit(str(err)) from None

    queue = build_queue({k: len(v) for k, v in convs.items()}, seed)
    rows = build_rows(convs, queue, assign_splits(convs, seed))
    io.write_jsonl(messages_path, rows)

    records = []
    for position, mid in enumerate(queue):
        key, number = parse_id(mid)
        m = convs[key][number - 1]
        records.append({
            "message_id": m.id, "conversation_key": key, "message_number": number,
            "author": m.author, "content": m.content, "queue_position": position,
            "algorithm_version": ALGORITHM_VERSION,
        })
    with httpx.Client(base_url=pb_url.rstrip("/"), timeout=30) as client:
        _sync(client, records)

    print(f"[label-import] synced {len(records)} messages from {len(convs)} conversations (seed {seed}).")
    print("  window size  min distance between overlapping windows")
    for size in (2, 3, 5, 10, 25):
        print(f"  {size:>11}  {min_overlap_distance(queue, size)}")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pb", default=default_pb_url(), help=f"PocketBase URL (default {DEFAULT_PB})")
    ap.add_argument("--new-manifest", action="store_true", help="write a new manifest from the local CSV")
    ap.add_argument("--seed", type=int, help="queue seed (only with --new-manifest)")
    args = ap.parse_args()
    if args.seed is not None and not args.new_manifest:
        raise SystemExit("--seed only applies with --new-manifest.")
    try:
        run_import(args.pb, seed=args.seed, new_manifest=args.new_manifest)
    except httpx.ConnectError:
        raise SystemExit(f"could not reach PocketBase at {args.pb} — start it with `pixi run pb`.") from None


if __name__ == "__main__":
    main()
