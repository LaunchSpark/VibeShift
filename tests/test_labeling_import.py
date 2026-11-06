import httpx
import pytest

from evaluation.common import io
from evaluation.labeling.importer import run_import
from evaluation.labeling.manifest import read_manifest

HEADER = "conversation_key,conversation_message_number,part_number,author,message\n"
SIZES = {"a": 4, "b": 6, "c": 3}
TOTAL = sum(SIZES.values())


@pytest.fixture
def paths(tmp_path):
    csv = tmp_path / "messages.csv"
    body = "".join(f"{k},{n},1,u{n},{k} says {n}\n" for k, size in SIZES.items()
                   for n in range(1, size + 1))
    csv.write_text("﻿" + HEADER + body, encoding="utf-8")
    return csv, tmp_path / "label_manifest.json", tmp_path / "messages.jsonl"


def _records(pb_url):
    r = httpx.get(f"{pb_url}/api/collections/label_messages/records", params={"perPage": 500})
    r.raise_for_status()
    return r.json()["items"]


def test_import_creates_rows_and_messages_file(pb_url, paths):
    csv, manifest, msgs = paths
    assert run_import(pb_url, csv, manifest, msgs, new_manifest=True) == TOTAL
    assert read_manifest(manifest).rows == TOTAL
    recs = _records(pb_url)
    assert sorted(r["queue_position"] for r in recs) == list(range(TOTAL))
    assert {r["message_id"] for r in recs} == {f"{k}:{n}" for k, s in SIZES.items() for n in range(1, s + 1)}
    rows = io.read_jsonl(msgs)
    assert len(rows) == TOTAL
    by_id = {r["message_id"]: r for r in recs}
    assert all(by_id[row["id"]]["queue_position"] == row["queue_position"] for row in rows)


def test_reimport_same_csv_is_idempotent(pb_url, paths):
    csv, manifest, msgs = paths
    run_import(pb_url, csv, manifest, msgs, new_manifest=True)
    first = {r["message_id"]: (r["id"], r["queue_position"]) for r in _records(pb_url)}
    run_import(pb_url, csv, manifest, msgs)
    second = {r["message_id"]: (r["id"], r["queue_position"]) for r in _records(pb_url)}
    assert first == second and len(second) == TOTAL


def test_import_refuses_changed_csv_before_touching_pb(pb_url, paths):
    csv, manifest, msgs = paths
    run_import(pb_url, csv, manifest, msgs, new_manifest=True)
    csv.write_text(csv.read_text(encoding="utf-8").replace("a says 1", "a says ONE"), encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the team's copy"):
        run_import(pb_url, csv, manifest, msgs)
    assert {r["content"] for r in _records(pb_url) if r["message_id"] == "a:1"} == {"a says 1"}


def test_import_without_manifest_requires_flag(pb_url, paths, tmp_path):
    csv, _, msgs = paths
    with pytest.raises(SystemExit, match="--new-manifest"):
        run_import(pb_url, csv, tmp_path / "none.json", msgs)


def test_labels_unique_per_labeler_window_message(pb_url):
    body = {"message_id": "a:1", "labeler": "lucas", "window_size": 5, "label": "joy",
            "queue_position": 0, "algorithm_version": 1}
    url = f"{pb_url}/api/collections/labels/records"
    assert httpx.post(url, json=body).status_code == 200
    assert httpx.post(url, json={**body, "label": "anger"}).status_code == 400
    assert httpx.post(url, json={**body, "window_size": 3}).status_code == 200


def test_labels_reject_unknown_label(pb_url):
    body = {"message_id": "a:1", "labeler": "lucas", "window_size": 5, "label": "love",
            "queue_position": 0, "algorithm_version": 1}
    assert httpx.post(f"{pb_url}/api/collections/labels/records", json=body).status_code == 400
