from dataclasses import replace

import pytest

from evaluation.labeling.dataset import Message, message_id
from evaluation.labeling.manifest import (
    Manifest, ManifestMismatch, check_manifest, read_manifest, write_manifest,
)
from evaluation.labeling.queue import assign_splits, build_queue
from evaluation.labeling.rows import build_rows

M = Manifest(seed=4240, csv_sha256="ab" * 32, rows=7300, conversations=73, algorithm_version=1)


def _convs():
    return {
        key: [Message(message_id(key, n), key, n, f"u{n}", f"{key}{n}") for n in range(1, size + 1)]
        for key, size in {"a": 5, "b": 3}.items()
    }


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "label_manifest.json"
    write_manifest(M, path)
    assert read_manifest(path) == M
    assert path.read_text().endswith("\n")


def test_read_missing_manifest_is_none(tmp_path):
    assert read_manifest(tmp_path / "x.json") is None


def test_check_names_csv_mismatch():
    with pytest.raises(ManifestMismatch, match="differs from the team's copy"):
        check_manifest(M, replace(M, csv_sha256="0"))


def test_check_passes_identical():
    check_manifest(M, M)


def test_build_rows_shape():
    convs = _convs()
    queue = build_queue({k: len(v) for k, v in convs.items()}, 4240)
    rows = build_rows(convs, queue, assign_splits(convs, 4240))
    assert [r["queue_position"] for r in rows] == list(range(len(queue)))
    assert [r["id"] for r in rows] == queue
    r = next(r for r in rows if r["id"] == "a:3")
    assert r["preceding_3"] == [convs["a"][0].content, convs["a"][1].content]
    assert r["preceding_25"] == r["preceding_3"]
    assert r["content"] == "a3" and r["author"] == "u3" and r["conversation_key"] == "a"
    assert next(r for r in rows if r["id"] == "a:1")["preceding_3"] == []
    assert set(r) == {"id", "conversation_key", "author", "content", "preceding_3",
                      "preceding_25", "queue_position", "split"}
