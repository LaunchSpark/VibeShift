import pytest

from evaluation.annotation.consensus import build_consensus
from evaluation.common import io
from evaluation.common.labels import LABELS


def test_majority_and_soft_probs():
    rows, ties = build_consensus({"a": {"x": "joy"}, "b": {"x": "joy"}, "c": {"x": "anger"}}, {})
    assert ties == []
    assert rows[0]["id"] == "x" and rows[0]["label"] == "joy"
    assert rows[0]["soft_probs"]["joy"] == pytest.approx(2 / 3)
    assert rows[0]["soft_probs"]["anger"] == pytest.approx(1 / 3)
    assert set(rows[0]["soft_probs"]) == set(LABELS)


def test_tie_needs_override():
    by = {"a": {"x": "joy"}, "b": {"x": "anger"}}
    assert build_consensus(by, {}) == ([], ["x"])
    rows, ties = build_consensus(by, {"x": "anger"})
    assert ties == [] and rows[0]["label"] == "anger"
    assert rows[0]["soft_probs"]["joy"] == pytest.approx(0.5)


def test_only_shared_prefix():
    rows, _ = build_consensus({"a": {"x": "joy", "y": "fear"}, "b": {"x": "joy"}}, {})
    assert [r["id"] for r in rows] == ["x"]


def test_load_label_files_reads_one_window(tmp_path):
    io.write_jsonl(tmp_path / "lucas_w5.jsonl", [{"id": "x", "label": "joy"}])
    io.write_jsonl(tmp_path / "ana_w5.jsonl", [{"id": "x", "label": "fear"}])
    io.write_jsonl(tmp_path / "ana_w3.jsonl", [{"id": "x", "label": "anger"}])
    assert io.load_label_files(5, tmp_path) == {"ana": {"x": "fear"}, "lucas": {"x": "joy"}}


def test_load_consensus_requires_window_when_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ANNOTATIONS", tmp_path)
    row = {"id": "x", "label": "joy", "soft_probs": {k: float(k == "joy") for k in LABELS}}
    io.write_jsonl(tmp_path / "consensus_w3.jsonl", [row])
    io.write_jsonl(tmp_path / "consensus_w5.jsonl", [{**row, "label": "anger"}])
    assert io.consensus_windows() == [3, 5]
    with pytest.raises(SystemExit, match="--window"):
        io.load_consensus()
    assert io.load_consensus(window=5)["x"]["label"] == "anger"


def test_load_consensus_single_file_needs_no_window(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ANNOTATIONS", tmp_path)
    row = {"id": "x", "label": "joy", "soft_probs": {k: float(k == "joy") for k in LABELS}}
    io.write_jsonl(tmp_path / "consensus_w5.jsonl", [row])
    assert io.load_consensus()["x"]["label"] == "joy"


def test_overrides_path(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ANNOTATIONS", tmp_path)
    assert io.overrides_path(5) == tmp_path / "overrides_w5.json"
