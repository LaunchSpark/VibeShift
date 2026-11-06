import httpx

from evaluation.common import io
from evaluation.labeling.exporter import export_labels


def _label(pb_url, labeler, size, mid, label, position):
    r = httpx.post(f"{pb_url}/api/collections/labels/records", json={
        "message_id": mid, "labeler": labeler, "window_size": size, "label": label,
        "queue_position": position, "algorithm_version": 1,
    })
    r.raise_for_status()


def test_export_one_file_per_labeler_and_window(pb_url, tmp_path):
    _label(pb_url, "lucas", 5, "a:2", "anger", 1)
    _label(pb_url, "lucas", 5, "a:1", "joy", 0)
    _label(pb_url, "lucas", 3, "a:1", "fear", 0)
    _label(pb_url, "ana", 5, "a:1", "joy", 0)
    paths = export_labels(pb_url, tmp_path)
    assert sorted(p.name for p in paths) == ["ana_w5.jsonl", "lucas_w3.jsonl", "lucas_w5.jsonl"]
    rows = io.read_jsonl(tmp_path / "lucas_w5.jsonl")
    assert [r["queue_position"] for r in rows] == [0, 1]
    assert rows[0] == {"id": "a:1", "labeler": "lucas", "window_size": 5, "label": "joy",
                       "queue_position": 0, "algorithm_version": 1}


def test_export_filename_uses_slug(pb_url, tmp_path):
    _label(pb_url, "lucas-s", 5, "a:1", "joy", 0)
    assert [p.name for p in export_labels(pb_url, tmp_path)] == ["lucas-s_w5.jsonl"]


def test_export_leaves_other_files(pb_url, tmp_path):
    (tmp_path / "zed_w5.jsonl").write_text('{"id": "a:1"}\n')
    _label(pb_url, "lucas", 5, "a:1", "joy", 0)
    export_labels(pb_url, tmp_path)
    assert (tmp_path / "zed_w5.jsonl").read_text() == '{"id": "a:1"}\n'
