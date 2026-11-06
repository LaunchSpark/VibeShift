import math

from evaluation.common import io
from evaluation.common.labels import LABELS
from evaluation.eval import frames, metrics
from evaluation.eval.evaluate import evaluate_model


def _gold(label):
    return {"label": label, "soft_probs": {k: float(k == label) for k in LABELS}}


def _pred(mid, label):
    return {"id": mid, "model": "m", "probs": {k: (0.94 if k == label else 0.01) for k in LABELS},
            "raw_top_label": label, "latency_ms": 1.0, "cost_usd": 0.0}


def test_context_sensitive_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ANNOTATIONS", tmp_path)
    io.write_jsonl(tmp_path / "consensus_w1.jsonl", [{"id": "x", **_gold("joy")}, {"id": "y", **_gold("anger")}])
    io.write_jsonl(tmp_path / "consensus_w5.jsonl", [{"id": "x", **_gold("joy")}, {"id": "y", **_gold("sadness")},
                                                     {"id": "z", **_gold("fear")}])
    assert io.context_sensitive_ids() == {"y"}


def test_context_sensitive_none_with_one_file(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ANNOTATIONS", tmp_path)
    io.write_jsonl(tmp_path / "consensus_w5.jsonl", [{"id": "x", **_gold("joy")}])
    assert io.context_sensitive_ids() is None


def test_message_frame_has_no_flag_columns():
    df = frames.message_frame([_pred("x", "joy")], {"x": _gold("joy")}, [{"id": "x", "content": "hi"}], 0.0)
    assert "is_sarcastic" not in df.columns and "is_context_dependent" not in df.columns


def test_context_slice_uses_context_ids():
    gold = {"x": _gold("joy"), "y": _gold("anger")}
    rows = [_pred("x", "joy"), _pred("y", "joy")]
    r = evaluate_model(rows, gold, 0.0, {"x"})
    assert r["macro_f1_context"] == metrics.macro_f1(["joy"], ["joy"])
    assert r["macro_f1_context"] != r["macro_f1"]
    assert "macro_f1_sarcastic" not in r
    assert math.isnan(evaluate_model(rows, gold, 0.0, None)["macro_f1_context"])


def test_report_explains_missing_gold(monkeypatch):
    import sys

    import pytest

    from evaluation.eval import evaluate

    monkeypatch.setattr(io, "load_messages", lambda: [{"id": "a:1", "content": "hi", "split": "test"}])
    monkeypatch.setattr(io, "load_consensus", lambda window=None: {"msg_001": _gold("joy")})
    monkeypatch.setattr(sys, "argv", ["evaluate"])
    with pytest.raises(SystemExit, match="label-consensus"):
        evaluate.main()
