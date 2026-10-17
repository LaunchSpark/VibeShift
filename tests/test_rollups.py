import pytest

from vibeshift.rollups import apply


def test_apply_round_trip():
    sums = {"joy": 1.5, "anger": 0.25}
    probs = {"joy": 0.7, "anger": 0.1, "fear": 0.2}
    s2, c2 = apply(sums, 3, probs, 1)
    assert c2 == 4
    assert s2["fear"] == pytest.approx(0.2)
    s3, c3 = apply(s2, c2, probs, -1)
    assert c3 == 3
    for k in ("joy", "anger"):
        assert s3[k] == pytest.approx(sums[k], abs=1e-9)
    assert s3["fear"] == pytest.approx(0.0, abs=1e-9)
    assert sums == {"joy": 1.5, "anger": 0.25}
