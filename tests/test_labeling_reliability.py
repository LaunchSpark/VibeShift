import pytest

from evaluation.annotation.reliability import fleiss_kappa, pairwise_cohen


def test_fleiss_reference_example():
    # Fleiss (1971) worked example: 10 items x 14 raters x 5 categories, kappa = 0.210.
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert fleiss_kappa(table) == pytest.approx(0.2099, abs=1e-4)


def test_fleiss_perfect_agreement():
    assert fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == pytest.approx(1.0)


def test_pairwise_cohen_perfect():
    by = {"a": {"x": "joy", "y": "anger"}, "b": {"x": "joy", "y": "anger"}}
    assert pairwise_cohen(by) == {("a", "b"): pytest.approx(1.0)}


def test_pairwise_cohen_uses_shared_ids_only():
    by = {"a": {"x": "joy", "y": "anger", "z": "fear"}, "b": {"x": "joy", "y": "anger"}}
    assert pairwise_cohen(by)[("a", "b")] == pytest.approx(1.0)
