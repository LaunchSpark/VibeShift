import pytest

from evaluation.labeling.dataset import Message, message_id, parse_id
from evaluation.labeling.queue import assign_splits, build_queue, min_overlap_distance, window

REAL_SHAPE = {f"c{k:02d}": 100 for k in range(73)}
MIXED = {"a": 5, "b": 100, "c": 40}


def test_queue_is_a_permutation():
    q = build_queue(MIXED, 4240)
    assert sorted(q) == sorted(message_id(c, i) for c, n in MIXED.items() for i in range(1, n + 1))


def test_queue_is_deterministic_and_seed_sensitive():
    assert build_queue(REAL_SHAPE, 4240) == build_queue(REAL_SHAPE, 4240)
    assert build_queue(REAL_SHAPE, 4240) != build_queue(REAL_SHAPE, 4241)


def test_adjacent_positions_differ_in_conversation():
    q = build_queue(REAL_SHAPE, 4240)
    assert all(parse_id(x)[0] != parse_id(y)[0] for x, y in zip(q, q[1:]))


@pytest.mark.parametrize("size,floor", [(2, 2500), (3, 1500), (5, 1000), (10, 350), (25, 146)])
def test_overlapping_windows_are_far_apart(size, floor):
    assert min_overlap_distance(build_queue(REAL_SHAPE, 4240), size) >= floor


def test_every_prefix_spans_all_conversations():
    assert {parse_id(m)[0] for m in build_queue(REAL_SHAPE, 4240)[:73]} == set(REAL_SHAPE)


def test_window_clips_at_conversation_start():
    conv = [Message(message_id("a", n), "a", n, "Al", f"m{n}") for n in range(1, 6)]
    assert [m.number for m in window(conv, 2, 5)] == [1, 2]
    assert [m.number for m in window(conv, 5, 3)] == [3, 4, 5]
    assert [m.number for m in window(conv, 4, 1)] == [4]


def test_min_overlap_distance_none_when_size_1():
    assert min_overlap_distance(["a:1", "a:2"], 1) is None


def test_assign_splits():
    s = assign_splits(REAL_SHAPE, 4240)
    assert sum(v == "dev" for v in s.values()) == 18 and set(s) == set(REAL_SHAPE)
    assert set(s.values()) == {"dev", "test"}
    assert s == assign_splits(REAL_SHAPE, 4240)
