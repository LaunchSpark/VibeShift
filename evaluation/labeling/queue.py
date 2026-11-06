"""The shared labeling order, windows, and the dev/test split.

Every labeler walks the same queue of *targets* (the last message of a window),
so any prefix — "the first 500" — is identical for everyone. The order is the
same for every window size. It is built so windows that share messages sit far
apart:

1. Conversations are shuffled once (by the seed) and visited round-robin, so a
   conversation recurs exactly every ``len(conversations)`` positions.
2. Within a conversation, targets are visited in golden-ratio order rather than
   reading order, so neighbours ``i`` and ``i+1`` (whose windows overlap the most)
   land roughly 0.618 of the conversation apart.

Changing this algorithm reorders everyone's queue: bump ``ALGORITHM_VERSION``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence

from .dataset import Message, message_id, parse_id

ALGORITHM_VERSION = 1
DEFAULT_SEED = 4240
_PHI = (5**0.5 - 1) / 2


def build_queue(lengths: Mapping[str, int], seed: int) -> list[str]:
    """Message ids in labeling order, for conversations of the given lengths."""
    rng = random.Random(seed)
    keys = sorted(lengths)
    rng.shuffle(keys)
    seqs: dict[str, list[int]] = {}
    for key in keys:
        u = rng.random()
        seqs[key] = sorted(range(1, lengths[key] + 1), key=lambda i: ((i * _PHI + u) % 1.0, i))
    rounds = max(lengths.values(), default=0)
    return [message_id(k, seqs[k][r]) for r in range(rounds) for k in keys if r < len(seqs[k])]


def window(conversation: Sequence[Message], number: int, size: int) -> list[Message]:
    """Messages ``max(1, number-size+1) .. number`` of one conversation (1-based, contiguous)."""
    return list(conversation[max(0, number - size) : number])


def min_overlap_distance(queue: Sequence[str], size: int) -> int | None:
    """Smallest queue distance between two same-conversation targets whose windows overlap."""
    pos = {parse_id(mid): p for p, mid in enumerate(queue)}
    best: int | None = None
    for (key, i), p in pos.items():
        for k in range(1, size):
            q = pos.get((key, i + k))
            if q is not None and (best is None or abs(q - p) < best):
                best = abs(q - p)
    return best


def assign_splits(keys: Iterable[str], seed: int, dev_fraction: float = 0.25) -> dict[str, str]:
    """Whole conversations go to 'dev' or 'test', so tuning never sees test conversations."""
    order = sorted(keys)
    random.Random(f"split:{seed}").shuffle(order)
    n_dev = round(dev_fraction * len(order))
    return {key: ("dev" if i < n_dev else "test") for i, key in enumerate(order)}
