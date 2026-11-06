"""Build Contract 1A rows (``evaluation/data/messages.jsonl``) from the CSV."""

from __future__ import annotations

from .dataset import Message, parse_id


def build_rows(
    convs: dict[str, list[Message]], queue: list[str], splits: dict[str, str]
) -> list[dict]:
    """One row per message, in queue order, with its preceding-3/25 context."""
    rows = []
    for position, mid in enumerate(queue):
        key, number = parse_id(mid)
        conv = convs[key]
        target = conv[number - 1]
        before = [m.content for m in conv[: number - 1]]
        rows.append({
            "id": target.id,
            "conversation_key": key,
            "author": target.author,
            "content": target.content,
            "preceding_3": before[-3:],
            "preceding_25": before[-25:],
            "queue_position": position,
            "split": splits[key],
        })
    return rows
