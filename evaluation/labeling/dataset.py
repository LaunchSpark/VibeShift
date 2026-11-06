"""Load the windowed-labeling dataset from ``data/discord_messages.csv``.

The CSV is grouped by conversation (``conversation_key``) and numbered from 1
within each conversation. A message's stable id is ``"{conversation_key}:{number}"``;
labels, the queue order, and ``messages.jsonl`` all key on it.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "data" / "discord_messages.csv"


@dataclass(frozen=True)
class Message:
    id: str
    conversation_key: str
    number: int
    author: str
    content: str


def message_id(conversation_key: str, number: int) -> str:
    return f"{conversation_key}:{number}"


def parse_id(mid: str) -> tuple[str, int]:
    key, number = mid.rsplit(":", 1)
    return key, int(number)


def load_conversations(path: Path = CSV_PATH) -> dict[str, list[Message]]:
    """Messages grouped by conversation (keys sorted), each list ordered by number.

    Raises ValueError for rows with ``part_number != 1``, duplicate message
    numbers, or numbering that isn't contiguous from 1.
    """
    convs: dict[str, dict[int, Message]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if row["part_number"].strip() != "1":
                raise ValueError(
                    f"line {line}: part_number={row['part_number']!r}; only single-part "
                    "messages (part_number 1) are supported"
                )
            key, number = row["conversation_key"], int(row["conversation_message_number"])
            msgs = convs.setdefault(key, {})
            if number in msgs:
                raise ValueError(f"line {line}: duplicate message {message_id(key, number)}")
            msgs[number] = Message(
                id=message_id(key, number),
                conversation_key=key,
                number=number,
                author=row["author"],
                content=row["message"].replace("\\n", "\n"),
            )
    out: dict[str, list[Message]] = {}
    for key in sorted(convs):
        numbers = sorted(convs[key])
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"conversation {key}: message numbers are not contiguous from 1")
        out[key] = [convs[key][n] for n in numbers]
    return out


def fingerprint(path: Path = CSV_PATH) -> str:
    """sha256 of the CSV bytes, so teammates can confirm they hold the same file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
