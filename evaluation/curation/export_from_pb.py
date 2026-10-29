"""Export the 300-message dataset from PocketBase (owned by Teammates A & B).

Pulls messages from the test server's PocketBase ``messages`` collection and
writes ``evaluation/data/messages.jsonl`` in Contract 1A shape, attaching each
message's preceding 3 and 25 messages (same channel, by ``created_at``) as the
context windows. Aim for coverage of ordinary chat, slang, short replies,
sarcasm, and context-dependent messages, then assign 75 ``dev`` / 225 ``test``.

STUB STATE: the PocketBase query and context-window assembly are not implemented.
A & B: use ``vibeshift.store.Store`` (see src/vibeshift/store.py — httpx client
over the same collection the bot writes) to list records, then map each
``StoredMessage`` to the Contract 1A row below. Start PocketBase with ``pixi run pb``.

    pixi run eval-export
"""

from __future__ import annotations

from evaluation.common import io


def to_row(record: dict, preceding_3: list[str], preceding_25: list[str], split: str) -> dict:
    """Map one PocketBase message record to a Contract 1A row (see schema.md)."""
    return {
        "id": record["message_id"],
        "channel_id": record["channel_id"],
        "author_id": record["author_id"],
        "created_at": record["created_at"],
        "content": record["content"],
        "preceding_3": preceding_3,
        "preceding_25": preceding_25,
        "split": split,
    }


def main() -> None:
    raise SystemExit(
        "export_from_pb is a stub. Teammates A & B: implement the PocketBase pull "
        "(via vibeshift.store.Store), build the context windows, assign the 75/225 "
        "split, and write evaluation/data/messages.jsonl with io.write_jsonl(). "
        "Until then, the sample fixture (sample_messages.jsonl) drives the pipeline."
    )


if __name__ == "__main__":
    main()
