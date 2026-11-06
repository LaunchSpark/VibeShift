"""The committed fingerprint that keeps every labeler on the same queue.

``evaluation/data/label_manifest.json`` records the seed, the CSV's sha256 and
shape, and the queue algorithm version. ``label-import`` refuses to run when the
local CSV doesn't match it, so two people's "first 500" can never silently differ.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from evaluation.common import io

from .dataset import Message, fingerprint
from .queue import ALGORITHM_VERSION

MANIFEST_PATH = io.DATA / "label_manifest.json"


@dataclass(frozen=True)
class Manifest:
    seed: int
    csv_sha256: str
    rows: int
    conversations: int
    algorithm_version: int


class ManifestMismatch(Exception):
    pass


def read_manifest(path: Path = MANIFEST_PATH) -> Manifest | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Manifest(**{f.name: data[f.name] for f in fields(Manifest)})


def write_manifest(m: Manifest, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(m), indent=2, sort_keys=True) + "\n")


def manifest_for(csv_path: Path, convs: dict[str, list[Message]], seed: int) -> Manifest:
    return Manifest(
        seed=seed,
        csv_sha256=fingerprint(csv_path),
        rows=sum(len(v) for v in convs.values()),
        conversations=len(convs),
        algorithm_version=ALGORITHM_VERSION,
    )


def check_manifest(expected: Manifest, actual: Manifest) -> None:
    """Raise ManifestMismatch naming every field that differs."""
    problems = []
    for f in fields(Manifest):
        want, got = getattr(expected, f.name), getattr(actual, f.name)
        if want == got:
            continue
        if f.name == "csv_sha256":
            problems.append(
                "your data/discord_messages.csv differs from the team's copy "
                f"(sha256 {got[:12]}… vs manifest {want[:12]}…)"
            )
        else:
            problems.append(f"{f.name}: local {got!r} vs manifest {want!r}")
    if problems:
        raise ManifestMismatch("label manifest mismatch:\n  - " + "\n  - ".join(problems))
