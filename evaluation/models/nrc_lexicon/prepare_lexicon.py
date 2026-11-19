"""Install a personally obtained NRC EmoLex archive into the ignored local data directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "data" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
MEMBER = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="NRC archive downloaded under its research/educational terms")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        try:
            content = archive.read(MEMBER)
        except KeyError as exc:
            raise SystemExit(f"{args.archive} does not contain {MEMBER}") from exc
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(content)
    print(f"Installed local lexicon at {TARGET}")


if __name__ == "__main__":
    main()
