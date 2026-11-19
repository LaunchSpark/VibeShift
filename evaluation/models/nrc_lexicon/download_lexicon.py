"""Download the official EmoLex archive and install its word-level file locally.

Run only for non-commercial research or educational use. The official terms
prohibit redistribution; only the local ignored text file is retained.
"""

from __future__ import annotations

from io import BytesIO
from urllib.request import Request, urlopen
import zipfile

from evaluation.models.nrc_lexicon.prepare_lexicon import MEMBER, TARGET

URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"


def main() -> None:
    request = Request(URL, headers={"User-Agent": "VibeShift research evaluation"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        try:
            lexicon = archive.read(MEMBER)
        except KeyError as exc:
            raise SystemExit(f"Official EmoLex archive does not contain {MEMBER}") from exc
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(lexicon)
    print(f"Installed local EmoLex file at {TARGET}")


if __name__ == "__main__":
    main()
