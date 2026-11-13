"""Run the installed PocketBase binary on every supported platform."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "operations" / "tools" / "pocketbase" / (
    "pocketbase.exe" if os.name == "nt" else "pocketbase"
)


if __name__ == "__main__":
    raise SystemExit(subprocess.call([str(BINARY), *sys.argv[1:]], cwd=ROOT))
