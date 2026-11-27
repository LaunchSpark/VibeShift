"""Launch the platform-specific PocketBase executable with its arguments."""
from __future__ import annotations

import subprocess
import sys

from scripts.install_pocketbase import BINARY


if __name__ == "__main__":
    raise SystemExit(subprocess.call([str(BINARY), *sys.argv[1:]]))
