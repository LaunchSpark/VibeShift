"""Install the PocketBase binary: python scripts/install_pocketbase.py VERSION.

Downloads the release zip from GitHub into operations/tools/pocketbase/.
No-op when the installed binary already reports VERSION.
"""
from __future__ import annotations

import io
import platform
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "operations" / "tools" / "pocketbase"
BINARY = DEST / "pocketbase"


def asset_name(version: str) -> str:
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    arch = {"x86_64": "amd64", "AMD64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine()
    )
    if system is None or arch is None:
        raise SystemExit(f"unsupported platform: {platform.system()} {platform.machine()}")
    return f"pocketbase_{version}_{system}_{arch}.zip"


def installed_version() -> str | None:
    if not BINARY.exists():
        return None
    try:
        out = subprocess.run(
            [str(BINARY), "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().split()[-1] if out.strip() else None


def install(version: str) -> Path:
    if installed_version() == version:
        return BINARY
    url = f"https://github.com/pocketbase/pocketbase/releases/download/v{version}/{asset_name(version)}"
    print(f"downloading {url}", file=sys.stderr)
    resp = httpx.get(url, follow_redirects=True, timeout=120)
    resp.raise_for_status()
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        data = zf.read("pocketbase")
    BINARY.write_bytes(data)
    BINARY.chmod(BINARY.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return BINARY


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: install_pocketbase.py VERSION")
    install(sys.argv[1])
