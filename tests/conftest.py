import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
PB_VERSION = "0.40.5"
PB_BIN = ROOT / "operations" / "tools" / "pocketbase" / "pocketbase"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pb_binary() -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_pocketbase.py"), PB_VERSION],
        check=True,
    )
    return PB_BIN


@pytest.fixture
def pb_url(pb_binary, tmp_path) -> str:
    port = _free_port()
    proc = subprocess.Popen(
        [
            str(pb_binary), "serve", f"--http=127.0.0.1:{port}",
            f"--dir={tmp_path / 'pb_data'}",
            f"--migrationsDir={ROOT / 'pb_migrations'}",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while True:
            try:
                if httpx.get(f"{url}/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None or time.time() > deadline:
                raise RuntimeError("PocketBase failed to start")
            time.sleep(0.1)
        yield url
    finally:
        proc.kill()
        proc.wait()
