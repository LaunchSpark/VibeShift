"""Ensure a local PocketBase superuser exists so `serve` never opens the installer in a browser.

Reads PB_SUPERUSER_EMAIL / PB_SUPERUSER_PASSWORD from the environment or .env.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dotenv() -> dict[str, str]:
    out = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


if __name__ == "__main__":
    cfg = {**_dotenv(), **os.environ}
    email = cfg.get("PB_SUPERUSER_EMAIL") or "admin@example.com"
    password = cfg.get("PB_SUPERUSER_PASSWORD") or "12345678"
    subprocess.run(
        [str(ROOT / "operations/tools/pocketbase/pocketbase"), "superuser", "upsert",
         email, password, "--dir=operations/data/pocketbase", "--migrationsDir=pb_migrations"],
        check=True, cwd=ROOT,
    )
