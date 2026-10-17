"""Run PocketBase, the bot and the dashboard dev server together; Ctrl-C stops all."""
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pump(name: str, proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        sys.stdout.write(f"[{name}] {line}")
        sys.stdout.flush()


def main() -> int:
    # PocketBase only via `pixi run pb`: it upserts a superuser first so no browser tab opens.
    # Never pass --open to vite.
    cmds = {
        "pb": ["pixi", "run", "pb"],
        "bot": ["pixi", "run", "bot"],
    }
    if (ROOT / "dashboard" / "package.json").exists():
        cmds["dash"] = ["npm", "--prefix", "dashboard", "run", "dev"]
    else:
        print("[dev] dashboard/package.json not found; skipping dashboard", flush=True)

    procs: dict[str, subprocess.Popen] = {}
    try:
        for name, cmd in cmds.items():
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=True)
            procs[name] = p
            threading.Thread(target=pump, args=(name, p), daemon=True).start()
        while all(p.poll() is None for p in procs.values()):
            try:
                procs[next(iter(procs))].wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        dead = [n for n, p in procs.items() if p.poll() is not None]
        print(f"[dev] {', '.join(dead)} exited; stopping the rest", flush=True)
    except KeyboardInterrupt:
        print("\n[dev] stopping...", flush=True)
    finally:
        for p in procs.values():
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGINT)  # session leader: signal only it, then group-kill
                except ProcessLookupError:
                    pass
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
