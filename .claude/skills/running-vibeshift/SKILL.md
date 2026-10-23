---
name: running-vibeshift
description: Use when asked to run, start, launch, demo, screenshot, or smoke-test VibeShift (the Discord bot, PocketBase, or the React dashboard), to seed demo data, or to confirm a change works in the running app rather than only in tests.
---

# Running VibeShift

## Overview

VibeShift has three processes. Run every command from the repo root; pixi pins Python, Node and PocketBase.

| Process | Address | Started by |
|---|---|---|
| PocketBase | `http://127.0.0.1:8090` (admin UI at `/_/`) | `pixi run pb` |
| Bot | Discord gateway; Laya runs in-process | `pixi run bot` |
| Dashboard (Vite) | `http://localhost:5173` | `pixi run dash` |

## Check the platform first

| OS | Status |
|---|---|
| macOS (Intel or ARM), Linux x64 | Supported through pixi. |
| Windows | **Only the dashboard runs natively**; see "Dashboard on Windows" below. The rest fails on Windows: pixi has no `win-64` platform, `scripts/install_pocketbase.py` exits with `unsupported platform`, and `scripts/dev.py` calls the POSIX-only `os.killpg`. For the full stack, use a Linux WSL2 distro. This is untested. |

### Dashboard on Windows

1. Install from inside `dashboard/`, with npm 11:

   ```sh
   cd dashboard && npx -y npm@11 ci
   ```

   - npm 11 matches the lockfile, which was written by pixi's Node 24.
   - Node 22's bundled npm 10 rewrites `package-lock.json` and skips `@rolldown/binding-win32-x64-msvc`. Vite then fails with `Cannot find native binding`.
   - `npm --prefix dashboard install` fails on npm 10 with `ENOENT ...\package.json`.
2. Start the server: `npm run dev` from `dashboard/`, or `npm --prefix dashboard run dev` from the repo root. `--prefix` is relative to the current directory.
3. Open **`http://localhost:5173`**. Vite listens on `[::1]` only, so `127.0.0.1:5173` refuses connections.
4. Expect a slow first start: loading the config alone can take about 20 s.
5. Run the tests with `npx vitest --run` from `dashboard/`.

## Setup (once)

1. Install [pixi](https://pixi.sh), then run `pixi install`.
2. `cp .env.example .env`. Set `DISCORD_TOKEN` (needed only for the bot).
3. In the Discord developer portal, turn on the **Message Content** and **Server Members** intents. The bot won't start without them.

## Quick reference

| Goal | Command |
|---|---|
| Whole stack | `pixi run dev`. Output is prefixed `[pb]`, `[bot]`, `[dash]`. Ctrl-C stops everything, and if one process exits, the others are stopped. |
| PocketBase only | `pixi run pb`. Installs PocketBase 0.40.5, creates or updates the superuser, and applies `pb_migrations/`. |
| Dashboard only | `pixi run dash-install` once, then `pixi run dash` |
| Bot only | `pixi run bot`. Needs PocketBase running and `DISCORD_TOKEN` set. |
| Demo data | `npm --prefix dashboard run seed`, with PocketBase running |
| Is PocketBase up? | `curl -s http://127.0.0.1:8090/api/health` |
| Is the model working? | `pixi run python -c "from vibeshift.emotions import predict_emotion; print(predict_emotion('we actually won!'))"` |
| Tests | `pixi run test`, `pixi run test-model` (loads the real model; slow), `pixi run dash-test` |
| Notebook | `pixi run notebook` |

## Checking the UI without Discord

1. Start PocketBase: `pixi run pb`.
2. Seed demo data: `npm --prefix dashboard run seed`.
3. Start the dashboard: `pixi run dash`.
4. Open `http://localhost:5173/?guild=100000000000000001`. That is the seeded **Seed Server**.
5. The Backfill page is at `/backfill`. Filters live in the query string (`range`, `ma`, `people`, `channels`, `emo`), so a URL reproduces the exact view.
6. PocketBase admin login: `admin@example.com` / `12345678`. Override with `PB_SUPERUSER_EMAIL` and `PB_SUPERUSER_PASSWORD` in `.env`.

## Checking with Discord

1. The bot logs `Logged in as ...` once it connects.
2. Post in a guild text channel. The message is stored as `pending`, then `scored`; the dashboard's pending count drops as the scorer catches up.
3. `!vibe <text>` replies with `Vibe: **<label>**`.

## Common mistakes

- **The dashboard must be on port 5173.** PocketBase only accepts browser requests from `localhost:5173` and `127.0.0.1:5173`. If 5173 is busy, Vite moves to 5174 and every request fails with a CORS error. Free the port instead.
- **The seed writes into the real database** (`operations/data/pocketbase`). To remove it, delete the rows with `guild_id = 100000000000000001`. To keep demo data separate instead, run PocketBase with a different `--dir` and point the seed at it with `PB_URL`.
- **Another PocketBase URL:** set `VITE_POCKETBASE_URL` for the dashboard, `POCKETBASE_URL` for the bot, and `PB_URL` for the seed.
- **The bot's first start is slow:** it downloads `LaunchSpark/laya-emotion-tracker` from Hugging Face. Scoring takes about 5 s per message on an Intel Mac CPU.
- **Bot started before PocketBase:** it retries with backoff and logs exceptions until PocketBase is up. Start `pb` first, or use `pixi run dev`.
- **Messages missing:** threads, voice-channel chat, bot messages and empty messages are never captured.
- **Don't pass `--open` to Vite.** `pixi run dev` relies on it not opening a browser.
