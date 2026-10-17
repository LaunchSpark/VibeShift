# VibeShift
Sentiment Analytics for Discord

Classifies Discord messages into 13 emotions using the fine-tuned
[LaunchSpark/laya-emotion-tracker](https://huggingface.co/LaunchSpark/laya-emotion-tracker)
model.

## Setup

Install [pixi](https://pixi.sh) (the installer puts it in `~/.pixi/bin`; add that to your `PATH`), then:

```bash
pixi install
```

PyTorch comes from conda-forge, which still publishes current builds for Intel
Macs. Run tests with `pixi run test` (or `pixi run test-model` for the real
model).

## Notebook

```bash
pixi run notebook
```

## Discord bot

1. Create an app at https://discord.com/developers/applications, add a bot,
   and under **Bot -> Privileged Gateway Intents** turn on **Message Content
   Intent** and **Server Members Intent** (the bot will not start without them).
2. `cp .env.example .env` and set `DISCORD_TOKEN`.
3. `pixi run dev` installs the dashboard's npm packages (`pixi run dash-install`,
   also runnable on its own) and starts PocketBase, the bot and the dashboard dev
   server, with prefixed output; Ctrl-C stops all. Or run the pieces separately:
   `pixi run pb`, `pixi run bot`, `pixi run dash`.

The bot captures every message in guild text channels into PocketBase, scores
them newest-first, keeps daily rollups, catches up on messages sent while it was
offline, and `!vibe <text>` replies with a prediction. Threads (and voice-channel
chat) are not captured yet.

## Dashboard

- Dashboard: http://localhost:5173 (served by `pixi run dash` / `pixi run dev`).
- PocketBase admin: http://127.0.0.1:8090/_/ — default login `admin@example.com` /
  `12345678` (override with `PB_SUPERUSER_EMAIL` / `PB_SUPERUSER_PASSWORD` in `.env`).
- PocketBase only accepts browser requests from the dashboard's origin
  (`localhost:5173` / `127.0.0.1:5173`), so other websites can't read your messages.
- Tests: `pixi run dash-test`.

### Demo data

`npm --prefix dashboard run seed` (with `pixi run pb` running) writes a fake
**Seed Server** guild (id `100000000000000001`: channels, members, rollups,
messages and sample jobs) into the **same database the bot uses**
(`operations/data/pocketbase`). Re-running it replaces that guild's rows. To remove
it, delete the rows with `guild_id = 100000000000000001` from every collection in
the PocketBase admin, or keep demo data apart by running PocketBase with a separate
`--dir` and pointing the seed at it with `PB_URL`.
