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
3. `pixi run dev` starts PocketBase, the bot and (when `dashboard/package.json`
   exists) the dashboard dev server, with prefixed output; Ctrl-C stops all.
   Or run the pieces separately: `pixi run pb`, `pixi run bot`.

The bot captures every message into PocketBase, scores them newest-first,
keeps daily rollups, catches up on messages sent while it was offline, and
`!vibe <text>` replies with a prediction.
