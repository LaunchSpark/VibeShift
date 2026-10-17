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
   and turn on **Message Content Intent**.
2. `cp .env.example .env` and set `DISCORD_TOKEN`.
3. `pixi run bot`

The bot logs the emotion of every message it sees, and `!vibe <text>` replies
with a prediction.
