# VibeShift
Sentiment Analytics for Discord

Classifies Discord messages into 13 emotions using the fine-tuned
[LaunchSpark/laya-emotion-tracker](https://huggingface.co/LaunchSpark/laya-emotion-tracker)
model.

## Setup

```bash
uv sync
```

Python is pinned to 3.11. On Intel Macs the newest torch is 2.2.2, and its
`torch.compile` won't run on 3.12 or later.

## Notebook

```bash
uv run marimo edit notebooks/emotion_tracker.py
```

## Discord bot

1. Create an app at https://discord.com/developers/applications, add a bot,
   and turn on **Message Content Intent**.
2. `cp .env.example .env` and set `DISCORD_TOKEN`.
3. `uv run vibeshift-bot`

The bot logs the emotion of every message it sees, and `!vibe <text>` replies
with a prediction.
