# VibeShift
Sentiment Analytics for Discord

Classifies Discord messages into 7 emotions (joy, sadness, anger, fear, disgust, surprise,
neutral) using the fine-tuned
[LaunchSpark/laya-emotion-tracker](https://huggingface.co/LaunchSpark/laya-emotion-tracker)
model (revision `v2-7-labels`, trained by `notebooks/finetune_laya_emotions.py`).

## Setup

Install [pixi](https://pixi.sh) (the installer puts it in `~/.pixi/bin`; add that to your `PATH`), then:

```bash
pixi install
```

PyTorch comes from conda-forge, which still publishes current builds for Intel
Macs. Run tests with `pixi run test` (or `pixi run test-model` for the real
model).

## Labeler quick start

This is the workflow for someone who has just pulled the repository and needs
to label messages. Labeling is local and offline; you do not need Discord,
bot credentials, PocketBase, or an import command.

1. Install Pixi and prepare the environment:

   ```bash
   pixi install
   ```

2. Copy the dataset CSV files into the repository's `data/` directory. You can
   label any CSV in that directory. Each file must have these columns:

   ```text
   conversation_key,conversation_message_number,author,message
   ```

3. Start the dashboard and leave this terminal running:

   ```bash
   pixi run dash
   ```

4. Open [http://localhost:5173/label](http://localhost:5173/label), choose a
   dataset from the Dataset menu, enter your labeler name, and choose a window
   size. The window size is the maximum number of messages shown, including the
   message being labeled. Use keys `1`–`7` to choose an emotion and Backspace
   to go back.

5. Your work is written locally to
   `data/annotations/labels/`. That directory is gitignored. Keep your
   annotation files if they need to be submitted or shared; they are not
   uploaded anywhere automatically.

If the Dataset menu is empty, confirm that the files are directly inside
`data/` (not a nested folder), have a `.csv` extension, and contain the four
required columns. Refresh the page after adding or changing a dataset.

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

### Label data

Put one or more source CSV files in `data/`, then run:

```bash
pixi run dash
```

Open http://localhost:5173/label, choose a dataset, enter your name, and label
with the on-screen buttons or keys `1`–`7`. The dashboard reads CSV files
directly and writes each dataset/annotator's JSONL file to
`data/annotations/labels/`.
No PocketBase, bot, Discord connection, or import step is required.

### Demo data

`npm --prefix dashboard run seed` (with `pixi run pb` running) writes a fake
**Seed Server** guild (id `100000000000000001`: channels, members, rollups,
messages and sample jobs) into the **same database the bot uses**
(`operations/data/pocketbase`). Re-running it replaces that guild's rows. To remove
it, delete the rows with `guild_id = 100000000000000001` from every collection in
the PocketBase admin, or keep demo data apart by running PocketBase with a separate
`--dir` and pointing the seed at it with `PB_URL`.
