# VibeShift Emotion Dashboard — Design

Date: 2026-10-17
Status: awaiting owner approval

## Intent

VibeShift is a Discord sentiment-analytics bot. The owner wants a Statbot-style
dashboard (https://statbot.net/dashboard), but the thing it tracks is the
**emotional mood of a server over time** rather than raw activity counts.

The headline screen is an **Emotion Prevalence History** chart: one line per
emotion over time, smoothed with a configurable moving average, defaulting to
the whole server and filterable down to individual people. The owner's
priority is *features existing first, speed later*: inference is slow on their
machine (about 5 s per message, measured), and that is explicitly tabled.

## Users and access

- One user: the owner, on their own machine. Everything runs locally.
- No login, no Discord OAuth, no multi-user permissions, no hosting.

## Goals (v1)

1. **Capture.** The bot records every new (non-bot) message in servers it is
   in, scores it with the emotion model, and stores the message metadata and
   all 13 emotion probabilities.
2. **Emotion chart.** A dashboard page shows emotion prevalence over time with
   a moving average, the whole server by default.
3. **Filters.** Narrow the chart by person (the key filter), by channel, and by
   time range. Exclude bots by default.
4. **Manual history backfill.** The owner can choose to score past history
   with a chosen look-back range (for example the last 30 days). It never
   happens automatically.

## Non-goals (v1)

- Voice, status/presence, games/activity, invites, Statdocks-style counters.
- Performance work: GPU, ONNX, batching queues. The decide_batch-vs-threads
  benchmark is in `scripts/bench_parallel.py` for later.
- Login and permissions for anyone other than the owner.

## Future TODOs (design for, don't build)

- **Impact measurement:** for example "how did the server's mood shift after a
  person joined, an event happened, or an announcement went out." The exact
  definition is still open. Implication for v1: store enough per message
  (timestamp, author, channel, reply-to message id, guild) that this can be
  computed later without re-collecting data.
- **Auto-roles** (Statbot's "Statroles"): assign Discord roles based on
  emotion or activity stats.

## Anti-goals

- A dashboard that looks generic or unlike the reference. The visual target is
  the owner's mockup, [assets/emotion-chart-mockup.png](assets/emotion-chart-mockup.png): a dark card, the title "Emotion Prevalence History",
  pill toggles per emotion with colored dots, a smooth multi-line chart with
  point markers and a 0–100% y-axis, and a row of per-emotion stat cards
  ("X (Current)", a big percentage, "Peak: Y%") under the chart.
- Silent automatic history scraping. Going back further than the bot's own
  downtime is opt-in, every time. (Filling the gap since the last message the
  bot saw is automatic; see Scoring priority.)

## Decided by the owner

- **UI:** shadcn/ui (https://ui.shadcn.com/). Use its Sidebar, Card, Chart
  (Recharts-based), Toggle/Badge, Select, Popover/Calendar, and Dialog
  components rather than hand-rolled equivalents.
- **Data store:** PocketBase.
- **Environment and tooling:** pixi, replacing uv, so that Python, Node, and the
  PocketBase version are all pinned and versioned in this repo. uv's
  `pyproject.toml` and `uv.lock` setup from the initial scaffold gets migrated.
- **Default scope:** the whole server.
- **Backfill:** manual, with a chosen range.
- **Dashboard layout reference:** Statbot's shell (left sidebar with a server
  switcher at the top and page links; a collapsible "Filters" panel at the top
  of the page listing active filters such as "Time — Last 60 days" and
  "User Type — Users", with a "+" to add a filter).

## Existing state

- `src/vibeshift/emotions.py`: loads `LaunchSpark/laya-emotion-tracker` via
  `huggingface_hub.snapshot_download` and `laya.Agent`, and reads the question
  schema from the model's `questions.json`. Its 13 labels are: anger, sarcasm,
  surprise, love, desire, shame, fear, sadness, disgust, confusion, neutral,
  guilt, happiness.
- `laya.decide(agent, text, questions=q, return_details=True)` returns the
  chosen label **and** a probability for every label from the same forward
  pass, at no extra cost.
- `src/vibeshift/bot.py`: a starter discord.py bot (logs emotions; `!vibe`).
- `notebooks/emotion_tracker.py`: marimo notebook for ad-hoc testing.
- Platform constraints: the machine is an Intel Mac (x86_64, i7-4980HQ). With
  PyPI wheels, torch caps at 2.2.2, which forces Python 3.11, transformers<5,
  and numpy<2. conda-forge (which pixi uses) lists osx-64 pytorch builds that
  may be newer. The builder should check this; if a newer torch works, the
  3.11 and transformers pins can go. Otherwise keep the current pins.
- `../ticket_to_ride` already vendors a PocketBase binary under
  `operations/tools/pocketbase/` and has platform-aware binary discovery.
  That is prior art to copy, not to depend on. PocketBase is not on
  conda-forge.

## Architecture

```
Discord ──> bot (discord.py, Python) ──> emotion model (laya, in-process)
                    │
                    └──> PocketBase (local, pinned version) <── dashboard (React + shadcn, Vite)
```

- **The bot** writes guilds, channels, members, and messages with their scores
  into PocketBase.
- **The dashboard** reads PocketBase directly through its JS SDK/REST API.
  There is no separate Python API unless aggregation turns out to need one;
  that call is left to the builder (see below).
- **Backfill trigger:** the dashboard has a "Backfill history" control (a
  dialog with a guild, an optional channel set, and a look-back range). It
  creates a `backfill_jobs` record in PocketBase. The bot watches for pending
  jobs, pages through Discord history, stores the messages as pending in the
  pending set, and updates job progress and status, which the
  dashboard shows. The same mechanism handles downtime catch-up (see Scoring
  priority).
- **Dev workflow:** pixi tasks start PocketBase, the bot, and the dashboard
  dev server, with one task that starts all three.

## Data model (PocketBase collections)

These are minimum fields; the builder may add more.

- `guilds`: discord_id, name, icon_url
- `channels`: discord_id, guild (relation), name, type
- `members`: discord_id, guild (relation), display_name, avatar_url, is_bot
- `messages`: discord_id (unique), guild, channel, author (relations),
  created_at (Discord timestamp, not insert time), reply_to_discord_id
  (nullable), content, emotion (top label), probabilities (JSON of all 13),
  confidence, model_id
- `backfill_jobs`: guild, channels (optional), since, until, origin
  (downtime/manual), status (pending/running/done/failed), scanned, scored,
  error

Storing `content` is fine; this is local-only, single-user. Messages are
idempotent on `discord_id`, so live capture and backfill can overlap safely.

Additional message fields (my calls):

- `score_status` (pending/scored/failed): **store first, score later.** At
  about 5 s per message the scorer lags behind a busy server, so a message is
  saved as `pending` the moment it arrives, and a worker scores pending
  messages most-recent-first (see Scoring priority). The backlog survives bot restarts, and nothing that
  arrived is dropped. Score fields are nullable until scored. The chart only
  uses scored messages; the dashboard shows the pending count.
- `edited_at`: on edit, update `content` and set the message back to `pending`
  so it is re-scored. Use raw events so that uncached messages are handled.
- `deleted_at`: on delete, keep the row but set `deleted_at` (a soft delete),
  and exclude those rows from charts. Keeping the row preserves history for
  future impact work.

## Scoring priority (decided by the owner)

There is **one pending set, scored most-recent-first**: a priority queue (a
max-heap on recency). The scorer always takes the pending message with the
greatest `priority_at`, where `priority_at = edited_at ?? created_at` (using
`edited_at` is my call, so that a fresh edit to an old message is re-scored
right away instead of sinking).

Live messages are by definition the most recent, so they come before backfill
automatically, with no separate live queue needed. Backfill fills in from
newest to oldest, so recent chart days complete first.

The set is backed by PocketBase, not held in memory: "pop" means "fetch the
pending message with the max `priority_at`", using an index on
(`score_status`, `priority_at`). The backlog therefore survives restarts with
no rebuild step.

**Every kind of backfill goes through the same pipeline: a `backfill_jobs`
record whose fetched messages are stored as pending.** There is no
special-case path. A job has one of two origins:

- **`downtime` (created automatically):** on startup the bot creates one job
  per guild covering the time since the newest message it stored. This is the
  only automatically created job.
- **`manual` (created by the owner):** a range chosen in the dashboard.

Both kinds are fetched, stored, and scored the same way, and both show up in
the Backfill page's job list with their progress. Add an `origin`
(downtime/manual) field to `backfill_jobs`.

A live message that arrives while an older message is being scored is scored
next. The current item finishes first, with no preemption mid-inference.

## Emotion chart semantics

- **Metric (my call):** for each time bucket, the mean probability of each
  emotion across the messages in that bucket that pass the active filters,
  shown as 0–100%. This matches the mockup's "probability scores" wording and
  gives smooth lines. An alternate "share of messages whose top label is X"
  mode is a nice-to-have toggle, not required.
- **Buckets:** daily by default. Days with no messages are gaps, not zeros.
- **Moving average:** selectable window (raw, 7, 14, 21, or 30 days;
  default 7). It is a trailing average over buckets and, like Statbot's
  `avg[]=21`, can be shown alongside or instead of the raw line. Use
  message-weighted averaging so that quiet days don't swing the line.
- **Emotion toggles:** pills for all 13 emotions with a stable color per
  emotion. Default on: the 4 most prevalent in the current view (my call;
  13 lines at once is unreadable).
- **Stat cards:** one per visible emotion: "Current" (the latest smoothed
  value) and "Peak" (the max smoothed value in range).

## Filters

- **Time range:** presets (7, 30, 60, or 90 days, all) plus a custom range.
  Default 30 days.
- **People:** multi-select searchable members combobox. With several people
  selected, show their combined data in v1; per-person comparison lines are a
  nice-to-have.
- **Channels:** multi-select.
- **User type:** not in v1. Bot-authored messages are never captured (my
  call): they would compete with real users for the slow model and add little
  to "the mood of the server."
- **Server:** the sidebar switcher picks the guild.
- Filters are reflected in the URL query string, so a view can be bookmarked.

## Pages (v1)

1. **Emotions:** the chart, stat cards, and filters. This is the landing page.
2. **Backfill:** start a job and list past jobs with their progress.

Sidebar entries for future pages (Impact, Auto-roles) may appear disabled or
be omitted.

## Left to the builder

- Whether aggregation happens client-side, in a PocketBase JS hook or custom
  route, or in a small Python service. Pick based on what stays responsive at
  around 100k messages.
- The PocketBase version, the pixi project layout, and how to migrate off uv.
- Scoring in the bot: inline per message vs. a background worker queue. It
  must not block the Discord event loop.
- The emotion color palette, as long as it fits the mockup's dark theme.
- Discord rate-limit handling during backfill.
- Testing approach.
