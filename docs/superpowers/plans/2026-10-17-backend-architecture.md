# VibeShift Backend Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pixi-managed project where a discord.py bot captures messages into PocketBase, scores them newest-first from a persistent deque, maintains per-day emotion rollups, and runs downtime catch-up and history-extension jobs.

**Architecture:** PocketBase (a pinned binary, schema in JS migrations) is the only store and the contract with the UI ([UI plan](2026-10-17-dashboard-ui.md)). The bot writes over PocketBase's REST API with `httpx`. The backlog is the `messages` collection itself: pending rows ordered by `priority_key` descending, and the scorer takes a chunk off the top. The scorer also updates `daily_rollups`, which the dashboard aggregates from, so the UI never loads raw messages.

**Tech Stack:** pixi (conda-forge + PyPI), Python 3.12, PyTorch 2.14.1 (conda-forge, osx-64), laya (git), discord.py 2.7, httpx, PocketBase 0.40.5, pytest, pytest-asyncio.

**Spec:** [docs/superpowers/specs/2026-10-17-emotion-dashboard-design.md](../specs/2026-10-17-emotion-dashboard-design.md)

**Ordering with the UI plan:** Tasks 1–2 here are the shared foundation (pixi plus the PocketBase schema). The UI plan starts after Task 2 lands, and the rest of both plans can then run in parallel.

## Global Constraints

- Local only: PocketBase binds `127.0.0.1:8090`. Collection API rules are all `""` (public). There is no auth (the spec says one user on their own machine).
- **Every Discord id (guild/channel/author/message) is a PocketBase `text` field, never `number`.** Snowflakes exceed 2^53, and PocketBase numbers are float64.
- `priority_key` is the snowflake as a **20-digit zero-padded string**, so lexical order equals numeric order.
- Bot-authored messages are never stored. Messages with empty text content are not stored.
- 13 labels, verbatim and in this order: `anger, sarcasm, surprise, love, desire, shame, fear, sadness, disgust, confusion, neutral, guilt, happiness` (from the model's `questions.json`; always read them from there, never hardcode them in Python).
- Model: `LaunchSpark/laya-emotion-tracker`.
- Day bucket: `YYYY-MM-DD` of the message's `created_at` (never the edit time) in timezone `VIBESHIFT_TZ` (default: system local).
- Arbitrary history ranges are not allowed. History only grows at its two ends, through `downtime` and `extend` jobs.
- Scoring never runs on the Discord event loop thread: use `asyncio.to_thread`.

## Review Focus

1. **A live message arrives while a downtime catch-up is mid-fetch:** it must still be scored before every catch-up message (Task 4 `test_live_beats_catchup_inserted_later`).
2. **A message is edited after being scored:** the rollup must subtract the old probabilities before adding the new ones, so the day's count stays the same (Task 6 `test_rescore_after_edit_does_not_double_count`).
3. **A message is deleted while still pending, or after scoring:** pending ones must never be scored, and scored ones must be removed from the rollup (Task 4 `test_pop_skips_deleted`, Task 6 `test_delete_after_score_subtracts`).
4. **Model failure on one message:** that message becomes `failed` and the scorer keeps going (Task 6 `test_failed_chunk_marks_failed_and_continues`).
5. **A channel the bot cannot read (403) during catch-up or extend:** skip it, keep the job running, record the skip in the job's `error` (Task 7 `test_forbidden_channel_skipped`).

---

### Task 1: Migrate to pixi with conda-forge PyTorch

**Files:**
- Modify: `pyproject.toml` (add `[tool.pixi.*]`; drop the Intel torch/numpy/transformers caps and the 3.11 pin)
- Delete: `uv.lock`, `.python-version`
- Modify: `.gitignore` (add `.pixi/`, `operations/tools/pocketbase/`, `operations/data/`)
- Create: `tests/test_emotions_model.py`

**Interfaces:**
- Produces: `pixi run test` (pytest, excluding `-m model`) and `pixi run test-model`; the env has torch, laya, discord.py, httpx, marimo 0.25.1, and marimo-studio 0.3.x.

- [ ] **Step 1: Install pixi** with `brew install pixi`. Verify with `pixi --version` (expect 0.81.x).
- [ ] **Step 2: Write `[tool.pixi.workspace]`** with `channels = ["conda-forge"]` and `platforms = ["osx-64", "osx-arm64", "linux-64"]`. Conda deps: `python = "3.12.*"`, `pytorch = ">=2.9"`, `transformers`, `numpy`, `nodejs = "24.*"`. PyPI deps: everything else currently in `[project.dependencies]` (marimo==0.25.1, marimo-studio[recommended]>=0.3.0,<0.4, click>=8.5, anywidget, wigglystuff, discord.py>=2.7,<3, python-dotenv, laya @ git, huggingface_hub, tqdm) plus `httpx>=0.28` and the editable `vibeshift`. A `test` feature adds pytest and pytest-asyncio (`asyncio_mode = "auto"` in `[tool.pytest.ini_options]`, with marker `model`). Tasks: `test = "pytest -m 'not model'"`, `test-model = "pytest -m model"`, `bot = "vibeshift-bot"`, `notebook = "marimo edit notebooks/emotion_tracker.py"`.
- [ ] **Step 3: Write the model test** `tests/test_emotions_model.py::test_predicts_known_label` (marked `model`): `predict_emotion("I cannot believe we actually won!") in emotion_labels()` and `len(emotion_labels()) == 13`.
- [ ] **Step 4: Run** `pixi install && pixi run test-model`. Expected: PASS, and `pixi run python -c "import torch; print(torch.__version__)"` prints 2.14.x. **If the model fails to load:** fall back to `python = "3.11.*"` and `pytorch = "2.2.*"`, with `transformers<5` and `numpy<2`, and note why in pyproject comments.
- [ ] **Step 5: Run** `pixi run marimo check notebooks/emotion_tracker.py` and expect no errors. Update the README setup section to `pixi install` / `pixi run ...`.
- [ ] **Step 6: Commit** `build: migrate from uv to pixi with conda-forge pytorch`.

### Task 2: PocketBase binary, schema migration, and test fixture (the shared contract)

**Files:**
- Create: `scripts/install_pocketbase.py`, `pb_migrations/1760700000_init.js`, `tests/conftest.py`, `tests/test_schema.py`
- Modify: `pyproject.toml` (pixi tasks)

**Interfaces:**
- Produces: the collections below, which are **the contract the UI plan reads**. Change them only by adding a new migration.
- Produces: pixi tasks `pb-install` (`python scripts/install_pocketbase.py 0.40.5`) and `pb` (`operations/tools/pocketbase/pocketbase serve --http=127.0.0.1:8090 --dir=operations/data/pocketbase --migrationsDir=pb_migrations`, depends on `pb-install`).
- Produces: a pytest fixture `pb_url: str` that starts a fresh PocketBase on a free port in a `tmp_path` dir with `pb_migrations`, waits for `/api/health`, yields the base URL, and kills it afterwards.

Schema (all rules `""`; `text` unless noted):

| collection | fields | indexes |
|---|---|---|
| `guilds` | `guild_id`, `name`, `icon_url`, `history_depth` (date, nullable) | unique `guild_id` |
| `channels` | `channel_id`, `guild_id`, `name`, `type`, `newest_fetched_id`, `oldest_fetched_id` | unique `channel_id`; `guild_id` |
| `members` | `guild_id`, `author_id`, `display_name`, `avatar_url` | unique (`guild_id`,`author_id`) |
| `messages` | `message_id`, `guild_id`, `channel_id`, `author_id`, `created_at` (date), `edited_at` (date), `deleted_at` (date), `reply_to_id`, `content`, `day`, `priority_key`, `score_status` (select: pending/scored/failed), `emotion`, `probabilities` (json), `confidence` (number), `model_id`, `rollup_applied` (bool) | unique `message_id`; (`score_status`,`priority_key`); (`guild_id`,`channel_id`,`created_at`) |
| `daily_rollups` | `guild_id`, `channel_id`, `author_id`, `day`, `count` (number), `prob_sums` (json: label→float) | unique (`guild_id`,`channel_id`,`author_id`,`day`); (`guild_id`,`day`) |
| `backfill_jobs` | `guild_id`, `origin` (select: downtime/extend), `target` (date, extend only), `status` (select: pending/running/done/failed), `scanned` (number), `error`, plus autodate `created`/`updated` | (`status`) |

- [ ] **Step 1: Write `scripts/install_pocketbase.py VERSION`.** It downloads `pocketbase_{VERSION}_{darwin|linux}_{amd64|arm64}.zip` from the GitHub release into `operations/tools/pocketbase/` and is a no-op when `pocketbase --version` already reports VERSION. Prior art: `../ticket_to_ride` platform-aware binary discovery.
- [ ] **Step 2: Write the failing test** `tests/test_schema.py::test_collections_exist`: GET `{pb_url}/api/collections/<name>/records?perPage=1` returns 200 for all six collections. Also `test_snowflake_roundtrip`: create a `messages` record with `message_id="1234567890123456789"` and read it back unchanged.
- [ ] **Step 3: Run** `pixi run test tests/test_schema.py`. Expected: FAIL (no migration).
- [ ] **Step 4: Write `pb_migrations/1760700000_init.js`** with the schema above, using the PocketBase 0.40 JSVM `migrate((app) => {...}, (app) => {...})` API.
- [ ] **Step 5: Run** the tests. Expected: PASS.
- [ ] **Step 6: Commit** `feat: pocketbase schema, installer, and test fixture`.

### Task 3: Pure helpers for keys, days and rollups

**Files:**
- Create: `src/vibeshift/keys.py`, `src/vibeshift/rollups.py`, `tests/test_keys.py`, `tests/test_rollups.py`

**Interfaces:**
- Produces: `priority_key(snowflake: int) -> str` (zero-padded to 20 digits); `edit_priority_key(edited_at: datetime) -> str` (uses `discord.utils.time_snowflake(edited_at, high=True)`); `day_bucket(created_at: datetime, tz: ZoneInfo) -> str`.
- Produces: `apply(sums: dict[str, float], count: int, probs: dict[str, float], sign: int) -> tuple[dict[str, float], int]`, which adds (`sign=1`) or subtracts (`sign=-1`) one message.

- [ ] **Step 1: Write the failing tests.**
  - `priority_key(5) == "00000000000000000005"`
  - `priority_key(2**63 - 1) > priority_key(2**62)`
  - `edit_priority_key(t)` > `priority_key` of any message created before `t`
  - `day_bucket(datetime(2026,10,17,3,0,tzinfo=UTC), ZoneInfo("America/Los_Angeles")) == "2026-10-16"`
  - `apply` round-trip: adding then subtracting the same probs returns the original sums (within 1e-9) and count
- [ ] **Step 2: Run** `pixi run test tests/test_keys.py tests/test_rollups.py`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** them again. Expected: PASS.
- [ ] **Step 5: Commit** `feat: priority keys, day buckets, rollup math`.

### Task 4: Store, the PocketBase client and the deque operations

**Files:**
- Create: `src/vibeshift/config.py`, `src/vibeshift/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: Task 2 schema and the `pb_url` fixture; Task 3 helpers.
- Produces: `Settings` (env: `POCKETBASE_URL`=`http://127.0.0.1:8090`, `VIBESHIFT_TZ`, `SCORE_CHUNK`=8, `JOB_POLL_SECONDS`=5).
- Produces: `class Store(base_url: str)`, async, on `httpx.AsyncClient`. It has these methods:
  - `upsert_guild(guild_id, name, icon_url)`
  - `upsert_channel(channel_id, guild_id, name, type)`
  - `upsert_member(guild_id, author_id, display_name, avatar_url)`
  - `insert_pending(msg: NewMessage) -> bool`. Idempotent on `message_id`; returns False if it already exists.
  - `mark_edited(message_id, content, edited_at)`. Sets content, `edited_at`, `priority_key = edit_priority_key(edited_at)`, and `score_status="pending"`; a no-op if the message is unknown. It **keeps** the existing `probabilities` and `rollup_applied`, so the scorer can subtract the old values.
  - `mark_deleted(message_id, deleted_at) -> StoredMessage | None`
  - `pop_chunk(n) -> list[StoredMessage]`. Returns up to `n` rows with `score_status="pending"` and empty `deleted_at`, sorted `-priority_key`. It does not mutate them; the single scorer owns them.
  - `save_score(message_id, emotion, probabilities, confidence, model_id)`, `mark_failed(message_id)`
  - `get_rollup` / `put_rollup(guild_id, channel_id, author_id, day, count, prob_sums)`
  - `get_channel(channel_id)`, `set_cursors(channel_id, newest=None, oldest=None)` (newest only moves up, oldest only moves down)
  - `create_job(guild_id, origin, target=None)`, `next_pending_job()`, `update_job(id, **fields)`
- `NewMessage` / `StoredMessage` are dataclasses mirroring the `messages` fields, with ids as `str` and datetimes timezone-aware.

- [ ] **Step 1: Write the failing tests** in `tests/test_store.py` against `pb_url`:
  - `test_pop_newest_first`: insert keys 10, 30, 20 → pop(2) returns 30, 20
  - `test_extend_goes_to_bottom`: insert 100, 200 (stored range), then 50 (extend) → pop(3) returns 200, 100, 50
  - `test_live_beats_catchup_inserted_later`: insert live 500, then catch-up 300 and 400 afterwards → pop(1) returns 500
  - `test_edit_goes_to_top`: insert 10 and 20, score both, then `mark_edited(10, ...)` at a later time → pop(1) returns 10
  - `test_pop_skips_deleted`: insert 10, `mark_deleted(10)` → pop returns []
  - `test_insert_idempotent`: inserting the same id twice returns True then False
  - `test_cursors_monotonic`: `set_cursors(newest=5)` then `newest=3` leaves 5; `oldest=5` then `oldest=7` leaves 5
- [ ] **Step 2: Run** `pixi run test tests/test_store.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `config.py` and `store.py`. Use PocketBase filter syntax and `sort=-priority_key`. Compare cursors as snowflake strings by length, then lexically.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: pocketbase store with newest-first deque`.

### Task 5: Batch scoring with full probabilities

**Files:**
- Modify: `src/vibeshift/emotions.py`
- Modify: `tests/test_emotions_model.py`

**Interfaces:**
- Produces: `@dataclass Score(emotion: str, probabilities: dict[str, float], confidence: float)` and `score_batch(texts: list[str]) -> list[Score]`, which calls `agent.decide_batch(texts, questions=q, return_details=True)` and reads `values["emotion_tracker"]`.
- Produces: `MODEL_ID = MODEL_REPO` for the `model_id` field.

- [ ] **Step 1: Write the failing test** `test_score_batch_shape` (`model`): for 2 texts, it returns 2 `Score`s. Each has 13 probability keys equal to `emotion_labels()`, a sum within 0.02 of 1.0, and `emotion == max(probabilities, key=probabilities.get)`.
- [ ] **Step 2: Run** `pixi run test-model`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** it again. Expected: PASS.
- [ ] **Step 5: Commit** `feat: batch scoring with per-label probabilities`.

### Task 6: Scorer worker and rollup maintenance

**Files:**
- Create: `src/vibeshift/scorer.py`, `tests/test_scorer.py`

**Interfaces:**
- Consumes: `Store`, `Score`, `rollups.apply`, `day_bucket`.
- Produces: `class Scorer(store: Store, score_fn: Callable[[list[str]], list[Score]], chunk: int, tz: ZoneInfo)` with these members:
  - `async run_once() -> int`: pops a chunk, scores it via `asyncio.to_thread(score_fn, texts)`, and returns the count processed
  - `async run_forever(idle_sleep: float = 1.0)`
  - `async on_deleted(msg: StoredMessage)`: subtracts it from the rollup if `rollup_applied`
  - `wake()`: an `asyncio.Event` that `on_message` sets so the idle wait ends immediately
- Rollup update per scored message: if `rollup_applied`, subtract the **old** probabilities first, then add the new ones. Set `rollup_applied=True`. All rollup read-modify-writes go through one `asyncio.Lock` on the scorer.

- [ ] **Step 1: Write the failing tests** with a fake `score_fn` returning fixed `Score`s:
  - `test_scores_chunk_and_writes_rollup`: 3 pending messages on the same day/channel/author → rollup `count == 3`, and `prob_sums` equals the summed fake probs
  - `test_rescore_after_edit_does_not_double_count`: score, `mark_edited`, then score again with different probs → `count == 1`, sums equal the new probs
  - `test_delete_after_score_subtracts`: score, delete, `on_deleted` → `count == 0`
  - `test_failed_chunk_marks_failed_and_continues`: `score_fn` raises for a chunk → retry each message singly; the one that still raises gets `score_status == "failed"`, and the others get scored
- [ ] **Step 2: Run** `pixi run test tests/test_scorer.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: newest-first scorer with rollup maintenance`.

### Task 7: History jobs (downtime catch-up and extend depth)

**Files:**
- Create: `src/vibeshift/history.py`, `tests/test_history.py`

**Interfaces:**
- Consumes: `Store`, `priority_key`, `day_bucket`.
- Produces: `to_new_message(m: discord.Message, tz) -> NewMessage | None` (None for bot authors or empty content; this one function is shared with live capture).
- Produces: `async run_job(job, guild: discord.Guild, store: Store, tz)`:
  - **downtime:** for each text channel with `newest_fetched_id`, iterate `channel.history(after=Object(newest), limit=None, oldest_first=True)`, insert pending, and advance `newest`
  - **extend:** for each text channel, iterate `channel.history(before=Object(oldest) or None, after=target, limit=None)` (newest to oldest), insert pending, and lower `oldest`. On completion, set `guilds.history_depth = target`
  - Update `scanned` every 100 messages. Status goes pending → running → done, or failed with `error` set
- Produces: `async job_loop(bot, store, tz, poll_seconds)`, which runs pending jobs one at a time, oldest `created` first.
- Produces: `async create_downtime_jobs(bot, store)`, which creates one `downtime` job per guild that has any channel with `newest_fetched_id` set.

- [ ] **Step 1: Write the failing tests** with fake channel/guild objects whose `history()` is an async generator over canned messages:
  - `test_downtime_fetches_after_cursor_only`: cursor 100, history has 90, 110, 120 → inserts 110 and 120, and newest becomes 120
  - `test_extend_stops_at_target_and_sets_depth`
  - `test_forbidden_channel_skipped`: one channel raises `discord.Forbidden` → the job is `done`, the other channel is ingested, and `error` mentions the channel id
  - `test_bot_and_empty_messages_ignored`
- [ ] **Step 2: Run** `pixi run test tests/test_history.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: downtime catch-up and history-extension jobs`.

### Task 8: Bot wiring and the dev runner

**Files:**
- Modify: `src/vibeshift/bot.py`
- Create: `scripts/dev.py`
- Modify: `pyproject.toml` (pixi `dev` task), `README.md`, `.env.example`

**Interfaces:**
- Consumes: everything above.
- Bot events:
  - `on_ready`: upsert guilds, text channels and members (requires `intents.members = True`; document enabling Server Members Intent), then `create_downtime_jobs`, then start `Scorer.run_forever` and `job_loop` as tasks. Start them once, even across reconnects.
  - `on_message`: `to_new_message` → `insert_pending` → `set_cursors(newest=..., oldest=...` if the channel has none) → upsert member → `scorer.wake()`
  - `on_raw_message_edit`: if `content` is in the payload data → `mark_edited`
  - `on_raw_message_delete`: `mark_deleted` → `scorer.on_deleted`
  - `on_guild_join`, `on_guild_channel_create`: upsert
  - The `!vibe` command stays.
- `scripts/dev.py` starts `pb`, the bot, and `npm --prefix dashboard run dev` as subprocesses. It prefixes their output and stops all three on Ctrl-C. The `dev` task runs it.

- [ ] **Step 1: Implement the wiring.** No new unit tests: the logic lives in the tested modules.
- [ ] **Step 2: Smoke test.** Run `pixi run pb` and `pixi run bot` with a real token. Post 3 messages → they appear in the PocketBase admin UI (`http://127.0.0.1:8090/_/`) as pending, then scored. `daily_rollups` has a row with count 3. Edit one → still count 3, with new sums. Delete one → count 2. Stop the bot, post 2 messages, start it → a `downtime` job runs to `done` and both get scored.
- [ ] **Step 3: Run** `pixi run test`. Expected: all pass.
- [ ] **Step 4: Commit** `feat: wire bot to store, scorer, and history jobs`.
