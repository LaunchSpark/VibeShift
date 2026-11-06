# Windowed Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label every Discord message as the last message of a sliding window, in one shared seeded order that keeps overlapping windows far apart, stored in PocketBase per labeler and window size, and merged into the evaluation's gold labels.

**Architecture:** A Python package `evaluation/labeling/` reads `data/discord_messages.csv`, builds a deterministic queue order, checks the CSV against a committed manifest, writes `evaluation/data/messages.jsonl` and upserts the dataset into a local PocketBase collection. The dashboard Label page reads windows from PocketBase and saves labels there; an export task writes per-labeler JSONL files that are committed and merged into per-window-size consensus files for the evaluation scripts.

**Tech Stack:** Python 3.12 (stdlib `csv`/`random`/`hashlib`, httpx, scikit-learn), PocketBase 0.40.5 JS migrations, React 19 + TypeScript + shadcn + `pocketbase` JS SDK, pytest, vitest, pixi tasks.

**Spec:** `docs/superpowers/specs/2026-11-06-windowed-labeling-design.md`

## Global Constraints

- Label set and order: `LABELS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")` from `evaluation/common/labels.py` (TS mirror: `dashboard/src/lib/emotions.ts`). The labeler chooses one of these 7, nothing else.
- Message id: `f"{conversation_key}:{conversation_message_number}"`.
- Window for target n at size W: messages `max(1, n-W+1) … n` of the same conversation. Never crosses `conversation_key`.
- Queue order is independent of W. Default seed `4240`; `ALGORITHM_VERSION = 1`.
- CSV is read with `encoding="utf-8-sig"`; literal `\n` in `message` becomes a real newline.
- Fail loudly (ValueError) if any row has `part_number != "1"`.
- `data/` and `evaluation/data/messages.jsonl` are gitignored; only ids, labels, the manifest and overrides are committed.
- Existing migrations are never edited; schema changes go in a new file in `pb_migrations/`.
- Dev/test split by whole conversation, seeded, `round(0.25 * n_conversations)` dev.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **CSV text with commas, quotes, emoji, `\n` escapes, and a BOM** — must round-trip into `content` exactly (Task 1 test `test_load_handles_quotes_bom_and_escapes`).
2. **Labeler names with spaces, capitals or punctuation** ("Lucas S.") — must normalize to one stable slug, used both in PocketBase rows and export filenames (Task 5 `normalizeLabeler`, Task 7 `test_export_filename_uses_slug`).
3. **Fast repeated key presses** — one keypress = one saved label; presses while a save is in flight are ignored, never skipping a window (Task 6 Step 3, guarded by `saving` state).
4. **Page opened before `label-import` ran, or with PocketBase down** — shows "Run `pixi run label-import`" / the connection error, not a blank card (Task 6 Step 3, Task 5 `fetchQueueSize` returns 0).
5. **Re-running `label-import`** (same CSV, or a changed CSV) — same CSV is idempotent with identical `queue_position`s; changed CSV is refused before touching PocketBase (Task 4 tests).

---

## Phase A — labeling (can ship now)

### Task 1: CSV dataset loader

**Files:**
- Create: `evaluation/labeling/__init__.py` (empty), `evaluation/labeling/dataset.py`
- Test: `tests/test_labeling_dataset.py`

**Interfaces:**
- Produces:
  - `REPO = Path(__file__).resolve().parents[2]`; `CSV_PATH = REPO / "data" / "discord_messages.csv"`
  - `@dataclass(frozen=True) class Message: id: str; conversation_key: str; number: int; author: str; content: str`
  - `message_id(conversation_key: str, number: int) -> str`
  - `parse_id(mid: str) -> tuple[str, int]` (split on the last `:`)
  - `load_conversations(path: Path = CSV_PATH) -> dict[str, list[Message]]` — keys in sorted order, each list sorted by `number`
  - `fingerprint(path: Path = CSV_PATH) -> str` — sha256 hex of the file bytes

- [ ] **Step 1: Write failing tests** (write tiny CSVs into `tmp_path`)

```python
def test_load_groups_and_orders(tmp_path):  # rows out of order across 2 conversations
    convs = load_conversations(csv)
    assert list(convs) == ["a", "b"]
    assert [m.number for m in convs["a"]] == [1, 2, 3]
    assert convs["a"][0].id == "a:1"

def test_load_handles_quotes_bom_and_escapes(tmp_path):
    # file written with "﻿" header; message cell: "hi, \"you\"\\nok 😀"
    assert convs["a"][0].content == 'hi, "you"\nok 😀'

def test_rejects_part_number_not_1(tmp_path): pytest.raises(ValueError, match="part_number")
def test_rejects_gap_in_numbers(tmp_path):    # numbers 1, 3 -> ValueError match="contiguous"
def test_rejects_duplicate(tmp_path):         # a:1 twice -> ValueError match="duplicate"
def test_parse_id_roundtrip(): assert parse_id(message_id("298954459172700181", 12)) == ("298954459172700181", 12)
def test_fingerprint_changes_with_content(tmp_path): # edit one byte -> different hex
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_dataset.py -v` — Expected: FAIL (module not found)
- [ ] **Step 3: Implement `evaluation/labeling/dataset.py`** with `csv.DictReader` and the interfaces above.
- [ ] **Step 4: Run** the same command — Expected: 7 passed
- [ ] **Step 5: Commit** `feat(labeling): load windowed-labeling dataset from the Discord CSV`

### Task 2: Queue order, windows, split, overlap metric

**Files:**
- Create: `evaluation/labeling/queue.py`
- Test: `tests/test_labeling_queue.py`

**Interfaces:**
- Consumes: `Message`, `message_id`, `parse_id` (Task 1)
- Produces:
  - `ALGORITHM_VERSION = 1`, `DEFAULT_SEED = 4240`
  - `build_queue(lengths: Mapping[str, int], seed: int) -> list[str]` — message ids in labeling order
  - `window(conversation: Sequence[Message], number: int, size: int) -> list[Message]`
  - `min_overlap_distance(queue: Sequence[str], size: int) -> int | None` — smallest queue distance between two targets of the same conversation with `0 < |i-j| < size`; `None` if no such pair
  - `assign_splits(keys: Iterable[str], seed: int, dev_fraction: float = 0.25) -> dict[str, str]` — conversation key → `"dev"`/`"test"`

- [ ] **Step 1: Write failing tests**

```python
REAL_SHAPE = {f"c{k:02d}": 100 for k in range(73)}

def test_queue_is_a_permutation():
    q = build_queue({"a": 5, "b": 100, "c": 40}, 4240)
    assert sorted(q) == sorted(f"{c}:{i}" for c, n in {"a": 5, "b": 100, "c": 40}.items() for i in range(1, n + 1))

def test_queue_is_deterministic_and_seed_sensitive():
    assert build_queue(REAL_SHAPE, 4240) == build_queue(REAL_SHAPE, 4240)
    assert build_queue(REAL_SHAPE, 4240) != build_queue(REAL_SHAPE, 4241)

def test_adjacent_positions_differ_in_conversation():
    q = build_queue(REAL_SHAPE, 4240)
    assert all(parse_id(x)[0] != parse_id(y)[0] for x, y in zip(q, q[1:]))

@pytest.mark.parametrize("size,floor", [(2, 2500), (3, 1500), (5, 1000), (10, 350), (25, 146)])
def test_overlapping_windows_are_far_apart(size, floor):
    assert min_overlap_distance(build_queue(REAL_SHAPE, 4240), size) >= floor

def test_every_prefix_spans_all_conversations():
    assert {parse_id(m)[0] for m in build_queue(REAL_SHAPE, 4240)[:73]} == set(REAL_SHAPE)

def test_window_clips_at_conversation_start(): # conv of 5 messages
    assert [m.number for m in window(conv, 2, 5)] == [1, 2]
    assert [m.number for m in window(conv, 5, 3)] == [3, 4, 5]
    assert [m.number for m in window(conv, 4, 1)] == [4]

def test_min_overlap_distance_none_when_size_1(): assert min_overlap_distance(["a:1", "a:2"], 1) is None

def test_assign_splits():
    s = assign_splits(REAL_SHAPE, 4240)
    assert sum(v == "dev" for v in s.values()) == 18 and set(s) == set(REAL_SHAPE)
    assert s == assign_splits(REAL_SHAPE, 4240)
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_queue.py -v` — Expected: FAIL
- [ ] **Step 3: Implement `build_queue`** with this exact algorithm (the thresholds above were measured on it):

```python
PHI = (5 ** 0.5 - 1) / 2
rng = random.Random(seed)
keys = sorted(lengths); rng.shuffle(keys)                 # shuffled ONCE, reused every round
seqs = {}
for c in keys:
    u = rng.random()
    seqs[c] = sorted(range(1, lengths[c] + 1), key=lambda i: ((i * PHI + u) % 1.0, i))
queue = [message_id(c, seqs[c][r]) for r in range(max(lengths.values(), default=0))
         for c in keys if r < len(seqs[c])]
```

`assign_splits` uses `random.Random(f"split:{seed}")`, shuffles `sorted(keys)`, first `round(dev_fraction * n)` are dev. `window` slices by `number` (lists are 1-based contiguous from Task 1).

- [ ] **Step 4: Run** the same command — Expected: all passed
- [ ] **Step 5: Commit** `feat(labeling): seeded queue that spreads overlapping windows apart`

### Task 3: Manifest + Contract 1A rows

**Files:**
- Create: `evaluation/labeling/manifest.py`, `evaluation/labeling/rows.py`
- Modify: `evaluation/common/schema.md` (Contract 1A section)
- Test: `tests/test_labeling_manifest.py`

**Interfaces:**
- Consumes: Tasks 1–2
- Produces:
  - `MANIFEST_PATH = io.DATA / "label_manifest.json"`
  - `@dataclass class Manifest: seed: int; csv_sha256: str; rows: int; conversations: int; algorithm_version: int`
  - `read_manifest(path: Path = MANIFEST_PATH) -> Manifest | None`; `write_manifest(m: Manifest, path: Path = MANIFEST_PATH) -> None` (indent=2, sorted keys, trailing newline)
  - `manifest_for(csv_path: Path, convs: dict[str, list[Message]], seed: int) -> Manifest`
  - `class ManifestMismatch(Exception)`; `check_manifest(expected: Manifest, actual: Manifest) -> None` — raises with a message naming each differing field; for `csv_sha256` the message includes `"your data/discord_messages.csv differs from the team's copy"`
  - `build_rows(convs: dict[str, list[Message]], queue: list[str], splits: dict[str, str]) -> list[dict]` — rows in queue order with keys `id, conversation_key, author, content, preceding_3, preceding_25, queue_position, split`; `preceding_k` = contents of up to k messages before the target, oldest → newest

- [ ] **Step 1: Write failing tests**

```python
def test_manifest_roundtrip(tmp_path): write then read equals original
def test_read_missing_manifest_is_none(tmp_path): assert read_manifest(tmp_path / "x.json") is None
def test_check_names_csv_mismatch():
    with pytest.raises(ManifestMismatch, match="differs from the team's copy"): check_manifest(m, replace(m, csv_sha256="0"))
def test_check_passes_identical(): check_manifest(m, m)
def test_build_rows_shape():
    rows = build_rows(convs, queue, splits)
    assert [r["queue_position"] for r in rows] == list(range(len(queue)))
    r = next(r for r in rows if r["id"] == "a:3")
    assert r["preceding_3"] == [convs["a"][0].content, convs["a"][1].content]
    assert set(r) == {"id", "conversation_key", "author", "content", "preceding_3", "preceding_25", "queue_position", "split"}
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_manifest.py -v` — Expected: FAIL
- [ ] **Step 3: Implement** both modules per the interfaces.
- [ ] **Step 4: Run** — Expected: 5 passed
- [ ] **Step 5: Update `evaluation/common/schema.md` Contract 1A** to the new row shape (drop `channel_id`, `author_id`, `created_at`; add `conversation_key`, `author`, `queue_position`; note it's generated by `pixi run label-import`, gitignored, ~7,300 rows, split by conversation). Leave `sample_messages.jsonl` as is — loaders only read `id`, `content`, `preceding_*`, `split`.
- [ ] **Step 6: Commit** `feat(labeling): CSV fingerprint manifest and generated messages.jsonl rows`

### Task 4: PocketBase collections + `label-import`

**Files:**
- Create: `pb_migrations/1762400000_labeling.js`, `evaluation/labeling/importer.py`
- Modify: `pyproject.toml` (`[tool.pixi.tasks]`), `tests/test_schema.py` (`COLLECTIONS`)
- Test: `tests/test_labeling_import.py`

**Interfaces:**
- Consumes: Tasks 1–3
- Produces:
  - Collection `label_messages`: text `message_id`, text `conversation_key`, number `message_number`, text `author`, text `content`, number `queue_position`, number `algorithm_version`. Indexes: unique `message_id`, unique `queue_position`, `(conversation_key, message_number)`.
  - Collection `labels`: text `message_id`, text `labeler`, number `window_size`, select `label` (values = LABELS, maxSelect 1), number `queue_position`, number `algorithm_version`, autodate `created`/`updated`. Unique index `(labeler, window_size, message_id)`; index `(labeler, window_size, queue_position)`.
  - Both with all rules `""` like the init migration; the down migration deletes both.
  - `run_import(pb_url: str, csv_path: Path = CSV_PATH, manifest_path: Path = MANIFEST_PATH, messages_path: Path = io.MESSAGES, seed: int | None = None, new_manifest: bool = False) -> int` — returns rows synced
  - CLI `python -m evaluation.labeling.importer [--pb URL] [--new-manifest] [--seed N]`; pixi task `label-import`. `--pb` defaults to `VIBESHIFT_PB_URL` env or `http://127.0.0.1:8090`.

Behaviour of `run_import`, in order:
1. Load CSV, compute `actual = manifest_for(...)` (seed = `seed` arg, else the manifest's, else `DEFAULT_SEED`).
2. No manifest and not `new_manifest` → `SystemExit` telling Lucas to run with `--new-manifest`. Manifest present and not `new_manifest` → `check_manifest(expected, actual)`; on mismatch `SystemExit(str(err))` **before any PocketBase call**. `new_manifest` → write it.
3. Build queue, splits, rows; write `messages_path` via `io.write_jsonl`.
4. Sync `label_messages`: fetch all existing records once (paged, `perPage=500`, `fields=id,message_id,queue_position,content,algorithm_version`), create missing, patch changed, leave identical rows alone. Use a sync `httpx.Client`.
5. Print a table of `min_overlap_distance` for W in (2, 3, 5, 10, 25) and the synced count.

- [ ] **Step 1: Write failing tests** (use the existing `pb_url` fixture; small CSV of 3 conversations in `tmp_path`; manifest/messages paths in `tmp_path`)

```python
def test_collections_exist: add "label_messages", "labels" to COLLECTIONS in tests/test_schema.py

def test_import_creates_rows_and_messages_file(pb_url, tmp_path):
    n = run_import(pb_url, csv, manifest, msgs, new_manifest=True)
    assert n == total_messages and read_manifest(manifest).rows == total_messages
    recs = list all label_messages
    assert sorted(r["queue_position"] for r in recs) == list(range(total_messages))
    assert len(io.read_jsonl(msgs)) == total_messages

def test_reimport_same_csv_is_idempotent(pb_url, tmp_path):
    run twice -> same record count, each message_id keeps the same id and queue_position

def test_import_refuses_changed_csv_before_touching_pb(pb_url, tmp_path):
    run_import(..., new_manifest=True); edit one message in csv
    with pytest.raises(SystemExit, match="differs from the team's copy"): run_import(...)
    assert stored contents unchanged

def test_import_without_manifest_requires_flag(pb_url, tmp_path):
    with pytest.raises(SystemExit, match="--new-manifest"): run_import(pb_url, csv, tmp_path / "none.json", msgs)

def test_labels_unique_per_labeler_window_message(pb_url):
    create the same (labeler, window_size, message_id) twice -> second POST returns 400
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_schema.py tests/test_labeling_import.py -v` — Expected: FAIL
- [ ] **Step 3: Write the migration** following `pb_migrations/1760700000_init.js` (helpers `t`, `n`, `sel`, autodate fields as in `backfill_jobs`).
- [ ] **Step 4: Implement `importer.py`** and add the pixi task: `label-import = "python -m evaluation.labeling.importer"`.
- [ ] **Step 5: Run** — Expected: all passed
- [ ] **Step 6: Real-data smoke check:** `pixi run pb` in one terminal, then `pixi run label-import --new-manifest`. Expected: `7300` rows synced and an overlap table matching Task 2's floors (W=2 ≥ 2500 … W=25 ≥ 146). Commit the generated `evaluation/data/label_manifest.json`.
- [ ] **Step 7: Commit** `feat(labeling): PocketBase label collections and label-import task`

### Task 5: Dashboard labeling data layer

**Files:**
- Modify (rewrite): `dashboard/src/lib/labeling.ts`, `dashboard/src/lib/labeling.test.ts`

**Interfaces:**
- Consumes: Task 4 collections; `pb` from `@/lib/pb`; `LABELS`, `Label` from `@/lib/emotions`
- Produces:
  - `type LabelTarget = { message_id: string; conversation_key: string; message_number: number; author: string; content: string; queue_position: number; algorithm_version: number }`
  - `type WindowMessage = { message_id: string; author: string; content: string; isTarget: boolean }`
  - `normalizeLabeler(name: string): string` — trim, lowercase, runs of non `[a-z0-9]` → `-`, strip leading/trailing `-`; `''` if nothing remains
  - `windowRange(messageNumber: number, size: number): { from: number; to: number }`
  - `firstGap(positions: number[]): number` — smallest non-negative integer not in `positions`
  - `fetchQueueSize(): Promise<number>` — `totalItems` of `label_messages` (0 if not imported)
  - `fetchTarget(position: number): Promise<LabelTarget | null>`
  - `fetchWindow(target: LabelTarget, size: number): Promise<WindowMessage[]>` — oldest → newest, last has `isTarget: true`
  - `fetchLabeledPositions(labeler: string, size: number): Promise<number[]>` — `getFullList` on `labels`, `fields: 'queue_position'`
  - `fetchLabel(labeler: string, size: number, messageId: string): Promise<Label | null>`
  - `saveLabel(labeler: string, size: number, target: LabelTarget, label: Label): Promise<void>` — update the existing row matching the unique triple, else create
  - `labelerStorageKey = 'vibeshift.labeler'`, `windowSizeStorageKey = 'vibeshift.labelWindow'`, `DEFAULT_WINDOW_SIZE = 5`, `MAX_WINDOW_SIZE = 25`
- Removes: `shuffle`, `fetchLabelQueue`, `LabelItem`

- [ ] **Step 1: Write failing tests** for the pure helpers only (PocketBase calls stay thin, tested via Task 4 and the manual check in Task 6)

```ts
expect(normalizeLabeler('  Lucas S. ')).toBe('lucas-s')
expect(normalizeLabeler('Ana_María')).toBe('ana-mar-a')
expect(normalizeLabeler('!!!')).toBe('')
expect(windowRange(2, 5)).toEqual({ from: 1, to: 2 })
expect(windowRange(10, 3)).toEqual({ from: 8, to: 10 })
expect(windowRange(4, 1)).toEqual({ from: 4, to: 4 })
expect(firstGap([])).toBe(0)
expect(firstGap([0, 1, 2])).toBe(3)
expect(firstGap([2, 0, 1, 4])).toBe(3)
```

- [ ] **Step 2: Run** `npm --prefix dashboard test -- --run src/lib/labeling.test.ts` — Expected: FAIL
- [ ] **Step 3: Implement** the module; filters use the same quoting helper style as `lib/data.ts` (`q()`), or `pb.filter()` with params.
- [ ] **Step 4: Run** — Expected: PASS; also `npm --prefix dashboard run build` succeeds once Task 6 lands (the old page still imports removed names — do Task 6 before building)
- [ ] **Step 5: Commit** together with Task 6 (the page and its data layer change together).

### Task 6: Label page rebuild

**Files:**
- Modify (rewrite): `dashboard/src/pages/label.tsx`

**Interfaces:**
- Consumes: everything Task 5 produces

- [ ] **Step 1: Labeler gate.** If `localStorage[labelerStorageKey]` is empty (wrap in try/catch), show a card with a name input; on submit store `normalizeLabeler(name)` (reject `''`). Show the current labeler with a "Change" button in the page header.
- [ ] **Step 2: Window size control.** Numeric input (shadcn `Input type="number"`, min 1, max `MAX_WINDOW_SIZE`), persisted in `windowSizeStorageKey`, default `DEFAULT_WINDOW_SIZE`. Changing it reloads labeled positions for that size and jumps to `firstGap`.
- [ ] **Step 3: Labeling loop.**
  - On load / size change: `fetchQueueSize()`; if 0 render the Empty state with title "Nothing imported yet" and description "Run `pixi run label-import` with PocketBase running." Fetch errors render `ErrorAlert`.
  - `position = firstGap(fetchLabeledPositions(...))`; if `position >= size` render "All caught up" with the count.
  - Load `fetchTarget(position)` + `fetchWindow(target, W)` + `fetchLabel(...)` (for Back).
  - Render context messages oldest → newest as `author` + text (`whitespace-pre-wrap`, muted), and the target last, larger, with a highlighted border and a "Label this message" caption. Never render conversation key, message number or queue position of the target.
  - Picking (button or keys 1–7) → `saving = true`, `await saveLabel`, advance to the next unlabeled position (`firstGap` over the updated set), `saving = false`. Ignore picks and keys while `saving`; ignore `e.repeat` as today.
  - **Back** button (and `Backspace` key): go to the previous position in queue order (`position - 1`, min 0) showing its saved label as the selected toggle; picking re-saves and returns to the first gap.
  - Description line: `Window {position + 1} of {size} · {W} message window · {labeledCount} labeled`, with `Progress` at `labeledCount / size`.
- [ ] **Step 4: Verify.** `npm --prefix dashboard test -- --run && npm --prefix dashboard run build && npm --prefix dashboard run lint` — Expected: all pass. Then with `pixi run pb` + `pixi run label-import` + `pixi run dash`: enter a name, label 3 windows at W=5, reload (resumes at window 4), switch to W=3 (starts at window 1, same first target message), Back changes a label (PocketBase `labels` still has 3 rows for W=5), mash a key quickly (one row per window, none skipped).
- [ ] **Step 5: Commit** `feat(ui): windowed Label page backed by PocketBase with resume and window size`

### Task 7: `label-export`

**Files:**
- Create: `evaluation/labeling/exporter.py`
- Modify: `pyproject.toml` (pixi task), `evaluation/common/io.py` (add `LABEL_FILES = ANNOTATIONS / "labels"`)
- Test: `tests/test_labeling_export.py`

**Interfaces:**
- Consumes: Task 4 `labels` collection
- Produces:
  - `export_labels(pb_url: str, out_dir: Path = io.LABEL_FILES) -> list[Path]` — one file per (labeler, window_size): `{labeler}_w{window_size}.jsonl`, rows sorted by `queue_position`, each `{"id", "labeler", "window_size", "label", "queue_position", "algorithm_version"}`; overwrites that file only
  - CLI `python -m evaluation.labeling.exporter [--pb URL]`; pixi task `label-export`

- [ ] **Step 1: Write failing tests** (`pb_url` fixture; POST label rows directly)

```python
def test_export_one_file_per_labeler_and_window(pb_url, tmp_path):
    # rows: (lucas, 5) x2, (lucas, 3) x1, (ana, 5) x1
    paths = export_labels(pb_url, tmp_path)
    assert sorted(p.name for p in paths) == ["ana_w5.jsonl", "lucas_w3.jsonl", "lucas_w5.jsonl"]
    assert [r["queue_position"] for r in io.read_jsonl(tmp_path / "lucas_w5.jsonl")] == [0, 1]

def test_export_filename_uses_slug(pb_url, tmp_path): labeler "lucas-s" -> "lucas-s_w5.jsonl"
def test_export_leaves_other_files(pb_url, tmp_path): pre-existing "zed_w5.jsonl" untouched
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_export.py -v` — Expected: FAIL
- [ ] **Step 3: Implement** with a paged fetch of `labels` (`perPage=500`).
- [ ] **Step 4: Run** — Expected: 3 passed
- [ ] **Step 5: Commit** `feat(labeling): export per-labeler label files for merging`

---

## Phase B — evaluation contract (after team sign-off on the schema change)

### Task 8: Consensus per window size

**Files:**
- Create: `evaluation/annotation/consensus.py`
- Modify: `evaluation/common/io.py`, `evaluation/common/schema.md` (Contract 1B), `pyproject.toml`
- Test: `tests/test_labeling_consensus.py`

**Interfaces:**
- Consumes: Task 7 files
- Produces:
  - `io.load_label_files(window: int, directory: Path = LABEL_FILES) -> dict[str, dict[str, str]]` — labeler → {id → label}, from `*_w{window}.jsonl`
  - `io.consensus_path(window: int) -> Path` = `ANNOTATIONS / f"consensus_w{window}.jsonl"`; `io.overrides_path(window: int) -> Path` = `ANNOTATIONS / f"overrides_w{window}.json"` (`{"<id>": "<label>"}`)
  - `io.consensus_windows() -> list[int]` — sorted W values that have a consensus file
  - `io.load_consensus(path: Path | None = None, window: int | None = None)` — `window` given → that file; else if exactly one `consensus_w*.jsonl` exists → it; else if several → `SystemExit` listing them and asking for `--window`; else the existing sample fallback
  - `build_consensus(by_labeler: dict[str, dict[str, str]], overrides: dict[str, str]) -> tuple[list[dict], list[str]]` — returns (rows, unresolved tie ids). Ids = intersection over labelers (the shared prefix). Row: `{"id", "label", "soft_probs"}`; `label` = strict majority, else `overrides[id]` if present, else id goes to the tie list and is left out. `soft_probs` = share of labelers per label, keys exactly `LABELS`. Rows in `queue_position` order if available, else sorted id.
  - CLI `python -m evaluation.annotation.consensus --window W`; pixi task `label-consensus`. Prints counts and the tie list; refuses with fewer than 2 labelers.

- [ ] **Step 1: Write failing tests**

```python
def test_majority_and_soft_probs():
    rows, ties = build_consensus({"a": {"x": "joy"}, "b": {"x": "joy"}, "c": {"x": "anger"}}, {})
    assert rows[0]["label"] == "joy" and rows[0]["soft_probs"]["joy"] == pytest.approx(2 / 3) and ties == []
    assert set(rows[0]["soft_probs"]) == set(LABELS)

def test_tie_needs_override():
    by = {"a": {"x": "joy"}, "b": {"x": "anger"}}
    assert build_consensus(by, {}) == ([], ["x"])
    assert build_consensus(by, {"x": "anger"})[0][0]["label"] == "anger"

def test_only_shared_prefix(): a labeled x,y; b labeled x -> only x

def test_load_consensus_requires_window_when_ambiguous(tmp_path, monkeypatch):
    # point io.ANNOTATIONS at tmp_path with consensus_w3 and consensus_w5 -> SystemExit match="--window"
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_consensus.py -v` — Expected: FAIL
- [ ] **Step 3: Implement**; update Contract 1B in `schema.md` (per-labeler files, no flags, `consensus_w{W}.jsonl`, overrides file).
- [ ] **Step 4: Run** — Expected: 4 passed
- [ ] **Step 5: Commit** `feat(eval): per-window consensus from per-labeler label files`

### Task 9: Agreement (`eval-kappa`)

**Files:**
- Modify (rewrite): `evaluation/annotation/reliability.py`
- Test: `tests/test_labeling_reliability.py`

**Interfaces:**
- Consumes: `io.load_label_files` (Task 8)
- Produces: `fleiss_kappa(counts: list[list[int]]) -> float` (rows = items, cols = categories, equal raters per row); `pairwise_cohen(by_labeler) -> dict[tuple[str, str], float]` (sklearn `cohen_kappa_score` on shared ids); CLI `--window W` (required) printing both plus the disagreement list.

- [ ] **Step 1: Write failing tests**

```python
def test_fleiss_reference_example():  # Fleiss (1971) example, 10 items x 14 raters x 5 categories
    T = [[0,0,0,0,14],[0,2,6,4,2],[0,0,3,5,6],[0,3,9,2,0],[2,2,8,1,1],
         [7,7,0,0,0],[3,2,6,3,0],[2,5,3,2,2],[6,5,2,1,0],[0,2,2,3,7]]
    assert fleiss_kappa(T) == pytest.approx(0.2099, abs=1e-4)

def test_pairwise_cohen_perfect():
    assert pairwise_cohen({"a": {"x": "joy", "y": "anger"}, "b": {"x": "joy", "y": "anger"}}) == {("a", "b"): pytest.approx(1.0)}
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_reliability.py -v` — Expected: FAIL
- [ ] **Step 3: Implement**; drop the `is_context_dependent` / `is_sarcastic` fields and the `annotator_a/b` paths.
- [ ] **Step 4: Run** — Expected: 2 passed
- [ ] **Step 5: Commit** `feat(eval): pairwise Cohen and Fleiss kappa per window size`

### Task 10: Evaluation scripts on the new gold

**Files:**
- Modify: `evaluation/eval/evaluate.py`, `evaluation/eval/frames.py`, `evaluation/annotation/tau_sweep.py`, `evaluation/models/stacking/train.py`, `evaluation/README.md`
- Test: `tests/test_labeling_eval.py`

**Interfaces:**
- Consumes: `io.load_consensus(window=...)`, `io.consensus_windows()` (Task 8)
- Produces: `io.context_sensitive_ids() -> set[str] | None` — `None` if fewer than 2 consensus files; else ids present in both the smallest-W and largest-W consensus whose labels differ. `evaluate_model(rows, gold, tau, context_ids: set[str] | None)`.

- [ ] **Step 1: Write failing tests**

```python
def test_context_sensitive_ids(tmp_path, monkeypatch):
    # consensus_w1: x=joy, y=anger ; consensus_w5: x=joy, y=sadness -> {"y"}
def test_context_sensitive_none_with_one_file(...): -> None
def test_message_frame_has_no_flag_columns(): "is_sarcastic" not in frames.message_frame(...).columns
```

- [ ] **Step 2: Run** `pixi run pytest tests/test_labeling_eval.py -v` — Expected: FAIL
- [ ] **Step 3: Implement.**
  - `frames.message_frame`: drop `is_sarcastic` / `is_context_dependent`.
  - `evaluate.py`: drop `macro_f1_sarcastic`; `macro_f1_context` = Macro-F1 over `context_ids` (NaN when `None`); column header "Macro-F1 (ctx-sensitive)".
  - Add `--window` (int, optional) to `evaluate.py`, `tau_sweep.py`, `stacking/train.py`, passed to `io.load_consensus(window=...)`; include `W` in the report header.
  - Fix any explorer notebook cell that reads the removed flag columns.
  - README: replace the A/B annotation flow with `label-import` → label → `label-export` → commit → `label-consensus --window W` → `eval-kappa --window W` → `eval-report --window W`.
- [ ] **Step 4: Run** `pixi run test` — Expected: whole suite passes. Then `pixi run eval-report` on the sample fixture still runs (sample fallback path).
- [ ] **Step 5: Commit** `feat(eval): score against per-window gold; context-sensitive slice replaces flags`
