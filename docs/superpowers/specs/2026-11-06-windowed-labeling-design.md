# Windowed labeling — design

Date: 2026-11-06 · Owner: Lucas · Status: draft for review

## Why

The evaluation needs human gold labels for the 7 emotion labels. Emotion in chat
often depends on what came just before, so labelers should see a **sliding window**
of the conversation and label only its **last message** (the target). Two problems
with doing that naively:

1. **Context leakage across windows.** Windows that end close together in the same
   conversation share messages. If a labeler sees them near each other, they carry
   memory of context that is *outside* the window they are currently judging, so the
   label no longer reflects "this window only".
2. **No link between labels and data.** The current Label page keeps picks in page
   memory with placeholder messages; there is no schema tying a label to a message,
   a labeler, and the context it was judged in.

We also don't know how far the team will get, so labeling must be useful at any
stopping point: if Lucas stops at 500, teammates who reach 500 must have labeled
**exactly the same 500 windows**, so the sets are directly comparable.

## Goals

- Every message in `data/discord_messages.csv` is the target of exactly one window
  (per window size), so every message can get a score.
- Windows never cross a conversation (`conversation_key`) boundary.
- One fixed, seeded queue order shared by all labelers; progress is a prefix of it.
- Windows that share messages are as far apart in the queue as possible, and any
  prefix (e.g. the first 500) is already a good spread across all conversations.
- Window size is set in the GUI. Labels at different window sizes are stored and
  analysed separately, never mixed.
- Labels persist in PocketBase so a labeler can stop and resume.
- Each labeler's labels can be merged with teammates' to compute agreement and gold.
- The labeler picks one of the 7 labels and nothing else.

## Non-goals

- No `is_sarcastic` / `is_context_dependent` flags (decided: "just the 7").
- No shared/hosted PocketBase server; each person runs their own locally.
- No per-labeler randomisation — everyone gets the same order on purpose.
- No adjudication UI; disagreements are listed by a script, as today.

## Anti-goals (would make this a failure even if it "works")

- Two labelers' "first 500" differ in any way (different targets, order, or
  context shown) because their CSV copies, seeds, or code differ silently.
- Same-conversation overlapping windows appear close together in the queue.
- Labels from different window sizes end up in the same gold file.

## Data source

`data/discord_messages.csv` (gitignored). Columns:
`conversation_key, conversation_message_number, part_number, author, message`.
Current file: 7,300 rows, 73 conversations of 100 messages each, message numbers
contiguous from 1, `part_number` always 1, no empty messages. Message text contains
literal `\n` escape sequences that mean line breaks. The file starts with a UTF-8 BOM
(read with `utf-8-sig`).

**Message id** (stable, used everywhere): `"{conversation_key}:{conversation_message_number}"`
(e.g. `298954459172700181:3`). `part_number` is ignored while it is always 1; the
builder should fail loudly if any row has `part_number != 1` rather than guess.

## Windows

For window size **W** and target message *i* (1-based within its conversation), the
window is messages `max(1, i-W+1) … i` of the same conversation. The first W-1
messages of a conversation get shorter windows (message 1 has no context). W counts
the target, so W=1 is "message alone".

## Queue order (the core of this design)

The queue is an ordering of all **targets** (message ids). It is **independent of W**
— the same order is used for every window size — so the first N targets at W=3 and
at W=10 are the same N messages. This makes "how much does context change the label"
directly measurable, and means one order to reason about.

Requirements on the order, given seed *s*:

1. Deterministic from (CSV contents, seed). Same inputs → byte-identical order.
2. **Interleave conversations**: adjacent queue positions come from different
   conversations; each conversation's targets are spread evenly through the queue
   (one per "round" across all 73 conversations). The conversation order is
   shuffled by the seed **once** and reused every round, so the same conversation
   recurs exactly every 73 positions (a fresh shuffle per round could put one
   conversation at the end of round r and the start of round r+1).
3. **Scatter within a conversation**: a conversation's own targets are *not* visited
   in reading order. Targets close together in the conversation (|i-j| small, which
   is exactly when their windows overlap) must be visited far apart in that
   conversation's sequence.
4. Good for any prefix and for any reasonable W (target: 1–25) without recomputing.

Suggested approach (builder may improve on it, keeping the properties): per
conversation, order targets by `frac(i·φ + u_c)` where φ is the golden ratio and
`u_c` is a seeded per-conversation offset — this sends neighbours *i, i+1* roughly
0.618 of the conversation apart in visit order. Then round-robin across
conversations in the once-shuffled conversation order.

**Verification**: a test computes, for W in {2, 3, 5, 10, 25}, the minimum queue
distance between any two targets whose windows share a message, and asserts it is
large (with 73 conversations, round-robin alone gives ≥ ~73; scattering should push
it far higher). The import script prints this table so the team can see it.

## Fingerprint / reproducibility

A committed manifest `evaluation/data/label_manifest.json`:

```json
{ "seed": 4240, "csv_sha256": "…", "rows": 7300, "conversations": 73, "algorithm_version": 1 }
```

The import step hashes the local CSV and **refuses to proceed** if it doesn't match
the manifest (with a clear message: "your data/discord_messages.csv differs from the
team's copy"). Creating/replacing the manifest is an explicit flag
(`--new-manifest`), used once by Lucas. Changing the algorithm bumps
`algorithm_version`, which invalidates existing progress — the export records it so
mixed versions are detectable.

## Storage (PocketBase, local per person)

New migration (append-only, per the existing migration rule) adding two collections:

**`label_messages`** — the dataset, imported from the CSV.

| field | type | notes |
| --- | --- | --- |
| `message_id` | text, unique | `conversation_key:number` |
| `conversation_key` | text | |
| `message_number` | number | 1-based within the conversation |
| `author` | text | |
| `content` | text | `\n` escapes converted to real newlines on import |
| `queue_position` | number, unique | 0-based position in the shared order |

Index on `(conversation_key, message_number)` for window lookup.

**`labels`** — one row per (labeler, window size, target).

| field | type | notes |
| --- | --- | --- |
| `message_id` | text | the target |
| `labeler` | text | name the labeler enters once in the GUI |
| `window_size` | number | W the label was made at |
| `label` | select | one of the 7 `LABELS` |
| `queue_position` | number | copied from the target, so resume is a simple lookup |
| `algorithm_version` | number | from the manifest |
| `created` / `updated` | autodate | |

Unique index on `(labeler, window_size, message_id)`; re-labeling updates the row.

Import task: `pixi run label-import` — checks the fingerprint, builds the order,
upserts `label_messages`. Idempotent; safe to re-run.

## Label page (dashboard)

Replaces the current placeholder page at Evaluation › Label data.

- **Labeler name**: asked once, remembered in the browser; shown and changeable.
- **Window size**: a numeric control in the page. Changing it switches to that
  size's run and resumes at the labeler's first unlabeled queue position for that W.
- **Resume**: on load, find the lowest `queue_position` with no `labels` row for
  (labeler, W) and start there.
- **Display**: the window's messages oldest → newest with author names; the target
  is the last one and visually distinct (larger, highlighted) so it's unambiguous
  what is being labeled. Real line breaks render.
- **Input**: the 7 label buttons, keys 1–7, as today. Picking saves to PocketBase,
  then advances.
- **Back**: go to the previous window and change its label (updates the row).
- **Progress**: "Window 512 of 7,300 · W=5", plus a count labeled.
- Never show conversation key, message number, or anything else that hints where the
  window sits in its conversation.

## Merging with teammates

Each person labels in their own local PocketBase, then runs
`pixi run label-export`, which writes one file per (labeler, W):

`evaluation/data/annotations/labels/<labeler>_w<W>.jsonl`

```json
{"id": "298954459172700181:3", "labeler": "lucas", "window_size": 5, "label": "joy", "queue_position": 0, "algorithm_version": 1}
```

These are committed. Each person only writes their own files, so git never
conflicts. Files contain ids and labels only, no message text.

## Changes to the frozen evaluation contracts

The new CSV **replaces** the curated 300-message dataset. `evaluation/common/schema.md`
says contract changes need team agreement — this section is what to bring to them.

**Contract 1A (`messages.jsonl`)** — now generated from the CSV by a pixi task
(`label-import` or a sibling), all 7,300 rows:

- `id` → `conversation_key:number`
- `channel_id` → `conversation_key`; `author_id` → `author`
- `created_at` → removed (the CSV has no timestamps)
- `content`, `preceding_3`, `preceding_25` → unchanged meaning, cut from the same
  conversation (shorter near the start)
- `queue_position` → new
- `split` → see below

**Contract 1B (annotations)** — `annotator_a.jsonl` / `annotator_b.jsonl` are replaced
by the per-labeler files above. Flags `is_context_dependent` / `is_sarcastic` are
removed. `consensus.jsonl` is built **per window size**
(`consensus_w<W>.jsonl`), for the targets in the common prefix every labeler has
finished at that W:

- `label` = majority across labelers; ties go on the adjudication list
  (the script prints them; the adjudicated choice is recorded in a small
  committed overrides file).
- `soft_probs` = share of labelers choosing each label (keys exactly `LABELS`).

`eval-report` / `eval-tau` take `--window W` to pick which consensus file is gold.
`eval-kappa` reports pairwise Cohen's κ for every labeler pair plus Fleiss' κ,
per W.

**Consequences to accept:**

- Sliced Macro-F1 by sarcasm / context-dependence loses its source. Suggested
  replacement slice: "context-sensitive" = targets whose consensus label differs
  between the smallest and largest W labeled.
- The model runners still read `messages.jsonl`, but gold now exists only for the
  labeled prefix; evaluation scores only ids that have gold. Runners may be limited
  to the first N by `queue_position` to save cost.

**Split**: assign whole *conversations* to dev or test (seeded, ~25% dev) so a
conversation's messages never leak between tuning and testing. Because the queue
interleaves conversations, any prefix has roughly the same dev/test ratio.

## Decided by Lucas

- Windows end on the target; only the last message is labeled; every message is a
  target once.
- Windows stay within one conversation.
- One shared seed; everyone labels the same order; progress is a prefix.
- Overlapping windows as far apart as possible in the queue.
- Window size is set in the GUI; each window size is stored separately.
- Labels stored in PocketBase so labeling can be resumed.
- Only the 7 labels; no extra flags.
- This dataset replaces the 300-message curated plan.
- `data/` is gitignored.

## Calls made by Claude (check these)

- Queue order is the same for every window size.
- Message id format `conversation_key:number`; `part_number` ignored (fail if ≠ 1).
- Fingerprint manifest committed; import refuses on mismatch.
- Per-person local PocketBase + committed per-labeler export files for merging.
- Authors shown in the window; no positional hints shown.
- Back button allowed (labels can be revised).
- Consensus per W by majority; ties adjudicated via an overrides file; Fleiss' κ.
- Dev/test split by conversation, ~25% dev.
- "Context-sensitive" slice replaces the dropped flags.
- `created_at` dropped from Contract 1A.
- Message text is treated as private: `evaluation/data/messages.jsonl` is gitignored
  (regenerated locally from the CSV), like `data/`. Only ids, labels, and the
  manifest are committed.

## Left to the builder

- Exact within-conversation scatter function, as long as the verification test
  passes and the order is deterministic.
- Seed value.
- UI layout details within the existing shadcn dashboard style.
- Whether export is a pixi task only or also a button on the page.
