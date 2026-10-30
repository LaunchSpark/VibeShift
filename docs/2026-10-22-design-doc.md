# VibeShift — Design Doc

**CSC 4240/5240: Artificial Intelligence**
**Team:** Lucas Starkey, Trigg Lampkins, Trey Gannod, Braylon Trail, Isaiah Chastain
**Created:** 2026-10-22 · **Updated:** 2026-10-29 (7-label model and `evaluation/` merged)

> Background design detail lives in
> [`docs/superpowers/specs/2026-10-17-emotion-dashboard-design.md`](superpowers/specs/2026-10-17-emotion-dashboard-design.md)
> and the plans beside it (those describe the original 13-label system). This doc is
> the single source of truth for the 7-label space, the τ rule, and the work breakdown.

**Scope (Job A, locked):** one 7-label schema everywhere; our fine-tuned Laya plus a
tuned neutral cutoff τ; an offline benchmark of Laya against transformer, LLM and
lexicon baselines; channel and member mood trends on the dashboard. Stretch goals:
fine-tuning on labeled Discord messages, and member-influence analysis (Job B). We
make no accuracy claims in advance: Laya on Discord text is reported only through
the benchmark.

## §1 System Overview — `[BUILT]` vs `[TODO]`

| Component | Status | Code |
| --- | --- | --- |
| Discord capture + offline backfill | `[BUILT]` | [`bot.py`](../src/vibeshift/bot.py), [`history.py`](../src/vibeshift/history.py) |
| Scorer worker (drains pending deque, maintains rollups) | `[BUILT]` | [`scorer.py`](../src/vibeshift/scorer.py) |
| PocketBase storage + JSON daily rollups | `[BUILT]` | [`store.py`](../src/vibeshift/store.py), [`rollups.py`](../src/vibeshift/rollups.py) |
| React dashboard with SMA trend view | `[BUILT]` | [`emotions.tsx`](../dashboard/src/pages/emotions.tsx), [`series.ts`](../dashboard/src/lib/series.ts) |
| 7-label Laya checkpoint on Hugging Face | `[BUILT]` | `LaunchSpark/laya-emotion-tracker@v2-7-labels`, [`finetune_laya_emotions.py`](../notebooks/finetune_laya_emotions.py) |
| Dashboard 13→7 relabel | `[BUILT]` | [`emotions.ts`](../dashboard/src/lib/emotions.ts), [`seed.ts`](../dashboard/scripts/seed.ts) |
| Label data page (annotation UI, placeholder queue) | `[BUILT]` | [`labeling.ts`](../dashboard/src/lib/labeling.ts) |
| `evaluation/` pipeline on the 10-row sample fixture | `[BUILT]` | [`evaluation/`](../evaluation/) |
| `VIBESHIFT_TAU` + τ-gated rollup counting + `scored_at` | `[TODO]` | `scorer.py`, `store.py`, `config.py` |
| Dashboard τ view | `[TODO]` | [`data.ts`](../dashboard/src/lib/data.ts) |
| 300-message offline benchmark (real data) | `[TODO]` | [`evaluation/`](../evaluation/) |

## §2 Unified 7-Label Space & Universal τ Rule

The canonical label set, frozen in [`evaluation/common/labels.py`](../evaluation/common/labels.py)
and mirrored by the dashboard's `LABELS` and the checkpoint's `questions.json`:

```python
LABELS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")
```

This one set is used for human annotation, every benchmark metric,
`messages.probabilities`, `daily_rollups.prob_sums` and the dashboard.
`evaluation/common/test_labels.py` (`pixi run test-model`) guards against drift
between the schema and the live model.

### Models and how each reaches the 7 labels

- **Laya:** `LaunchSpark/laya-emotion-tracker` at tag `v2-7-labels`, fine-tuned from
  `convaiinnovations/laya` with [`notebooks/finetune_laya_emotions.py`](../notebooks/finetune_laya_emotions.py):
  - **Data:** [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
    (Reddit comments) mapped to the 7 labels with the table below, using its official
    train/validation/test splits.
  - **Training set:** up to 1,000 comments per label (fear and disgust have only 538
    and 495, so all are used).
  - **Results:** test macro-F1 0.594, against 0.437 for the untuned base model. Full
    results in [`docs/results/2026-10-23-goemotions/`](results/2026-10-23-goemotions/).
  - The previous 13-label model is tagged `v1-13-labels`.
  - [`emotions.py`](../src/vibeshift/emotions.py) pins `MODEL_REVISION` and reads the
    labels from `questions.json`; `MODEL_ID` records the revision, so each message's
    `model_id` says which checkpoint scored it.
- **DistilRoBERTa-Emotion** (`j-hartmann/emotion-english-distilroberta-base`): outputs
  exactly the 7 labels.
- **RoBERTa-GoEmotions** (`SamLowe/roberta-base-go_emotions`): 28 labels, mapped with
  Google's GoEmotions → Ekman grouping below. Its scores are independent sigmoids, so
  we sum within each group and normalize the 7 totals to 1.
- **Zero-shot LLM:** returns a JSON probability per label, renormalized to sum to 1.
- **NRC EmoLex:** 7-class word-count baseline with neutral fallback, plus the M=25
  macro-window mood vector used by the stacking head.

| Label | GoEmotions labels |
|---|---|
| joy | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| sadness | sadness, disappointment, embarrassment, grief, remorse |
| anger | anger, annoyance, disapproval |
| fear | fear, nervousness |
| disgust | disgust |
| surprise | surprise, realization, confusion, curiosity |
| neutral | neutral |

**Sarcasm and context-dependence are not output classes.** They are boolean
annotation flags (`is_sarcastic`, `is_context_dependent`) used to slice the results.

### The τ rule

Both production rollups and the offline benchmark gate a message to a final label
with **one rule**: a message is `neutral` if its top class is `neutral` **or**
`max_{e != "neutral"} p_e < τ`; otherwise it keeps its top class.

- `τ = 0.0` is plain argmax: the ungated baseline reported next to τ\*.
- A model with no neutral output treats `p(neutral) = 0`, so the rule becomes
  "neutral iff its top emotion is below τ".
- Raw probabilities are stored un-thresholded, so τ is a pure post-processing choice
  (see [`evaluation/eval/metrics.py`](../evaluation/eval/metrics.py) `gated_label`;
  the scorer will apply the same rule once `VIBESHIFT_TAU` lands — §7).

## §3 PocketBase & Rollup Storage

- Raw 7-class probabilities are stored on `messages.probabilities` (JSON), with the
  τ-gated label in `messages.emotion`, its score `p(e*)` in `messages.confidence`, and
  the checkpoint in `messages.model_id`.
- `daily_rollups.prob_sums` accumulates **τ-gated top-label counts**, not raw
  probability sums (the field keeps its name, so the schema doesn't change). The
  dashboard then plots the share of messages per emotion. `series.ts` needs no change:
  it already divides the summed values by `count`.
- **Why counts (measured).** On GoEmotions' test split (4,922 comments), at τ = 0 with
  `v2-7-labels`, the Jensen–Shannon divergence (base 2) from the human label mix was
  **0.007** for the share of top labels against **0.031** for the mean of the full
  probability vectors. Laya is fine-tuned on balanced classes, so averaging its
  probabilities inflates rare emotions: fear and disgust average 0.054–0.057 against
  a true share of 0.015–0.016. A prior correction (Saerens et al., 2002) was not
  adopted: it needs each server's label mix and adds a runtime step. Numbers in
  [`results.json`](results/2026-10-23-goemotions/results.json).
- **Changing the configured τ requires a rollup rebuild** from
  `messages.probabilities` (one-off script, §7).
- Add `scored_at` to `messages` so the dashboard can distinguish unscored from
  scored-as-neutral, and so end-to-end latency is `scored_at − created_at`.

### Dashboard τ view

Rollups are fixed at the configured τ, so the τ view reads message-level
probabilities with a new `fetchMessageProbs` in [`data.ts`](../dashboard/src/lib/data.ts),
mirroring `fetchRollups` and its `q` / `orGroup` helpers:

- Filter on guild, `score_status = "scored"`, `deleted_at = ""` and the indexed
  `created_at` (not `day`, which is not indexed on `messages`).
- Always pass `fields: 'day,channel_id,author_id,probabilities'`; without it every
  record also returns its message `content`.
- `created_at` is UTC but `day` is bucketed in `VIBESHIFT_TZ`, so widen the range by
  one day each side and trim to the wanted `day` range in the browser. Start at
  `shiftDay(from, -(ma - 1))`, the same lookback the chart uses.
- In the browser, for each τ ∈ {0} ∪ {0.15, 0.20, …, 0.85}: gate each message, sum
  one-hot labels per `day` into `Rollup`-shaped rows, and pass them to the existing
  `buildSeries`. `getFullList` pages at 1,000 rows, which is fine at test-server
  scale; a large server would need server-side aggregation.

## §4–5 Offline 300-Message Benchmark & `evaluation/` Architecture

A 300-message held-out benchmark (75 `dev` / 225 `test`) with preceding-3 and
preceding-25 context windows, dual-annotated and adjudicated to gold labels. The
[`evaluation/`](../evaluation/) tree is decoupled from production via two frozen
contracts (see [`evaluation/common/schema.md`](../evaluation/common/schema.md)):

1. **`data/messages.jsonl`** + **`data/annotations/consensus.jsonl`** — the dataset
   and gold labels (gold `label` for hard metrics; `soft_probs` preserving the
   pre-adjudication 50/50 split for soft/distribution checks).
2. **`predictions/{model}.jsonl`** — one file per model, un-thresholded 7-class
   probabilities + `raw_top_label` + latency + cost. Any model writing a valid
   file is scored automatically.

**Dataset rules.** Messages come from the test server (informed participants only),
excluding deleted ones, stratified by channel and length with messages of 3 tokens
or fewer oversampled. The set is frozen before any model runs on it and is excluded
from all Laya training data. Annotators see the preceding context (and the
`reply_to_id` parent if it falls outside the window). Report Cohen's κ on the label
and on each flag *before* adjudication, and report n next to every slice's CI.

**Models:** Laya (core), DistilRoBERTa-Emotion, RoBERTa-GoEmotions→7, zero-shot
LLM, NRC EmoLex (word-count + M=25 macro window), and a LogisticRegression
stacking head (Laya probs ⊕ NRC macro vector), kept small and regularized because
it is fit on only 75 dev messages.

**Metrics** ([`eval/metrics.py`](../evaluation/eval/metrics.py), all schema-ordered,
computed in numpy to avoid sklearn's alphabetical label sorting): Macro-F1 (overall,
and sliced by `is_context_dependent` and `is_sarcastic`, with 95% bootstrap CIs),
per-class precision/recall, confusion matrices, Top-2 accuracy, multi-class Brier
score, log-loss, and two Jensen–Shannon divergences — (a) mean predicted
distribution vs gold label mix, and (b) τ-gated top-label counts vs gold label mix.
Cohen's κ measures inter-annotator reliability; `tau_sweep.py` picks each model's τ\*
on **dev**, and **test** is reported only at τ = 0 and τ\*. The same distribution
metrics already run on the GoEmotions test split via
[`scripts/laya_test_report.py`](../scripts/laya_test_report.py). The whole pipeline
runs today on a committed 10-row sample fixture.

**Latency and cost.** Time local models on the same machine at batch size 1 and 8
(the bot's `SCORE_CHUNK`), reusing [`scripts/bench_parallel.py`](../scripts/bench_parallel.py).
LLM cost uses API token pricing on the run date; local models convert measured
throughput to $ per 1,000 messages at a stated cloud instance rate.

**System validation.** Compare PocketBase message counts with Discord
`channel.history()` for a fixed window; stop the bot, send N messages, restart, and
confirm the downtime job ingests all N; measure `scored_at − created_at` live.

**Charts:** Altair builders in [`eval/charts.py`](../evaluation/eval/charts.py) render
the report PNGs headlessly (vl-convert) and power a marimo explorer
(`pixi run eval-notebook`) with a live τ slider and click-to-drill-down confusion
matrix.

## §6 Work Breakdown & Component Ownership

| # | Owner | Scope | Directories | Concrete deliverables |
| - | ----- | ----- | ----------- | --------------------- |
| 1 | **Lucas Starkey** | Core platform + 7-label Laya model + Laya eval runner | `src/vibeshift/`, `dashboard/`, `notebooks/`, `evaluation/models/laya/`, `LaunchSpark/laya-emotion-tracker` | ~~Push the 7-label GoEmotions checkpoint~~ (done, `v2-7-labels`); ~~update `emotions.ts` / `seed.ts` (13→7)~~ (done); add `VIBESHIFT_TAU`, τ-gated top-label rollup counting, and `scored_at` to `scorer.py` / `store.py`; wire the dashboard τ view; `evaluation/models/laya/run.py` → `predictions/laya.jsonl`. |
| 2 | **Teammate A & Teammate B** | Dataset curation, dual annotation, statistical eval | `evaluation/curation/`, `evaluation/annotation/`, `evaluation/data/` | Export 300 messages (75 dev / 225 test) with 3- and 25-message context windows, covering ordinary chat, slang, short replies, sarcasm, and context-dependent messages (`curation/export_from_pb.py` → `data/messages.jsonl`); independently annotate all 300 (7 emotions + `is_context_dependent` + `is_sarcastic`); Cohen's κ, consensus adjudication with `soft_probs`, τ sensitivity sweep, and JSD validation (`annotation/reliability.py`, `annotation/tau_sweep.py`). |
| 3 | **Teammate C** | External transformer & zero-shot LLM baselines | `evaluation/models/transformers/`, `evaluation/models/llm/` | Consume `data/messages.jsonl`; run DistilRoBERTa-Emotion (`j-hartmann/emotion-english-distilroberta-base`), RoBERTa-GoEmotions (`SamLowe/roberta-base-go_emotions` → 7), and a zero-shot LLM; emit 7-class probability vectors + latency (ms/msg) + API/compute cost as `predictions/*.jsonl`. |
| 4 | **Teammate D** | NRC lexicon, context stacking, final plots | `evaluation/models/nrc_lexicon/`, `evaluation/models/stacking/`, `evaluation/eval/` | NRC EmoLex in both roles — 7-class word-count baseline with neutral fallback and the M=25 macro-window mood vector; train the LogisticRegression stacking head (Laya + NRC macro) on the 75 dev messages and score the 225 test messages; run `eval/evaluate.py` to generate the report tables (Macro-F1, Top-2, Brier) and confusion matrices. |

**Shared, not forked:** `evaluation/common/` (labels, I/O, frozen schemas). Change
the two contracts only with the whole team's agreement.

## §7 Production Code Follow-Up Checklist

Owner: Lucas. These close the gap between the built system and the 7-label design:

- [x] Fine-tune and push the 7-label checkpoint; pin `MODEL_REVISION = "v2-7-labels"` in `emotions.py`.
- [x] Switch `dashboard/src/lib/emotions.ts` (and `scripts/seed.ts`) from 13 to the 7 `LABELS`.
- [x] Update the label-count assertions in `dashboard/src/lib/emotions.test.ts` and `tests/test_emotions_model.py` to expect 7.
- [ ] Add `VIBESHIFT_TAU` config; in `score_batch` set `confidence = p(e*)` and `emotion` to the gated label.
- [ ] Count one-hot τ-gated labels in `daily_rollups` (when adding a message and when subtracting it on edits and deletes).
- [ ] Add `scored_at` to the `messages` collection (migration in `pb_migrations/`) and set it in `store.py`'s `save_score`.
- [ ] Add `fetchMessageProbs` and the τ view to the dashboard (§3).
- [ ] One-off script: reset messages scored by an older checkpoint to `pending`, and rebuild rollups from stored probabilities when τ changes.
- [ ] Wire the Label data page to real messages instead of the placeholder queue.
- [ ] Keep the drift guard green: `pixi run test-model` runs `evaluation/common/test_labels.py`.

## §8 Handoffs & timeline

- **Schema freeze first** — `common/schema.md` is frozen now, so C and D build
  against the sample fixtures while A & B curate and annotate.
- **A & B → C, D:** `data/messages.jsonl` + `data/annotations/consensus.jsonl`.
- **C, D, Lucas → D:** each model's `predictions/*.jsonl` flows into one
  `eval/evaluate.py` run.
- **Refine & report:** τ chosen from the sweep; final tables, confusion matrices,
  and the written report.
