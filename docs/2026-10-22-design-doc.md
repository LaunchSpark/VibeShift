# VibeShift — Design Doc

**CSC 4240/5240: Artificial Intelligence**
**Team:** Lucas Starkey, Trigg Lampkins, Trey Gannod, Braylon Trail, Isaiah Chastain
**Created:** 2026-10-22

> Background design detail lives in
> [`docs/superpowers/specs/2026-10-17-emotion-dashboard-design.md`](superpowers/specs/2026-10-17-emotion-dashboard-design.md)
> and the plans beside it. This doc is the single source of truth for the 7-label
> space, the τ rule, and the work breakdown.

## §1 System Overview — `[BUILT]` vs `[TODO]`

| Component | Status | Code |
| --- | --- | --- |
| Discord capture + offline backfill | `[BUILT]` | [`bot.py`](../src/vibeshift/bot.py), [`history.py`](../src/vibeshift/history.py) |
| Scorer worker (drains pending deque, maintains rollups) | `[BUILT]` | [`scorer.py`](../src/vibeshift/scorer.py) |
| PocketBase storage + JSON daily rollups | `[BUILT]` | [`store.py`](../src/vibeshift/store.py), [`rollups.py`](../src/vibeshift/rollups.py) |
| React dashboard with SMA trend view | `[BUILT]` | [`emotions.tsx`](../dashboard/src/pages/emotions.tsx), [`series.ts`](../dashboard/src/lib/series.ts) |
| 7-label Laya checkpoint on Hugging Face | `[TODO]` | `LaunchSpark/laya-emotion-tracker` (`questions.json`) |
| `VIBESHIFT_TAU` + τ-gated rollup counting + `scored_at` | `[TODO]` | `scorer.py`, `store.py` |
| Dashboard 13→7 relabel + τ view | `[TODO]` | [`emotions.ts`](../dashboard/src/lib/emotions.ts), [`seed.ts`](../dashboard/scripts/seed.ts) |
| 300-message offline benchmark | `[TODO]` | [`evaluation/`](../evaluation/) |

## §2 Unified 7-Label Space & Universal τ Rule

The canonical label set, frozen in [`evaluation/common/labels.py`](../evaluation/common/labels.py):

```python
LABELS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")
```

Both production rollups and the offline benchmark gate a message to a final label
with **one rule**: a message is `neutral` if its top class is `neutral` **or**
`max_{e != "neutral"} p_e < τ`; otherwise it keeps its top class. `τ = 0.0` is
plain argmax. Because raw probabilities are stored un-thresholded, τ is a pure
post-processing choice (see [`evaluation/eval/metrics.py`](../evaluation/eval/metrics.py)
`gated_label`; the scorer will apply the same rule once `VIBESHIFT_TAU` lands — §7).

## §3 PocketBase & Rollup Storage

- Raw 7-class probabilities are stored on `messages.probabilities` (JSON), with
  the argmax in `messages.emotion` and model id in `messages.model_id`.
- `daily_rollups` accumulates **τ-gated top-label counts**, not raw probability
  sums — so balanced-training priors don't inflate rare classes like `fear` and
  `disgust` in the aggregate trend.
- Add `scored_at` to `messages` so the dashboard can distinguish unscored from
  scored-as-neutral and the scorer can report freshness.
- The client-side τ query reads over the indexed `created_at` with
  `fields: 'day,channel_id,author_id,probabilities'`, applying τ in the browser so
  the threshold can change without re-scoring.

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

**Models:** Laya (core), DistilRoBERTa-Emotion, RoBERTa-GoEmotions→7, zero-shot
LLM, NRC EmoLex (word-count + M=25 macro window), and a LogisticRegression
stacking head (Laya probs ⊕ NRC macro vector).

**Metrics** ([`eval/metrics.py`](../evaluation/eval/metrics.py), all schema-ordered,
computed in numpy to avoid sklearn's alphabetical label sorting): Macro-F1 (overall,
and sliced by `is_context_dependent` and `is_sarcastic`), Top-2 accuracy, multi-class
Brier score, log-loss, and two Jensen–Shannon divergences — (a) mean predicted
distribution vs gold label mix, and (b) τ-gated top-label counts vs gold label mix.
Cohen's κ measures inter-annotator reliability; `tau_sweep.py` tunes τ. The whole
pipeline runs today on a committed 10-row sample fixture.

**Charts:** Altair builders in [`eval/charts.py`](../evaluation/eval/charts.py) render
the report PNGs headlessly (vl-convert) and power a marimo explorer
(`pixi run eval-notebook`) with a live τ slider and click-to-drill-down confusion
matrix.

## §6 Work Breakdown & Component Ownership

| # | Owner | Scope | Directories | Concrete deliverables |
| - | ----- | ----- | ----------- | --------------------- |
| 1 | **Lucas Starkey** | Core platform + 7-label Laya model + Laya eval runner | `src/vibeshift/`, `dashboard/`, `evaluation/models/laya/`, `LaunchSpark/laya-emotion-tracker` | Push the 7-label GoEmotions checkpoint (`questions.json`) to Hugging Face; add `VIBESHIFT_TAU`, τ-gated top-label rollup counting, and `scored_at` to `scorer.py` / `store.py`; update `emotions.ts` / `seed.ts` (13→7) and wire the dashboard τ view; `evaluation/models/laya/run.py` → `predictions/laya.jsonl`. |
| 2 | **Teammate A & Teammate B** | Dataset curation, dual annotation, statistical eval | `evaluation/curation/`, `evaluation/annotation/`, `evaluation/data/` | Export 300 messages (75 dev / 225 test) with 3- and 25-message context windows, covering ordinary chat, slang, short replies, sarcasm, and context-dependent messages (`curation/export_from_pb.py` → `data/messages.jsonl`); independently annotate all 300 (7 emotions + `is_context_dependent` + `is_sarcastic`); Cohen's κ, consensus adjudication with `soft_probs`, τ sensitivity sweep, and JSD validation (`annotation/reliability.py`, `annotation/tau_sweep.py`). |
| 3 | **Teammate C** | External transformer & zero-shot LLM baselines | `evaluation/models/transformers/`, `evaluation/models/llm/` | Consume `data/messages.jsonl`; run DistilRoBERTa-Emotion (`j-hartmann/emotion-english-distilroberta-base`), RoBERTa-GoEmotions (`SamLowe/roberta-base-go_emotions` → 7), and a zero-shot LLM; emit 7-class probability vectors + latency (ms/msg) + API/compute cost as `predictions/*.jsonl`. |
| 4 | **Teammate D** | NRC lexicon, context stacking, final plots | `evaluation/models/nrc_lexicon/`, `evaluation/models/stacking/`, `evaluation/eval/` | NRC EmoLex in both roles — 7-class word-count baseline with neutral fallback and the M=25 macro-window mood vector; train the LogisticRegression stacking head (Laya + NRC macro) on the 75 dev messages and score the 225 test messages; run `eval/evaluate.py` to generate the report tables (Macro-F1, Top-2, Brier) and confusion matrices. |

**Shared, not forked:** `evaluation/common/` (labels, I/O, frozen schemas). Change
the two contracts only with the whole team's agreement.

## §7 Production Code Follow-Up Checklist

Owner: Lucas. These close the gap between the `[BUILT]` 13-label system and the
7-label design:

- [ ] Switch `dashboard/src/lib/emotions.ts` (and `scripts/seed.ts`) from 13 to the 7 `LABELS`.
- [ ] Add `VIBESHIFT_TAU` config and apply the universal τ rule when counting `daily_rollups` top labels.
- [ ] Add `scored_at` to the `messages` collection (migration in `pb_migrations/`) and set it in `scorer.py` / `store.py`.
- [ ] Update the label-count assertions in `dashboard/src/lib/emotions.test.ts` (currently `toBe(13)` for `LABELS` and `EMOTION_COLORS`) to expect 7.
- [ ] Enable the drift guard: once the 7-label checkpoint is pushed, `pixi run test-model` must pass `evaluation/common/test_labels.py`.

## §8 Handoffs & timeline

- **Schema freeze first** — `common/schema.md` is frozen now, so C and D build
  against the sample fixtures while A & B curate and annotate.
- **A & B → C, D:** `data/messages.jsonl` + `data/annotations/consensus.jsonl`.
- **C, D, Lucas → D:** each model's `predictions/*.jsonl` flows into one
  `eval/evaluate.py` run.
- **Refine & report:** τ chosen from the sweep; final tables, confusion matrices,
  and the written report.
