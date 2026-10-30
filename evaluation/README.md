# VibeShift evaluation project

The research tree for CSC 4240/5240 — it curates a 300-message dataset, collects
dual human annotations, runs several emotion-classification baselines against the
7-label Laya model, trains a stacking head, and scores everything with one
evaluation script. It sits **alongside** the live production system
(`src/vibeshift`, `dashboard`) and never modifies it; the only shared point is the
7-label list in [`common/labels.py`](common/labels.py).

## The two contracts

Everything flows through two file schemas, frozen in [`common/schema.md`](common/schema.md):

1. **`data/messages.jsonl`** — the 300 curated messages (75 `dev` / 225 `test`),
   each with its preceding-3 and preceding-25 context windows.
2. **`predictions/{model}.jsonl`** — one file per model, un-thresholded 7-class
   probabilities + latency + cost. `io.write_predictions()` validates every row.

A model runner and the evaluation script are decoupled: any model that writes a
valid prediction file is picked up automatically by `eval-report`.

## Run it today (on the 10-row sample)

The real `messages.jsonl` / `consensus.jsonl` don't exist until A & B finish; the
loaders fall back to the committed **sample fixtures** with a warning, so the whole
pipeline runs now:

```bash
pixi run eval-nrc          # NRC word-count baseline  -> predictions/nrc_wordcount.jsonl
pixi run eval-nrc --role macro   # M=25 macro window  -> predictions/nrc_macro.jsonl
pixi run eval-transformers # DistilRoBERTa (stub)     -> predictions/transformers_distilroberta.jsonl
pixi run eval-llm          # zero-shot LLM (stub)     -> predictions/llm_zeroshot.jsonl
pixi run eval-stack        # stacking head            -> predictions/stacking.jsonl
pixi run eval-report       # metrics table + Altair PNGs in eval/report/
pixi run eval-tau          # τ sensitivity sweep (CSV + chart)
pixi run eval-notebook     # interactive explorer (below)
```

Stubs print a loud warning and emit a uniform distribution so the plumbing is
testable before the real models land.

## Interactive explorer

`pixi run eval-notebook` opens [`eval/explore_eval.py`](eval/explore_eval.py), a
marimo notebook over every file in `predictions/`. Drag the τ slider to re-gate
all models live, pick a model, and click a confusion-matrix cell to list the
messages behind it (content, preceding context, and the sarcasm and
context-dependence flags). With no cell selected, it lists every misclassified
message. No plotting code needed: drop in your `{model}.jsonl` and re-run.

The notebook and `eval-report` use the same Altair builders
([`eval/charts.py`](eval/charts.py)) on the same data
([`eval/frames.py`](eval/frames.py)), so the PNGs in the report match what you
see interactively. Each model has a fixed color slot in `charts.MODEL_ORDER`.
Add a new model there, or it shows in gray.

## Ownership

| Area | Directory | Owner |
| --- | --- | --- |
| Core platform + Laya runner | `src/vibeshift`, `dashboard`, `models/laya/` | Lucas |
| Curation (300-message export) | `curation/` | Teammates A & B |
| Dual annotation, κ, τ sweep, JSD | `annotation/`, `data/annotations/` | Teammates A & B |
| Transformer + zero-shot LLM baselines | `models/transformers/`, `models/llm/` | Teammate C |
| NRC lexicon + stacking + evaluation | `models/nrc_lexicon/`, `models/stacking/`, `eval/` | Teammate D |

Shared, do not fork: `common/` (labels, I/O, schema). Change the schemas only with
the team's agreement.

## τ gating

Both `eval/evaluate.py` and `annotation/tau_sweep.py` turn stored probabilities
into a final label with the one rule in `schema.md`: predict `neutral` if the raw
top label is neutral or the strongest non-neutral probability is below τ,
otherwise keep the raw top label. `τ = 0.0` is plain argmax.
