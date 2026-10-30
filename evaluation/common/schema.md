# Frozen schemas — the evaluation project's two contracts

Everything in `evaluation/` flows through two file formats. They are **frozen**:
change them only with the whole team's agreement, because four people build
against them in parallel. The canonical label order lives in
[`labels.py`](labels.py):

```python
LABELS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")
```

---

## Contract 1A — `evaluation/data/messages.jsonl` (owned by curation + annotation)

One JSON object per line. The 300 curated messages (75 `dev` / 225 `test`).

```json
{
  "id": "msg_001",
  "channel_id": "general",
  "author_id": "u_12",
  "created_at": "2026-10-20T18:30:00Z",
  "content": "oh great, another server crash",
  "preceding_3": ["msg t-3", "msg t-2", "msg t-1"],
  "preceding_25": ["msg t-25", "...", "msg t-1"],
  "split": "dev"
}
```

| field          | type        | notes                                                        |
| -------------- | ----------- | ------------------------------------------------------------ |
| `id`           | string      | stable message id, unique across the file                    |
| `channel_id`   | string      | source channel                                               |
| `author_id`    | string      | pseudonymous author id                                       |
| `created_at`   | string      | ISO-8601 UTC                                                 |
| `content`      | string      | the message text to classify                                 |
| `preceding_3`  | string[]    | the 3 messages immediately before (oldest → newest)          |
| `preceding_25` | string[]    | the 25 messages immediately before — the macro context window |
| `split`        | `"dev"`/`"test"` | 75 dev (tuning) / 225 test                              |

`splits.json` mirrors the split assignment as `{"dev": [...ids], "test": [...ids]}`.
It is a convenience; the `split` field in each row is the source of truth. If
`splits.json` is absent, derive splits from the rows (`io.split_ids`).

## Contract 1B — annotator & consensus files (owned by annotation)

`annotator_a.jsonl` and `annotator_b.jsonl` — one independent label per message:

```json
{"id": "msg_001", "label": "anger", "is_context_dependent": true, "is_sarcastic": true}
```

`consensus.jsonl` — the adjudicated gold, plus the pre-adjudication soft distribution:

```json
{
  "id": "msg_001",
  "label": "anger",
  "soft_probs": {"joy": 0.0, "sadness": 0.0, "anger": 0.5, "fear": 0.0, "disgust": 0.5, "surprise": 0.0, "neutral": 0.0},
  "is_context_dependent": true,
  "is_sarcastic": true
}
```

- `label` — the adjudicated **gold** label (used for Macro-F1, Top-2, confusion matrix).
- `soft_probs` — the normalized distribution over the two annotators' labels
  **before** adjudication (the 50/50 split when A and B disagreed). Used for the
  soft-distribution checks (JSD). Keys must be exactly `LABELS`.
- `is_context_dependent`, `is_sarcastic` — slice flags for the sliced Macro-F1.

## Contract 2 — `evaluation/predictions/{model_name}.jsonl` (every model writes this)

One row per message, per model. Store **un-thresholded** probabilities so
`evaluate.py` and `tau_sweep.py` can apply any τ cutoff after the fact.

```json
{
  "id": "msg_001",
  "model": "laya",
  "probs": {"joy": 0.02, "sadness": 0.08, "anger": 0.65, "fear": 0.03, "disgust": 0.12, "surprise": 0.04, "neutral": 0.06},
  "raw_top_label": "anger",
  "latency_ms": 6.4,
  "cost_usd": 0.0
}
```

| field           | type   | notes                                                       |
| --------------- | ------ | ----------------------------------------------------------- |
| `id`            | string | matches a `messages.jsonl` id                               |
| `model`         | string | the model name (also the file stem)                         |
| `probs`         | object | keys **exactly** `LABELS`, values sum to 1.0 (±1e-3)        |
| `raw_top_label` | string | the model's argmax label, **before** any τ gating           |
| `latency_ms`    | number | per-message inference latency                               |
| `cost_usd`      | number | per-message API/compute cost (0.0 for local models)         |

`io.write_predictions()` validates every row against this contract before writing.

### The universal τ rule

Both `evaluate.py` and `tau_sweep.py` turn `probs` into a final label the same way:

> Predict `"neutral"` if `raw_top_label == "neutral"` **or**
> `max_{e != "neutral"} probs[e] < τ`. Otherwise predict `raw_top_label`.

`τ = 0.0` reduces to plain argmax. Because `probs` are stored un-thresholded, the
τ sweep is a pure post-processing pass over the prediction files — no model re-runs.
