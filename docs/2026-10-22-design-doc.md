# VibeShift — Internal Design Doc (Job-A Scope)

**Course:** CSC 4240/5240 (Artificial Intelligence)
**Team:** Lucas Starkey, Trigg Lampkins, Trey Gannod, Braylon Trail, Isaiah Chastain
**Focus:** Multi-model emotion benchmarking, neutral calibration, community mood trends
**Grounded against:** `main` @ `431b763` (2026-10-22)

Throughout, **(built)** marks what already exists in the repo and **(new)** marks work still to do. Locked decisions are in §8.

---

## 1. System intent and scope

VibeShift helps community managers see how conversations *feel* over time, across channels and members. Off-the-shelf emotion classifiers struggle on Discord because of:

- **The neutral majority.** Most messages are logistics or filler ("gg", "hop on vc", "at 8"). An emotion model with no neutral class must assign an emotion to this chatter. Even a model *with* a neutral class can give confident-looking emotion scores to it. Either way, the dashboard's trend lines get distorted.
- **Short, ambiguous replies.** "sure", "oh great", "wow" often need the preceding messages to read correctly.

**Scope (Job A, locked):**

1. **Shared labels.** One 7-label schema everywhere: joy, sadness, anger, fear, disgust, surprise, neutral.
2. **Laya + τ.** Our Laya checkpoint is fine-tuned on the 7 labels with [notebooks/finetune_laya_emotions.py](../notebooks/finetune_laya_emotions.py) (§2) **(built)**. A tuned confidence cutoff τ (§3) routes low-confidence predictions to neutral **(new)**.
3. **Benchmark.** Compare Laya + τ\* against DistilRoBERTa-Emotion, RoBERTa-GoEmotions and a zero-shot LLM on accuracy, latency and cost. Add a context ablation (k = 3 preceding messages, optionally plus NRC lexicon features). VADER is a 3-class polarity reference **(new)**.
4. **Dashboard.** Channel and member mood trends over time using the existing 1/7/14/21/30-day simple moving average **(built)**, plus a client-side τ sensitivity view **(new)**.

**Stretch goals:**

- fine-tuning Laya, or the best-performing transformer, on labeled Discord messages
- exploratory member-influence analysis (Job B)

Laya's performance on Discord text is measured and reported only through the benchmark. We make no prior accuracy claims.

---

## 2. Label space and models

$$\mathcal{Y} = \{\text{joy, sadness, anger, fear, disgust, surprise, neutral}\}$$

This one label set is used for human annotation, every benchmark metric, `messages.probabilities`, `daily_rollups.prob_sums` and the dashboard. Each model reaches it differently:

- **Laya:** `LaunchSpark/laya-emotion-tracker` at tag `v2-7-labels`, our own Hugging Face repo. It is fine-tuned from `convaiinnovations/laya` on all **7 labels** with [notebooks/finetune_laya_emotions.py](../notebooks/finetune_laya_emotions.py):
  - **Data:** [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Reddit comments), mapped to the 7 labels with the grouping in the table below.
  - **Splits:** its official train/validation/test splits.
  - **Training set:** up to 1,000 comments per label. Fear and disgust have only 538 and 495, so all of them are used.
  - **Results:** test macro-F1 0.594, against 0.437 for the untuned base model. Full results are in [docs/results/2026-10-23-goemotions/](results/2026-10-23-goemotions/).
  - **The previous 13-label model** is tagged `v1-13-labels`.

  [emotions.py](../src/vibeshift/emotions.py) pins that revision and reads the label list from the checkpoint's `questions.json`. PocketBase stores probabilities as generic JSON dictionaries, so neither loading nor storage needed code changes for the new labels.
- **DistilRoBERTa-Emotion** (`j-hartmann/emotion-english-distilroberta-base`): already outputs exactly $\mathcal{Y}$.
- **RoBERTa-GoEmotions** (`SamLowe/roberta-base-go_emotions`): outputs 28 labels, which we map using Google's published GoEmotions → Ekman grouping (table below).
- **Zero-shot LLM:** prompted to return a JSON probability for each label in $\mathcal{Y}$. We renormalize those probabilities to sum to 1.

| $\mathcal{Y}$ | GoEmotions labels |
|---|---|
| joy | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| sadness | sadness, disappointment, embarrassment, grief, remorse |
| anger | anger, annoyance, disapproval |
| fear | fear, nervousness |
| disgust | disgust |
| surprise | surprise, realization, confusion, curiosity |
| neutral | neutral |

GoEmotions is multi-label, with independent sigmoid scores that don't sum to 1. We sum the scores within each group, then normalize the 7 group totals to sum to 1.

**Sarcasm and context-dependence are not output classes.** They exist only as boolean annotation flags on the offline dataset (§5.1), used to slice the results.

---

## 3. Neutral cutoff τ: one rule for every model

Let $p$ be a model's distribution over its labels. If the model has no neutral output, treat $p(\text{neutral}) = 0$. Let $e^* = \arg\max_{y \neq \text{neutral}} p(y)$ be the top non-neutral emotion. Then:

$$\hat{y} = \begin{cases} \text{neutral} & \text{if } \arg\max_{y} p(y) = \text{neutral} \ \text{ or } \ p(e^*) < \tau \\ e^* & \text{otherwise} \end{cases}$$

- **6-emotion head.** The argmax is never neutral, so the rule becomes: neutral iff $\max_i p_i < \tau$.
- **τ = 0** is the plain argmax: the ungated baseline we report next to τ\*.

**What the trends count.** Each message adds one count to its τ-gated top label:

$$g_\tau(m) = \text{one-hot}\big(\hat{y}_\tau(m)\big)$$

So the dashboard plots the **share of messages** per emotion. The full distribution $p$ stays on the message record for the distribution metrics (§5.2) and the τ view (§4.4).

**Why counts, not summed probabilities (measured).** On GoEmotions' official test split (4,922 comments), we compared each approach's overall emotion mix with the human label mix, using Jensen-Shannon divergence (base 2, so 0 is identical and 1 is disjoint), for the 7-label Laya checkpoint (`v2-7-labels`) at τ = 0:

| Aggregate | JSD vs. the human label mix |
|---|---|
| Mean of the full probability vectors | 0.031 |
| Share of top labels | **0.007** |

Laya is fine-tuned on balanced classes, so its probabilities carry a balanced prior. Averaging them inflates the rare emotions: fear and disgust each average 0.054–0.057 against a true share of 0.015–0.016, about 3.5 times too high. Joy and neutral come out too low (0.307 vs. 0.377 and 0.229 vs. 0.323). Taking the top label first cancels most of that bias with no extra parameters. A prior correction (e.g. Saerens et al., 2002) was the alternative; it was not adopted, because it needs an estimate of each server's label mix and adds a step at runtime. The full numbers are in [docs/results/2026-10-23-goemotions/results.json](results/2026-10-23-goemotions/results.json).

---

## 4. Pipeline, storage and dashboard

### 4.1 Pipeline (built, with small changes)

```text
Discord → bot (discord.py) → PocketBase `messages` (score_status = pending)
                                   ↓
          Scorer: newest-first chunks of SCORE_CHUNK = 8, one decide_batch call per chunk
                                   ↓
          messages.{emotion, probabilities, confidence, model_id}
          + daily_rollups (guild, channel, author, day): count, prob_sums
```

**What changes (new):**

| Field | Contents |
|---|---|
| `messages.probabilities` | Laya's **raw** distribution (7 or 6 labels). This is the source of truth, so any τ can be applied later. |
| `messages.confidence` | $p(e^*)$, the score that τ is compared against |
| `messages.emotion` | $\hat{y}$ at the configured τ (new setting `VIBESHIFT_TAU`, set to τ\* once tuned) |
| `daily_rollups.prob_sums` | Counts of each τ-gated top label at the configured τ: `prob_sums[y]` is the number of the day's messages labeled $y$. The field keeps its name, so the schema doesn't change. |
| `messages.scored_at` | New field, added by migration and set in `Store.save_score`. End-to-end latency is `scored_at − created_at`. |

**Changing the configured τ requires a rollup rebuild.** Rollups are summed at one τ, so a new value means recomputing them from `messages.probabilities` with a one-off script.

**Unchanged:**

- **Store.** PocketBase (SQLite underneath). The schema lives in [pb_migrations/1760700000_init.js](../pb_migrations/1760700000_init.js), changes only through new migrations, and has no hooks.
- **Capture.** Non-bot, non-empty messages in guild text channels. Threads and voice-channel chat are **not** captured.
- **Deduplication.** `insert_pending` looks up `message_id` before inserting, and a unique index catches races.
- **Edits and deletes.** An edit puts the message back in the queue for rescoring. A delete subtracts the message from its rollup.
- **Offline recovery.** `downtime` jobs walk `channel.history(after=newest_fetched_id)`. They are queued:
  - at startup
  - after a reconnect
  - on guild join or guild-available events
  - when capturing a live message fails

  `extend` jobs walk backwards from `oldest_fetched_id` ([history.py](../src/vibeshift/history.py)).
- **Stored context.** `reply_to_id` is stored, and the `(guild_id, channel_id, created_at)` index makes "k preceding messages in a channel" a cheap query.
- **Access.** Every collection's API rules are open (`""`). Only browser requests from `localhost:5173` / `127.0.0.1:5173` are allowed in. The data stays on the local machine, and there is **no authentication**.

### 4.2 Benchmark storage (new)

Benchmark predictions stay **out of PocketBase**. Evaluation runs offline on a frozen set of 300 messages, and the dashboard shows Laya only.

```text
eval/data/sample.csv           message_id, channel_id, author_id, created_at, content, ctx_1..ctx_3   (gitignored)
eval/data/annotations_{a,b}.csv   label, is_context_dependent, is_sarcastic
eval/data/gold.csv
eval/preds/<model>.parquet     message_id, model, p_joy..p_neutral, label_raw, latency_ms
```

`<model>` is one of `laya`, `laya_context`, `distilroberta`, `goemotions`, `llm_zeroshot` or `vader`. Files under `eval/data/` contain message text and must be gitignored.

### 4.3 Dashboard trends (built; now counts labels)

The **Emotion Prevalence History** chart is computed in the frontend by `buildSeries` in [series.ts](../dashboard/src/lib/series.ts). For each day $d$, it plots the **share of messages** with each label over a trailing window of $w \in \{1, 7, 14, 21, 30\}$ days:

$$\bar{P}(d, y) = \frac{\sum_{d'=d-w+1}^{d} S(d', y)}{\sum_{d'=d-w+1}^{d} N(d')}$$

- $S(d', y)$ is the number of messages on day $d'$ whose τ-gated top label is $y$ (`daily_rollups.prob_sums`, §3).
- `series.ts` needs no change: it already divides the summed values by the message count.
- $N(d')$ is the number of messages that day (`daily_rollups.count`).
- **Lookback.** [emotions.tsx](../dashboard/src/pages/emotions.tsx) fetches rollups from `shiftDay(from, -(ma - 1))`, so the first visible day already has a full window.

**Other built features:**

- **Filters:** guild, time range (7/30/60/90 days, all time, or custom), people (`author_id`), channels (`channel_id`), emotions. All filters are encoded in the URL ([filters.ts](../dashboard/src/lib/filters.ts)).
- **Stat cards:** current and peak value for each shown emotion.
- **Backfill page:** extend how far back the stored history goes.

### 4.4 τ view (new)

Rollups are fixed at the configured τ, so the τ view works from message-level probabilities instead. Today the dashboard reads `messages` only for the pending count, so this adds one new query in [data.ts](../dashboard/src/lib/data.ts). It mirrors `fetchRollups`, reusing that file's `q` and `orGroup` helpers:

```ts
export async function fetchMessageProbs(p: RollupParams & { fromIso: string; toIso: string }) {
  const parts = [
    `guild_id = ${q(p.guild)}`,
    `score_status = "scored"`,
    `deleted_at = ""`,
    `created_at >= ${q(p.fromIso)}`,
    `created_at <= ${q(p.toIso)}`,
  ]
  if (p.channels.length) parts.push(orGroup('channel_id', p.channels))
  if (p.people.length) parts.push(orGroup('author_id', p.people))
  return pb.collection('messages').getFullList({
    filter: parts.join(' && '),
    fields: 'day,channel_id,author_id,probabilities',
    batch: 1000,
  })
}
```

- **`fields` is required.** Without it, every record also returns its message `content`.
- **Filter on `created_at`, not `day`.** `created_at` is covered by the `(guild_id, channel_id, created_at)` index; `day` is not indexed on `messages`.
- **Time zones.** `created_at` is UTC, but `day` is bucketed in `VIBESHIFT_TZ`. So `fromIso` and `toIso` are widened by one day on each side, and rows are then trimmed to the wanted `day` range in the browser.
- **Lookback.** The range starts at `shiftDay(from, -(ma - 1))`, the same lookback the chart uses.
- **Pagination.** `getFullList` pages on its own, making sequential requests of 1,000 rows, so about 10 requests per 10,000 messages. That's fine at test-server scale; a large server would need server-side aggregation.

**From messages to the chart.** In the browser, for each slider value τ ∈ {0} ∪ {0.15, 0.20, …, 0.85}:

1. Compute $\hat{y}_\tau(m)$ for each message (§3) and count it as a one-hot vector.
2. Sum the vectors per `day` into `Rollup`-shaped rows (`{day, count, prob_sums}`).
3. Pass those rows to the existing `buildSeries`, so the τ view uses the same SMA with no new smoothing code.

---

## 5. Evaluation protocol

### 5.1 Held-out set and dual annotation

- **Source.** $N = 300$ messages from the test server's `messages` collection (informed participants only), excluding deleted messages. The set is frozen before any model is run on it.
- **No leakage.** These 300 messages are excluded from any Laya training data, now and in the fine-tuning stretch goal.
- **Split.** $N_{dev} = 75$ for tuning τ\* and the context stacker. $N_{test} = 225$, used only for final numbers.
- **Stratification.** By channel and by length, oversampling messages of 3 tokens or fewer so short replies are well represented.
- **Context shown.** The k = 3 preceding messages in the same channel, ordered by `created_at`. If the message replies to something outside that window, its `reply_to_id` parent is shown too.
- **Labels.** Two annotators independently assign one label from $\mathcal{Y}$, plus two boolean flags: `is_context_dependent` and `is_sarcastic`. We report Cohen's κ on the label and on each flag *before* resolving disagreements by discussion into the gold labels.
- **Small slices.** The context-dependent and sarcastic slices of the test set may have only a few dozen messages each. Report n for every slice next to its CI.

### 5.2 Comparison matrix

Metrics on the test set:

- overall Macro-F1
- Macro-F1 on the context-dependent slice and on the sarcastic slice, each with a 95% bootstrap CI
- per-class precision and recall
- confusion matrix
- latency (ms per message)
- cost ($ per 1,000 messages)

**Distribution metrics.** Macro-F1 scores only each message's top label after τ. The dashboard sums whole probability vectors, so the full 7-label distribution $p$ is scored too, at τ = 0:

| Metric | What it scores |
|---|---|
| Log-loss, $-\frac{1}{N}\sum_t \log p_t(y_t)$ | Probability on the gold label, per message |
| Brier score, $\frac{1}{N}\sum_t \sum_c (p_{t,c} - y_{t,c})^2$ | Squared error of the whole distribution, per message |
| Top-2 accuracy | Gold label among the two most likely: credit for blended emotions |
| JSD$(\bar{p} \,\Vert\, \bar{y})$ | Average predicted distribution vs. the gold label mix: what the dashboard trend reflects |

**Annotator disagreement as a soft label.** Where the two annotators disagree before resolution, their labels form a two-point distribution (e.g. anger 0.5, disgust 0.5). Report JSD between $p$ and that distribution on the disagreement slice, beside the hard-label metrics against the resolved gold label.

These metrics already run in [notebooks/finetune_laya_emotions.py](../notebooks/finetune_laya_emotions.py) (Step 4), on the GoEmotions test split.

| # | Model | Input | Notes |
|---|---|---|---|
| 1 | Laya + τ\* | $m_t$ | Our retrained checkpoint, run in-process via `score_batch`. |
| 2 | DistilRoBERTa-Emotion | $m_t$ | Native 7 labels. |
| 3 | RoBERTa-GoEmotions | $m_t$ | Mapped to 7 labels (§2). |
| 4 | Zero-shot LLM | $m_t$ | e.g. Gemini 2.5 Flash or Llama-3.1-8B. Returns JSON probabilities over $\mathcal{Y}$. |
| 5 | Context ablation | $m_{t-3..t}$ (+ NRC) | (a) Context concatenated into the input for Laya and the LLM. (b) A small stacker fit on dev (§5.2.1). |
| 6 | VADER | $m_t$ | 3-class polarity reference, reported separately. joy → positive; sadness, anger, fear, disgust → negative; neutral and surprise → neutral. |

**Latency.**

- Time all local models on the same machine, at batch size 1 and at batch size 8 (the bot's `SCORE_CHUNK`). Reuse [scripts/bench_parallel.py](../scripts/bench_parallel.py).
- On the owner's Intel Mac CPU, the 13-label checkpoint ran at about 5 s per message. Re-measure with the retrained checkpoint.

**Cost.**

- **LLM:** API token pricing on the date of the run.
- **Local models:** measured throughput converted to $ per 1,000 messages using a stated cloud instance hourly rate.

#### 5.2.1 Risks in the context ablation

- **Input format.** Laya scores single messages. Treat (a) as "does context help anyway?", not as a fair test of context.
- **Stacker overfitting.** Fitting a stacker on 75 dev messages will overfit easily. Keep it tiny: L2-regularized logistic regression on $[p(m_t),\ \bar{p}(\text{ctx}),\ \text{NRC counts}]$, about 25 features, chosen by cross-validation inside dev.

### 5.3 τ sensitivity

- **Calibration curve.** Sweep τ ∈ {0} ∪ {0.15, 0.20, …, 0.85} for each model. Plot Macro-F1 and Neutral-F1 on **dev**, and pick each model's τ\* there. Report **test** only at τ = 0 and at τ\*.
  - With a 6-emotion head, τ = 0 gives Neutral-F1 = 0, which shows directly why an ungated 6-class model fails on Discord.
- **Trend comparison.** Compare predicted-label distributions to the gold distribution with Jensen-Shannon divergence, at a low τ, at τ\* and at a high τ.
  - 225 test messages spread over many days are too sparse to compare day by day, so compare **overall and per channel**.
  - Separately, use the dashboard τ view to show how τ reshapes trends on the full unlabeled corpus.

### 5.4 System validation

- **Capture completeness.** For a fixed window, compare PocketBase message counts against counts from Discord `channel.history()`.
- **Offline recovery.**
  1. Stop the bot.
  2. Send N messages.
  3. Restart the bot.
  4. Confirm the downtime job ingests all N.
- **End-to-end latency.** `scored_at − created_at` for live messages (§4.1).

---

## 6. Work breakdown and status

| Role | Built | Remaining |
|---|---|---|
| **1. Bot & ingestion** | Bot, live capture, downtime and extend recovery, dedupe, edit and delete handling, `!vibe`, PocketBase schema | `scored_at` migration; rescore and rollup-rebuild script (§7); eval export script (sample plus k = 3 context); capture-completeness and recovery measurements |
| **2. Laya & context** | Laya loading and batch scoring (`emotions.py`, `score_batch`); 7-label fine-tuning notebook and GoEmotions results (`v2-7-labels`) | τ gate and top-label rollups (§3); context ablation (concatenation plus stacker); NRC features |
| **3. Baselines** | — | DistilRoBERTa runner; GoEmotions mapper; LLM runner; latency and cost profiler |
| **4. Annotation & eval** | Distribution metrics and the test-report script ([scripts/laya_test_report.py](../scripts/laya_test_report.py)) | Sampling and annotation sheets; κ; Macro-F1 with CIs; confusion matrices; τ sweep; JSD |
| **5. Dashboard** | SMA trend chart, guild/channel/people/emotion/range filters, stat cards, backfill page; 7 labels | τ view (§4.4) |

---

## 7. Code changes

**Done** (branch `feat/7-label-schema`):

| File | Change |
|---|---|
| [src/vibeshift/emotions.py](../src/vibeshift/emotions.py) | Pins `MODEL_REVISION = "v2-7-labels"`. `MODEL_ID` records the revision, so each message's `model_id` says which checkpoint scored it. |
| [dashboard/src/lib/emotions.ts](../dashboard/src/lib/emotions.ts) | `LABELS` and `EMOTION_COLORS` set to the 7 labels. |
| Dashboard tests, [scripts/seed.ts](../dashboard/scripts/seed.ts) | Updated for the 7 labels. |
| [tests/test_emotions_model.py](../tests/test_emotions_model.py) | Asserts 7 labels. |
| README | Describes the 7 labels. |

**Still to do:**

| File | Change |
|---|---|
| [src/vibeshift/emotions.py](../src/vibeshift/emotions.py) | In `score_batch`, set `confidence` to $p(e^*)$ and `emotion` to $\hat{y}$ at `VIBESHIFT_TAU`. |
| [src/vibeshift/scorer.py](../src/vibeshift/scorer.py) | Add the one-hot top label $g_\tau$ to rollups instead of the raw probabilities, both when adding a message and when subtracting it on edits and deletes. |
| [src/vibeshift/config.py](../src/vibeshift/config.py) | Add the `VIBESHIFT_TAU` setting. |
| [src/vibeshift/store.py](../src/vibeshift/store.py) + new migration | Add `scored_at`, set in `save_score`. |
| [dashboard/src/lib/data.ts](../dashboard/src/lib/data.ts) | Add `fetchMessageProbs` (§4.4). |
| Dashboard τ view | A slider, $\hat{y}_\tau$ per message, and per-day counts fed into the existing `buildSeries`. |
| Existing data | A one-off script resets messages scored by an older checkpoint to `pending` and clears `daily_rollups`. The same script rebuilds rollups from stored probabilities when the configured τ changes. |

`series.ts`, `filters.ts`, the migration's existing fields, `rollups.py` and the scorer's ordering, batching and locking stay as they are.

---

## 8. Locked decisions

1. **Labels:** 7 labels everywhere. Laya is `LaunchSpark/laya-emotion-tracker` at tag `v2-7-labels`, fine-tuned on GoEmotions with a 7-label head.
2. **τ:** one rule for every model (§3). The dashboard defaults to τ\*, and the τ view applies other values in the browser to per-message probabilities via `fetchMessageProbs`.
3. **Rollups count τ-gated top labels.** `messages.probabilities` keeps the full 7-label distribution. `daily_rollups` counts one-hot top labels, so the chart shows the share of messages per emotion. This was measured: JSD 0.007 against 0.031 for summed probabilities (§3).
4. **Sarcasm and context-dependence:** human-annotation flags only, used to slice Macro-F1.
5. **Benchmark storage:** offline files, not PocketBase.
6. **Smoothing:** the existing 1/7/14/21/30-day SMA in `series.ts`, with its `ma − 1` lookback, unchanged.
7. **Scope:** Job A. Member-influence analysis (Job B) is a one-sentence stretch goal.
8. **Accuracy claims:** none in advance.
9. **Latency:** `scored_at` for end-to-end latency; per-model inference latency in the offline benchmark.
