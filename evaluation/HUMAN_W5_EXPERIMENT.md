# Human labeled W5 experiment

This experiment evaluates the requested `j-hartmann/emotion-english-distilroberta-base` classifier against the supplied W=5 human labels. Each model input contains the target plus up to four preceding messages, with the target marked, matching the human window. This is a pretrained seven-category classifier; no supervised retraining is needed. This evaluation does not change any separate shared benchmark issue.

## Inputs and audit

- Human labels: `human_labeled_dataset/discord_messages_braylon-trail_w5.jsonl` (local input; one annotator).
- Messages: `evaluation/data/messages.jsonl` (local generated input; 7,300 rows).
- Alignment: exact message `id`; every human-labeled ID must exist exactly once in the message file. Invalid records, invalid labels, wrong window sizes, duplicate IDs, and unknown message IDs cause a fail-fast error. Messages without a human label are excluded and counted.
- The human file contains 1,127 unique labels, all W=5. Its distribution is neutral 877, joy 74, surprise 60, sadness 46, anger 36, disgust 25, fear 9. Neutral is 77.8%.
- All labeled messages belong to 51 conversations. W=5 means the target and up to four preceding messages in the same conversation; at conversation starts, fewer than five messages are available. No context crosses a conversation boundary.
- The labels are from one annotator, so inter-annotator agreement cannot be calculated.

## Split and input behavior

The experiment creates a deterministic approximately 80/20 split with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4240)`. The first fold is held out for final testing. Whole conversations stay in one partition. The split is constructed before tau selection; no model scores are used to choose it. On the supplied data this yields 913 train/calibration rows from 41 conversations and 214 test rows from 10 conversations.

The comparison serializes the last four available messages from `preceding_25`, followed by the target marked as `TARGET MESSAGE`. At conversation starts fewer than four preceding messages are supplied. Transformer tokenization is capped at 512 tokens, so older context can be truncated when a five-message window is long.

The selected human dataset intentionally differs from the current GitHub tracking issue #9, which specifies consensus labels at W=1 and W=4. This W=5 experiment is separate and does not overwrite those shared contracts.

## Tau calibration

The sweep includes tau from 0.00 through 1.00 by 0.01 and all observed non-neutral score breakpoints (including the value immediately above each breakpoint). At each tau it records accuracy, balanced accuracy, Macro-F1, weighted F1, predicted neutral percentage, human neutral percentage, absolute neutral-rate difference, and predicted class percentages.

Selection uses training/calibration data only: minimize the absolute difference between predicted and human neutral percentages; ties prefer higher balanced accuracy, then higher Macro-F1, then the lower tau. Matching the neutral ratio alone is not treated as evidence of good classification. Test results are calculated after the selected tau is frozen, at tau=0 and the selected tau.

## Outputs

Run from the repository root:

```powershell
pixi run eval-human-w5 --mode audit
pixi run eval-human-w5 --mode predict
pixi run eval-human-w5 --mode evaluate
pixi run eval-human-w5 --mode all
pixi run eval-w5-compare --models distilroberta,laya,nrc_emolex,local_llm
```

Prediction output has exactly one probability record per labeled message ID, plus the resolved model commit, latency, and cost. Model predictions are not thresholded in the saved file. A model revision can be pinned with `--revision <commit>`; the first run records the resolved commit in `model_run.json`.

Outputs go to `evaluation/results/human_labeled_w5/`:

- `dataset_audit.json` and `split.jsonl`: input hashes, validation, label counts, split and caveats.
- `predictions_distilroberta_w5.jsonl`: unthresholded model probabilities aligned to the labels.
- `tau_calibration_train.csv`, `tau_vs_accuracy.png`, `tau_vs_neutral_ratio.png`: calibration results only.
- `evaluation_results.json`, `evaluation_report.md`: train calibration and held-out metrics.
- `confusion_test_raw.csv`, `confusion_test_normalized.csv`, `confusion_test.png`: held-out confusion matrices.
- `classification_test_selected.csv`: precision, recall, F1 and support by class.
- `test_errors.jsonl`, `representative_examples.jsonl`: IDs and label outcomes; pass `--include-text-examples` only when you want message excerpts included in these local files.

The report includes accuracy, balanced accuracy, per-class precision/recall/F1, Macro-F1, weighted F1, Top-2 accuracy, Brier score, log loss, Cohen's kappa (human versus model), confusion matrices, majority-class baseline, neutral rates, and 95% conversation-cluster bootstrap intervals for key hard-label metrics. The bootstrap resamples conversations, not individual messages, to account for within-conversation dependence.

## W=5 model comparison

Run all locally available models with:

```powershell
pixi run eval-w5-compare --models distilroberta,laya,nrc_emolex
```

The comparison uses the same human labels, conversation-grouped split, seven-label schema, and held-out conversations for each model. Tau is calibrated separately per model on the training/calibration partition using neutral-rate alignment; the held-out test is used only for final scoring. Results include accuracy, balanced accuracy, per-class precision/recall/F1, Macro-F1, weighted F1, Top-2 accuracy, Brier score, log loss, Cohen's kappa, neutral rates, latency, cost where applicable, confusion matrices, and conversation-cluster bootstrap intervals.

Weighted F1 is calculated as `sum(class_support / total_support * class_F1)` over the seven labels. It weights common labels more heavily; since neutral makes up about 78% of this test set, weighted F1 is heavily influenced by neutral. Macro-F1 treats all seven labels equally, and balanced accuracy averages class recall. Read them together: weighted F1 reflects the dataset's observed class mix, while Macro-F1 and balanced accuracy show how performance spreads across rare and common emotions.

The zero-shot local LLM uses `Qwen/Qwen2.5-0.5B-Instruct` (small instruction-tuned language model; no API key or service billing required):

```powershell
pixi run eval-w5-compare --models local_llm
```

Override its Hugging Face model ID with `--llm-model`. The runner scores the model's next-token logits for seven answer codes, maps them to emotion labels, and normalizes them across the choices; these are not guaranteed calibrated probabilities. It rotates the mapping order deterministically by message to reduce fixed answer-position bias. Each model writes machine-readable predictions and tau sweep outputs under `evaluation/results/human_labeled_w5/`; the combined report is `model_comparison_report.md`.

## Current known limitations

- The project's former Laya checkpoint pin may be unavailable; this comparison uses the specifically requested `convaiinnovations/laya` checkpoint, pinned to a resolved Hugging Face revision.
- The held-out split has only one fear example and three anger examples. Their per-class scores and intervals will be unstable and must be described cautiously.
- The dataset contains only one annotator's labels. Results measure agreement with that reviewer, not consensus human truth.
- One human annotator means there is no inter-annotator agreement estimate.
- The separate W=1/W=4 benchmark issue remains untouched; this W=5 run is a separate evaluation and does not claim to complete that issue.

## Verification

The tests in `tests/test_human_labeled_w5.py` and `tests/test_w5_model_comparison.py` cover ID alignment, W=5 context construction, deterministic conversation grouping, tau selection, metrics, and reproducible conversation-cluster bootstrap behavior. Run:

```powershell
pixi run pytest tests/test_human_labeled_w5.py tests/test_w5_model_comparison.py
```
