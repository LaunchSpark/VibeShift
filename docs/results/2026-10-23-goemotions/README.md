# Laya 7-label fine-tune on GoEmotions (2026-10-23)

This folder holds the test results for `LaunchSpark/laya-emotion-tracker` at revision `v2-7-labels`:

- `results.json`: every number.
- `report.png`: the training loss, macro-F1, F1 by label, and the confusion matrix.

## Setup

- **Base model:** `convaiinnovations/laya`, fine-tuned with RLCD (`laya.train.finetune`): 4 epochs, effective batch 64, encoder learning rate 3e-5.
- **Data:** [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), with its 28 labels mapped to the 7 VibeShift labels using Google's GoEmotions → Ekman grouping.
  - Comments whose labels span more than one of the 7 were dropped (4,794), and so were repeated comments (301).
- **Splits:** GoEmotions' official ones, never mixed.
  - **Training:** up to 1,000 comments per label: 6,033 in total, because fear and disgust have only 538 and 495. Laya held out 400 of them to calibrate its confidence.
  - **Validation:** 633 comments. Laya reports progress on these during training.
  - **Test:** 4,922 comments. The official 4,962, minus 40 that also appear in the boltuix model's training data.
- **Hardware and time:** GTX 1070 (8 GB), about 8 hours.

Training and the test cell are in [notebooks/finetune_laya_emotions.py](../../../notebooks/finetune_laya_emotions.py). The comparison was produced by [scripts/laya_test_report.py](../../../scripts/laya_test_report.py).

## Test results (4,922 comments, τ = 0)

| Model | Macro-F1 | Accuracy | Top-2 accuracy | Log-loss | Brier | ECE | JSD of mean probabilities vs. label mix |
|---|---|---|---|---|---|---|---|
| Base Laya (no fine-tuning) | 0.437 | 0.477 | 0.737 | 2.209 | 0.840 | 0.352 | 0.042 |
| Laya on boltuix/emotions-dataset, 300 per label | 0.489 | 0.555 | 0.740 | 1.293 | 0.603 | 0.021 | 0.073 |
| **Laya on GoEmotions (`v2-7-labels`)** | **0.594** | **0.655** | **0.846** | **1.140** | **0.517** | 0.089 | **0.031** |

For `v2-7-labels`, the share of *top* labels has a JSD of **0.007** against the label mix. That's why the dashboard's daily rollups count τ-gated top labels rather than summing probabilities (design doc §3).

Caveat: this test set is Reddit comments from the same source as the training data. The benchmark that counts is the annotated Discord set (design doc §5). Also, `SamLowe/roberta-base-go_emotions` was trained on GoEmotions, so it has a home-field advantage on this test set.
