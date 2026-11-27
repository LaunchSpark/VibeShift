# Combined W=5 Laya Retraining

Laya's training framework fine-tuned the pretrained checkpoint on the combined human and machine training split. Each metric below is recalculated using the same held-out split and training-only tau calibration subset as the prior evaluation.

Machine-only results measure agreement with pretrained-Laya pseudo-labels; use human-only results as the primary quality comparison.

| Model | Partition | N | Accuracy | Balanced accuracy | Macro F1 | Cohen's kappa | Tau |
|---|---|---:|---:|---:|---:|---:|---:|
| pretrained_bot_laya | combined | 614 | 0.865 | 0.753 | 0.724 | 0.658 | 0.7215 |
| pretrained_bot_laya | human_only | 214 | 0.626 | 0.373 | 0.316 | 0.262 | 0.6642 |
| pretrained_bot_laya | machine_only | 400 | 0.995 | 0.999 | 0.992 | 0.985 | 0.7519 |
| learned_behavior_laya_previous | combined | 614 | 0.837 | 0.663 | 0.635 | 0.590 | 0.5756 |
| learned_behavior_laya_previous | human_only | 214 | 0.659 | 0.331 | 0.305 | 0.228 | 0.5939 |
| learned_behavior_laya_previous | machine_only | 400 | 0.932 | 0.898 | 0.842 | 0.815 | 0.5741 |
| learned_behavior_laya_retrained | combined | 614 | 0.816 | 0.612 | 0.602 | 0.495 | 0.0000 |
| learned_behavior_laya_retrained | human_only | 214 | 0.654 | 0.322 | 0.275 | 0.171 | 0.0000 |
| learned_behavior_laya_retrained | machine_only | 400 | 0.912 | 0.843 | 0.789 | 0.732 | 0.3971 |
