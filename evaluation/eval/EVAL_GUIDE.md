# Statistical Evaluation & Threshold Analysis Guide (EVAL_GUIDE)

This guide documents the statistical evaluation tools, the thresholding method, preliminary observations from early manual annotations, and standard `pixi run` execution commands.

---

## 1. Scope & Capabilities Built

Our work focuses on **statistical evaluation, uncertainty quantification, and confidence threshold optimization**:

- **Confidence Thresholding ($\tau$):** An adjustable confidence cutoff. If a model's predicted probability for non-neutral emotions is below $\tau$, the prediction defaults to `neutral` rather than guessing a strong emotion.
- **Automated Threshold Optimization (`pixi run eval-tau`):** Evaluates candidate cutoffs ($\tau = 0.0$ to $0.85$) on the development split, selects $\tau^*$ (maximizing balanced Macro-F1 with ties broken by lowest distribution mismatch), and outputs `eval/report/tau_star.json`.
- **Model Evaluation with Uncertainty (`pixi run eval-report`):** Compares models side-by-side at baseline ($\tau = 0.0$) and optimal threshold ($\tau^*$), computing 95% percentile bootstrap confidence intervals ($1{,}000$ iterations) and flagging small sample sizes ($n < 30$).
- **Interactive Evaluation Explorer (`pixi run eval-notebook`):** A Marimo notebook to inspect confusion matrices, adjust $\tau$ live, and review individual Discord messages.
- **Qualitative Error Analysis (`error_analysis.py`):** Automatically extracts test misclassifications and categorizes them into failure modes (short replies, slang/emoji, context changes, or ambiguous disagreement).
- **Unit Testing:** 8 comprehensive tests in `tests/test_teammate_b.py` verifying all scoring math, bootstrap bounds, and selection rules.

---

## 2. Preliminary Observations & Methodology

### Observations from Initial Annotations
During exploratory annotation on Discord messages (based on an initial set of ~1,100 messages annotated by a single reviewer), an overwhelming majority of text—approximately **78%**—was classified as `neutral`.
- **Causes:** Discord dialogue features brief logistical replies (*"ready"*, *"what time?"*, *"nah"*, *"gg"*) and un-annotated voice-chat responses that lack vocal inflection, tone, and emoji cues.
- **Model Behavior:** Standard models trained on expressive datasets (e.g. Reddit) tend to over-predict emotion on brief dialogue, misclassifying flat statements like *"nah"* as `disgust`.

### How This Applies to the Upcoming Team Consensus
- The 78% figure is a **preliminary baseline observation** from single-annotator data.
- Once the multi-annotator group consensus (`consensus_w4.jsonl`) is compiled, the exact neutral percentage and class distribution will shift to reflect the team average.
- **The Thresholding Method Remains Unchanged:** Regardless of the exact final neutral percentage, the $\tau$ selection rule automatically adapts. It tests cutoffs on the `dev` split, finds the confidence threshold that balances multi-class accuracy, and breaks ties by aligning predicted distributions with the team's true human distribution.

---

## 3. How to Run the Tools

All tasks use the repository's native `pixi run` structure:

### 1. Run the Threshold Sensitivity Sweep
Sweeps models across confidence thresholds ($\tau = 0.0$ to $0.85$) on the development split and saves optimal thresholds to `eval/report/tau_star.json`:
```bash
pixi run eval-tau --model all --split dev
```
*(You can also sweep a specific model, e.g. `pixi run eval-tau --model distilroberta`)*

### 2. Generate the Benchmark Evaluation Report
Evaluates predictions against consensus gold labels with 95% bootstrap confidence intervals and per-class precision/recall:
```bash
pixi run eval-report --split test --with-tau-star
```
Output is written to `eval/report/metrics.md`.

### 3. Run Interactive Evaluation Explorer
Launches the interactive Marimo notebook to inspect confusion matrices, adjust $\tau$ live, and examine individual Discord messages:
```bash
pixi run eval-notebook
```

### 4. Run Qualitative Error Analysis
Extracts misclassified test messages and groups them by failure mode (short replies, slang/emoji, context changes, or ambiguous disagreement):
```bash
pixi run python -m evaluation.eval.error_analysis --model distilroberta --split all
```
Output is saved to `eval/report/error_analysis.md`.

### 5. Run Unit Tests
Executes the test suite for our statistical and metric functions:
```bash
pixi run pytest tests/test_teammate_b.py -v
```
