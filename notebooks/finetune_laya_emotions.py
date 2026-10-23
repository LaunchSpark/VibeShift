# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "laya @ git+https://github.com/NandhaKishorM/laya.git",
#     "datasets",
#     "huggingface_hub",
#     "matplotlib",
#     "scikit-learn",
#     "torch",
# ]
#
# # PyPI's Windows torch is CPU-only, so take a CUDA build from PyTorch's index there.
# # cu126 still supports older GPUs (Pascal, e.g. GTX 10xx). Linux keeps PyPI's CUDA torch,
# # which supports the Blackwell GPUs on molab that cu126 does not.
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pkgutil

    import laya
    import laya.train
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from datasets import Dataset, concatenate_datasets, load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from scipy.spatial.distance import jensenshannon
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        classification_report,
        f1_score,
    )
    from sklearn.model_selection import KFold

    return (
        ConfusionMatrixDisplay,
        Dataset,
        HfApi,
        KFold,
        classification_report,
        concatenate_datasets,
        f1_score,
        jensenshannon,
        json,
        laya,
        load_dataset,
        mo,
        np,
        pkgutil,
        plt,
        snapshot_download,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Fine-tune Laya on the VibeShift emotion labels

    Fine-tunes the [Laya](https://github.com/NandhaKishorM/laya) decision model on
    [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
    (Reddit comments with documented, multi-rater labels), relabeled to VibeShift's 7
    labels (joy, sadness, anger, fear, disgust, surprise, neutral). It then scores the
    model on GoEmotions' official test split, which training never touches, and can push
    it to the Hugging Face Hub.

    Training only starts when you press a **Train** button. When the notebook runs as a
    script (`uv run notebooks/finetune_laya_emotions.py`), both training steps run
    automatically. Nothing is pushed to the Hub in script mode.

    **On molab:** attach a GPU from the notebook specs button in the header, raise the
    micro-batch in Step 3, and push to the Hub (Step 5) before the session ends. molab
    doesn't keep files the notebook writes, so the checkpoint is lost otherwise.
    """)
    return


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    # Training data and checkpoints go under the git-ignored operations/data/.
    runs_dir = mo.notebook_dir().parent / "operations" / "data" / "laya"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return is_script_mode, runs_dir


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: Inspect the Laya package
    """)
    return


@app.cell
def _(laya, mo, pkgutil):
    laya_submodules = [m.name for m in pkgutil.iter_modules(laya.__path__)]
    mo.md("**laya submodules:** " + ", ".join(f"`{m}`" for m in laya_submodules))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Relabel the dataset and define the schema

    GoEmotions has 27 emotions plus neutral, and a comment can carry several. They are
    grouped onto the 7 VibeShift labels with Google's published GoEmotions → Ekman
    mapping. Comments whose labels fall into more than one VibeShift emotion are dropped,
    since Laya learns one choice per message.

    GoEmotions ships official train / validation / test splits, and they are used as-is:
    training examples come from train only, laya reports progress on validation, and test
    is scored once at the end.

    Laya scores unstructured text (the *state*) against a typed schema (the *question*).
    The 7 labels form one `choice` question, and each row becomes Laya's
    `{state, questions, expected}` JSONL format.
    """)
    return


@app.cell
def _():
    TARGET_LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
    # Google's published GoEmotions -> Ekman grouping (ekman_mapping.json), plus neutral.
    EKMAN_GROUPS = {
        "joy": [
            "joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring",
        ],
        "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
        "anger": ["anger", "annoyance", "disapproval"],
        "fear": ["fear", "nervousness"],
        "disgust": ["disgust"],
        "surprise": ["surprise", "realization", "confusion", "curiosity"],
        "neutral": ["neutral"],
    }
    EKMAN_MAP = {go: ekman for ekman, gos in EKMAN_GROUPS.items() for go in gos}
    return EKMAN_GROUPS, EKMAN_MAP, TARGET_LABELS


@app.cell
def _(
    Dataset,
    EKMAN_GROUPS,
    EKMAN_MAP,
    TARGET_LABELS,
    concatenate_datasets,
    load_dataset,
    mo,
):
    goemotions = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = goemotions["train"].features["labels"].feature.names
    unmapped = set(names) - EKMAN_MAP.keys()
    assert not unmapped, f"GoEmotions labels with no mapping: {sorted(unmapped)}"

    # Test first, then validation, then train: a comment repeated across splits is kept only
    # in the earliest, so nothing is trained on a validation or test comment.
    seen = set()
    parts = []
    dropped = {"spanning several emotions": 0, "repeated": 0}
    for split_name in ("test", "validation", "train"):
        texts, split_labels = [], []
        for text, ids in zip(goemotions[split_name]["text"], goemotions[split_name]["labels"]):
            targets = {EKMAN_MAP[names[n]] for n in ids}
            key = text.strip().lower()
            if len(targets) != 1:
                dropped["spanning several emotions"] += 1
            elif key in seen:
                dropped["repeated"] += 1
            else:
                seen.add(key)
                texts.append(text)
                split_labels.append(targets.pop())
        part = Dataset.from_dict({"Sentence": texts, "Label": split_labels}).shuffle(seed=42)
        parts.append(("train pool" if split_name == "train" else split_name, part))
    rows = concatenate_datasets([part for _, part in parts])

    # Row numbers per label within each split, in each split's shuffled order.
    labels = rows["Label"]
    splits = {}
    offset = 0
    for split_name, part in parts:
        grouped = {label: [] for label in TARGET_LABELS}
        for row in range(offset, offset + len(part)):
            grouped[labels[row]].append(row)
        splits[split_name] = grouped
        offset += len(part)
    splits = {name: splits[name] for name in ("train pool", "validation", "test")}

    mo.ui.table(
        [
            {
                "label": label,
                "GoEmotions labels": ", ".join(EKMAN_GROUPS[label]),
                **{name: len(split[label]) for name, split in splits.items()},
            }
            for label in TARGET_LABELS
        ],
        selection=None,
        label=(
            f"{len(rows):,} comments in GoEmotions' official splits ("
            + ", ".join(f"{v:,} {k}" for k, v in dropped.items())
            + " dropped)"
        ),
    )
    return rows, splits


@app.cell
def _(TARGET_LABELS):
    questions = {
        "emotion_tracker": {
            "type": "choice",
            "instructions": "Identify the primary emotion expressed in the given text message.",
            "criteria": {
                label: f"Select this if the primary emotion is {label}."
                for label in TARGET_LABELS
            },
        }
    }
    return (questions,)


@app.cell
def _(json, questions):
    def write_jsonl(dataset_rows, path):
        """Write dataset rows as Laya training records."""
        with open(path, "w", encoding="utf-8") as f:
            for row in dataset_rows:
                record = {
                    # Laya expects the state to be an object, not a bare string.
                    "state": {"text": row["Sentence"]},
                    "questions": questions,
                    "expected": {"emotion_tracker": row["Label"]},
                }
                f.write(json.dumps(record) + "\n")

    return (write_jsonl,)


@app.cell
def _(mo):
    mo.md(r"""
    Training and validation take up to the same number of rows from every label, so joy
    and neutral don't drown out the rest. Disgust and fear have only about 500 training
    comments each, so they are capped at what exists. The test file is the whole test
    split, at its natural label mix, which macro-F1 then weights equally.
    Laya also holds out 10% of the training file (at most 400 items) to fit its
    calibration temperature.
    """)
    return


@app.cell
def _(mo):
    per_label = mo.ui.slider(10, 5000, step=10, value=1000, label="Training examples per label")
    val_per_label = mo.ui.slider(10, 2000, step=10, value=100, label="Validation examples per label")
    mo.hstack([per_label, val_per_label], justify="start")
    return per_label, val_per_label


@app.cell
def _(mo, per_label, rows, runs_dir, splits, val_per_label, write_jsonl):
    # Each split was shuffled, so the first n of a label are a random draw (or all of them,
    # for labels with fewer). Shuffling the selection keeps the file from being grouped by label.
    train_rows = rows.select(
        [i for idx in splits["train pool"].values() for i in idx[: per_label.value]]
    ).shuffle(seed=0)
    val_rows = rows.select(
        [i for idx in splits["validation"].values() for i in idx[: val_per_label.value]]
    )
    test_rows = rows.select(sorted(i for idx in splits["test"].values() for i in idx))

    train_file = runs_dir / "train_data.jsonl"
    val_file = runs_dir / "validation_data.jsonl"
    test_file = runs_dir / "test_data.jsonl"
    for split_rows, split_file in ((train_rows, train_file), (val_rows, val_file), (test_rows, test_file)):
        write_jsonl(split_rows, split_file)

    mo.md(
        f"Wrote {len(train_rows):,} training, {len(val_rows):,} validation and "
        f"{len(test_rows):,} test examples to `{runs_dir}`."
    )
    return test_file, test_rows, train_file, val_file


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: Fine-tune with RLCD

    Laya is a non-autoregressive decision model built on ModernBERT, trained with
    reinforcement learning against strictly proper scoring rules (RLCD).
    `laya.train.finetune` also fits the calibration temperatures and writes
    `questions.json` next to the weights, so the saved checkpoint is calibrated and carries
    its own label list. Given the validation file, it also reports accuracy before and after
    fine-tuning.
    """)
    return


@app.cell
def _(mo, snapshot_download):
    with mo.status.spinner("Downloading base model convaiinnovations/laya..."):
        base_model_dir = snapshot_download("convaiinnovations/laya")
    return (base_model_dir,)


@app.cell
def _(mo):
    micro_batch = mo.ui.dropdown(
        {"8 (8 GB GPU, Colab T4)": 8, "16": 16, "32": 32, "64 (molab's 96 GB GPU)": 64},
        value="8 (8 GB GPU, Colab T4)",
        label="Micro-batch",
    )
    train_button = mo.ui.run_button(label="Train")
    mo.hstack([micro_batch, train_button], justify="start")
    return micro_batch, train_button


@app.cell
def _(laya, micro_batch):
    def make_config(eval_data=None):
        # The effective batch stays at laya's default of 64 whatever the micro-batch, so a
        # bigger micro-batch only trades GPU memory for speed.
        return laya.train.TrainConfig(
            micro_batch=micro_batch.value,
            grad_accum=max(1, 64 // micro_batch.value),
            encoder_lr=3e-5,
            eval_data=eval_data,
            log_every=10,  # a loss line every 10 steps, for the training monitor
        )

    return (make_config,)


@app.cell
def _(
    base_model_dir,
    is_script_mode,
    laya,
    make_config,
    mo,
    runs_dir,
    train_button,
    train_file,
    val_file,
):
    mo.stop(
        not (train_button.value or is_script_mode),
        mo.md("Press **Train** to start fine-tuning."),
    )

    with mo.status.spinner("Fine-tuning and calibrating..."):
        summary = laya.train.finetune(
            data=str(train_file),
            model_dir=base_model_dir,
            output_dir=str(runs_dir / "laya_emotion_tracker"),
            config=make_config(eval_data=str(val_file)),
        )

    report = summary["train_report"]
    mo.md(f"""
    Fine-tuning and calibration complete.

    - Training items: {summary["train_items"]}
    - Calibration items: {summary["calibration_items"]}
    - Fitted temperature: {summary["temperature"]}
    - Validation accuracy: {report["before"]["accuracy"]:.3f} before → {report["after"]["accuracy"]:.3f} after ({report["eval_items"]} items)
    - Saved to `{summary["output_dir"]}`
    """)
    return (summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3.5: K-fold cross-validation

    Train once per fold and compare the training-loss curves to check that the model
    generalizes rather than memorizing one split.
    """)
    return


@app.cell
def _(mo):
    kfold_rows = mo.ui.slider(30, 1000, step=10, value=150, label="Rows")
    kfold_splits = mo.ui.slider(2, 10, value=3, label="Folds")
    kfold_button = mo.ui.run_button(label="Train folds")
    mo.hstack([kfold_rows, kfold_splits, kfold_button], justify="start")
    return kfold_button, kfold_rows, kfold_splits


@app.cell
def _(
    KFold,
    base_model_dir,
    is_script_mode,
    kfold_button,
    kfold_rows,
    kfold_splits,
    laya,
    make_config,
    mo,
    rows,
    runs_dir,
    splits,
    write_jsonl,
):
    mo.stop(
        not (kfold_button.value or is_script_mode),
        mo.md("Press **Train folds** to run cross-validation."),
    )

    # Folds come from the training pool only, never from validation or test rows.
    pool_rows = rows.select([i for idx in splits["train pool"].values() for i in idx])
    data_subset = list(pool_rows.shuffle(seed=1).select(range(kfold_rows.value)))
    kf = KFold(n_splits=kfold_splits.value, shuffle=True, random_state=42)

    all_fold_losses = []
    for fold, (fold_train_idx, _test_idx) in mo.status.progress_bar(
        list(enumerate(kf.split(data_subset), 1)), title="Training folds"
    ):
        fold_file = runs_dir / f"train_fold_{fold}.jsonl"
        write_jsonl([data_subset[i] for i in fold_train_idx], fold_file)
        fold_summary = laya.train.finetune(
            data=str(fold_file),
            model_dir=base_model_dir,
            output_dir=str(runs_dir / f"laya_fold_{fold}"),
            config=make_config(),
        )
        all_fold_losses.append(fold_summary["epoch_loss"])
    return (all_fold_losses,)


@app.cell
def _(all_fold_losses, plt):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, losses in enumerate(all_fold_losses, 1):
        ax.plot(range(1, len(losses) + 1), losses, marker="o", label=f"Fold {i}")
    ax.set_title("Training loss per fold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Test-set evaluation

    Score every test message with the fine-tuned, calibrated checkpoint. Macro-F1
    weights the 7 labels equally, which is the headline metric in the VibeShift benchmark.
    It only looks at each message's top label, so the distribution metrics score Laya's
    whole probability output too: log-loss and Brier score per message, top-2 accuracy
    for blended emotions, and Jensen-Shannon divergence between the average predicted
    distribution and the human label mix (what the dashboard's daily sums depend on).
    """)
    return


@app.cell
def _(laya, summary):
    agent = laya.Agent(summary["output_dir"])
    return (agent,)


@app.cell
def _(TARGET_LABELS, agent, np, questions, test_rows):
    test_texts = list(test_rows["Sentence"])
    y_true = list(test_rows["Label"])
    # One row per message: the full probability distribution, in TARGET_LABELS order.
    probs = np.array([
        [r.values["emotion_tracker"]["probabilities"][label] for label in TARGET_LABELS]
        for start in range(0, len(test_texts), 64)
        for r in agent.decide_batch(
            test_texts[start : start + 64], questions=questions, return_details=True
        )
    ])
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = [TARGET_LABELS[i] for i in probs.argmax(axis=1)]
    return probs, y_pred, y_true


@app.cell
def _(
    ConfusionMatrixDisplay,
    TARGET_LABELS,
    classification_report,
    f1_score,
    jensenshannon,
    mo,
    np,
    plt,
    probs,
    y_pred,
    y_true,
):
    macro_f1 = f1_score(y_true, y_pred, labels=TARGET_LABELS, average="macro", zero_division=0)
    one_hot = np.array([[t == label for label in TARGET_LABELS] for t in y_true], dtype=float)
    distribution_metrics = {
        # Gold label among the two most likely (sklearn's version needs sorted labels).
        "Top-2 accuracy": (
            np.argsort(-probs, axis=1)[:, :2] == one_hot.argmax(axis=1)[:, None]
        ).any(axis=1).mean(),
        # Computed directly: sklearn's log_loss assumes columns in sorted label order.
        "Log-loss": -np.log(np.clip((probs * one_hot).sum(axis=1), 1e-7, 1)).mean(),
        "Brier score": ((probs - one_hot) ** 2).sum(axis=1).mean(),
        "JSD, average distribution vs label mix": jensenshannon(
            probs.mean(axis=0), one_hot.mean(axis=0), base=2
        ) ** 2,
    }
    cm_fig, cm_ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=TARGET_LABELS, ax=cm_ax, colorbar=False, xticks_rotation=45
    )
    cm_fig.tight_layout()
    mo.vstack([
        mo.md(f"**Test macro-F1: {macro_f1:.3f}** ({len(y_true):,} messages)"),
        mo.hstack(
            [mo.stat(f"{v:.3f}" if v >= 0.001 else f"{v:.4f}", label=k)
             for k, v in distribution_metrics.items()],
            widths="equal",
        ),
        mo.md(
            "```text\n"
            + classification_report(y_true, y_pred, labels=TARGET_LABELS, zero_division=0)
            + "```"
        ),
        cm_fig,
    ])
    return


@app.cell
def _(mo):
    test_message = mo.ui.text(
        value="I'm so incredibly frustrated right now",
        label="Try a message",
        full_width=True,
    )
    test_message
    return (test_message,)


@app.cell
def _(agent, laya, questions, test_message):
    result = laya.decide(agent, test_message.value, questions=questions)
    result
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Push the model to the Hugging Face Hub

    Pushing replaces the model the bot downloads. Use a new repo (or branch) to keep the
    current one. Paste a Hugging Face **write** token, or leave it blank to use the token
    saved by `hf auth login`.
    """)
    return


@app.cell
def _(mo):
    repo_id = mo.ui.text(value="LaunchSpark/laya-emotion-tracker", label="Repo")
    hf_token = mo.ui.text(kind="password", label="HF token")
    push_button = mo.ui.run_button(label="Push to Hub")
    mo.vstack([repo_id, hf_token, push_button])
    return hf_token, push_button, repo_id


@app.cell
def _(HfApi, hf_token, mo, push_button, repo_id, summary):
    mo.stop(not push_button.value)

    api = HfApi(token=hf_token.value or None)
    with mo.status.spinner(f"Uploading to {repo_id.value}..."):
        api.create_repo(repo_id=repo_id.value, exist_ok=True)
        api.upload_folder(
            folder_path=summary["output_dir"],
            repo_id=repo_id.value,
            commit_message="Upload 7-label Laya Emotion Tracker model",
        )
    mo.md(f"Pushed to https://huggingface.co/{repo_id.value}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Using the model

    The checkpoint ships its `questions.json`, so the schema always matches training.
    See `notebooks/laya_emotions_analyzer.py` for batch inference, or use it directly:

    ```python
    import json
    from pathlib import Path

    import laya
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download("LaunchSpark/laya-emotion-tracker", revision="v2-7-labels")
    agent = laya.Agent(model_dir)
    questions = json.loads((Path(model_dir) / "questions.json").read_text())
    result = laya.decide(agent, "this is completely unacceptable!", questions=questions)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
