# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import re
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    return Path, math, mo, plt, re, time


@app.cell
def _(mo):
    mo.md(r"""
    # Laya training monitor

    Follows a fine-tuning run by re-reading its log. It only reads the log, so opening or
    closing this page never affects training. Start a run with
    `notebooks/finetune_laya_emotions.py` (script mode logs a loss line every 10 steps).
    """)
    return


@app.cell
def _(mo):
    log_path = mo.ui.text(
        value=str(mo.notebook_dir().parent / "operations" / "data" / "laya" / "overnight_train.log"),
        label="Training log",
        full_width=True,
    )
    refresh = mo.ui.refresh(options=["5s", "15s", "60s"], default_interval="15s")
    mo.vstack([log_path, refresh])
    return log_path, refresh


@app.cell
def _(Path, log_path, math, re, refresh, time):
    refresh  # re-read the log on every tick

    path = Path(log_path.value)
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n") if path.exists() else ""
    started = re.search(r"run started at (\d+(?:\.\d+)?)", text)
    started_at = float(started[1]) if started else (path.stat().st_ctime if path.exists() else time.time())

    def parse_run(chunk):
        """One training run: its step budget, logged losses and epoch means."""
        budget = re.match(r"(\d+) train item.*?micro_batch (\d+).*?over (\d+) epoch", chunk)
        n_items, micro, epochs = (int(g) for g in budget.groups())
        per_epoch = math.ceil(n_items / micro)
        return {
            "items": n_items,
            "epochs": epochs,
            "per_epoch": per_epoch,
            "total": per_epoch * epochs,
            "steps": [((int(e) - 1) * per_epoch + int(s), float(v))
                      for e, s, v in re.findall(r"epoch (\d+)/\d+ step (\d+) loss ([\d.]+)", chunk)],
            "means": [(int(e), float(v))
                      for e, v in re.findall(r"epoch (\d+)/\d+ mean loss ([\d.]+)", chunk)],
            "accuracy": re.search(r"Accuracy\s+([\d.]+)\s+([\d.]+)", chunk),
        }

    # The notebook trains the main model first, then one small run per cross-validation fold.
    runs = [parse_run(chunk) for chunk in text.split("optimizer budget: ")[1:]]
    main_run = runs[0] if runs else None
    failed = "Traceback (most recent call last)" in text
    report = path.parent / "report" / "report.png"
    finished = report.exists() and report.stat().st_mtime > started_at
    return failed, finished, main_run, report, runs, started_at


@app.cell
def _(failed, finished, main_run, mo, runs, started_at, time):
    def duration(seconds):
        h, m = divmod(int(seconds) // 60, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    elapsed = time.time() - started_at
    if failed:
        phase = "Stopped with an error: see the end of the log."
    elif finished:
        phase = "Done: the report is below."
    elif not runs:
        phase = "Loading the dataset and the base model…"
    elif main_run["accuracy"] is None:
        phase = "Training the main model"
    elif len(runs) == 1:
        phase = "Main model trained; cross-validation is starting"
    else:
        phase = f"Cross-validation fold {len(runs) - 1} of 3, then scoring the test set"

    done = main_run["steps"][-1][0] if main_run and main_run["steps"] else 0
    total = main_run["total"] if main_run else 1
    training = main_run is not None and main_run["accuracy"] is None and not failed
    # Rough: assumes the pace so far holds. Includes ~2 min of setup before step 1.
    remaining = elapsed / done * (total - done) if training and done else None

    epoch = min(done // main_run["per_epoch"] + 1, main_run["epochs"]) if main_run else 0
    mo.vstack([
        mo.md(f"### {phase}"),
        mo.Html(
            f'<progress value="{done}" max="{total}" style="width:100%;height:22px"></progress>'
        ),
        mo.hstack(
            [
                mo.stat(f"{100 * done / total:.0f}%", label="Main model",
                        caption=f"step {done:,} of {total:,}"),
                mo.stat(f"{epoch} of {main_run['epochs']}" if main_run else "–", label="Epoch"),
                mo.stat(f"{main_run['steps'][-1][1]:.3f}" if done else "–", label="Latest loss"),
                mo.stat(duration(elapsed), label="Elapsed"),
                mo.stat(duration(remaining) if remaining else "–", label="Main model left",
                        caption="rough estimate"),
            ],
            widths="equal",
        ),
    ])
    return


@app.cell
def _(main_run, mo, plt):
    mo.stop(not (main_run and main_run["steps"]), mo.md("The loss chart appears after the first logged step."))

    xs = [s for s, _ in main_run["steps"]]
    ys = [v for _, v in main_run["steps"]]
    window = 10
    smooth = [sum(ys[max(0, i - window + 1) : i + 1]) / len(ys[max(0, i - window + 1) : i + 1])
              for i in range(len(ys))]

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(xs, ys, color="#86b6ef", lw=1, label="Loss every 10 steps")
    ax.plot(xs, smooth, color="#2a78d6", lw=2.5, label=f"Average of last {window}")
    if main_run["means"]:
        ax.plot([e * main_run["per_epoch"] for e, _ in main_run["means"]],
                [v for _, v in main_run["means"]], "o", color="#104281", ms=8,
                label="Epoch mean")
    for e in range(1, main_run["epochs"]):
        ax.axvline(e * main_run["per_epoch"], color="#e4e3df", lw=1)
    ax.set_xlim(0, main_run["total"])
    ax.set_xlabel("Training step", color="#52514e")
    ax.set_ylabel("Loss", color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Training loss (main model)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig
    return


@app.cell
def _(main_run, mo):
    mo.stop(not (main_run and main_run["accuracy"]))
    before, after = (float(v) for v in main_run["accuracy"].groups())
    mo.md(
        f"**Validation accuracy:** {before:.3f} before fine-tuning → **{after:.3f}** after "
        f"({after - before:+.3f}). The test-set scores come in the report below."
    )
    return


@app.cell
def _(finished, mo, report):
    mo.stop(not finished)
    mo.image(src=str(report))
    return


if __name__ == "__main__":
    app.run()
