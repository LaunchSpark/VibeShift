"""Score base Laya and fine-tuned checkpoints on one held-out file; write results.json + report.png.

usage: python scripts/laya_test_report.py LOG TEST_JSONL OUT_DIR NAME=CHECKPOINT_DIR [...] [--exclude JSONL ...] [--limit N]
The base model (convaiinnovations/laya) is always scored first as "Base Laya". --exclude drops
test sentences that appear in the given training files, so no model is scored on its own
training data.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import laya
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
# dataviz reference palette, light mode: categorical slots 1-3, text and grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def parse_log(path):
    """Main-run losses only: everything before the first evaluation table."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    text = text.split("=== Evaluation")[0]
    budget = re.search(r"optimizer budget: (\d+) train item.*?micro_batch (\d+)", text)
    per_epoch = math.ceil(int(budget[1]) / int(budget[2])) if budget else None
    steps = [
        {"epoch": int(e), "step": int(s), "loss": float(v),
         "global_step": (int(e) - 1) * per_epoch + int(s) if per_epoch else None}
        for e, s, v in re.findall(r"epoch (\d+)/\d+ step (\d+) loss ([\d.]+)", text)
    ]
    means = [{"epoch": int(e), "loss": float(v)}
             for e, v in re.findall(r"epoch (\d+)/\d+ mean loss ([\d.]+)", text)]
    return {"steps_per_epoch": per_epoch, "steps": steps, "epoch_means": means}


def score(model_dir, texts, questions, batch=64):
    """Full probability distribution per text, as an (N, 7) array in LABELS order."""
    agent = laya.Agent(model_dir)
    rows = []
    for i in range(0, len(texts), batch):
        for r in agent.decide_batch(texts[i : i + batch], questions=questions, return_details=True):
            probs = r.values["emotion_tracker"]["probabilities"]
            rows.append([float(probs[label]) for label in LABELS])
    P = np.array(rows)
    return P / P.sum(axis=1, keepdims=True)


def soft_metrics(y_true, P):
    """Metrics on the whole distribution, not just its top label."""
    Y = np.array([[t == label for label in LABELS] for t in y_true], dtype=float)
    true_mix = Y.mean(axis=0)
    pred_mix = np.bincount(P.argmax(axis=1), minlength=len(LABELS)) / len(P)
    return {
        # Computed directly: sklearn's log_loss assumes columns in sorted label order.
        "log_loss": float(-np.log(np.clip((P * Y).sum(axis=1), 1e-7, 1)).mean()),
        "brier": float(((P - Y) ** 2).sum(axis=1).mean()),
        # Gold label among the two most likely (sklearn's version needs sorted labels).
        "top2_accuracy": float((np.argsort(-P, axis=1)[:, :2] == Y.argmax(axis=1)[:, None]).any(axis=1).mean()),
        # Aggregate (dashboard-style) agreement with the human label mix, base 2 so 0..1.
        "jsd_mean_probability": float(jensenshannon(P.mean(axis=0), true_mix, base=2) ** 2),
        "jsd_top_label": float(jensenshannon(pred_mix, true_mix, base=2) ** 2),
        "mean_probability": dict(zip(LABELS, P.mean(axis=0).tolist())),
        "true_mix": dict(zip(LABELS, true_mix.tolist())),
    }


def ece(y_true, preds, confs, bins=10):
    hit = np.array([t == p for t, p in zip(y_true, preds)], dtype=float)
    conf = np.array(confs)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("test_file")
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="+", help="NAME=CHECKPOINT_DIR")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def key(text):
        return text.strip().lower()

    excluded = {key(json.loads(l)["state"]["text"])
                for f in a.exclude for l in open(f, encoding="utf-8")}
    all_rows = [json.loads(l) for l in open(a.test_file, encoding="utf-8")]
    rows = [r for r in all_rows if key(r["state"]["text"]) not in excluded][: a.limit]
    texts = [r["state"]["text"] for r in rows]
    y_true = [r["expected"]["emotion_tracker"] for r in rows]
    questions = rows[0]["questions"]

    models = [("Base Laya", snapshot_download("convaiinnovations/laya"))]
    models += [tuple(n.split("=", 1)) for n in a.models]

    results = {"test_file": str(a.test_file), "eval_items": len(rows),
               "excluded_overlap": len(all_rows) - len(rows) if not a.limit else None,
               "labels": LABELS, "training": parse_log(a.log), "models": []}
    for name, path in models:
        print(f"scoring {name} ({path})", flush=True)
        P = score(path, texts, questions)
        preds = [LABELS[i] for i in P.argmax(axis=1)]
        confs = P.max(axis=1).tolist()
        results["models"].append({
            "name": name,
            "path": str(path),
            "accuracy": accuracy_score(y_true, preds),
            "macro_f1": f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0),
            "per_label_f1": dict(zip(LABELS, f1_score(y_true, preds, labels=LABELS, average=None,
                                                      zero_division=0).tolist())),
            "ece": ece(y_true, preds, confs),
            "mean_confidence": float(np.mean(confs)),
            "confusion": confusion_matrix(y_true, preds, labels=LABELS).tolist(),
            **soft_metrics(y_true, P),
        })
        m = results["models"][-1]
        print(f"  macro-F1 {m['macro_f1']:.3f}  accuracy {m['accuracy']:.3f}  "
              f"top-2 {m['top2_accuracy']:.3f}  log-loss {m['log_loss']:.3f}  "
              f"Brier {m['brier']:.3f}  JSD {m['jsd_mean_probability']:.4f}", flush=True)

    train_report = Path(models[-1][1]) / "train_report.json"
    if train_report.exists():
        results["laya_train_report"] = json.loads(train_report.read_text(encoding="utf-8"))
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results, out / "report.png")
    print(f"wrote {out / 'results.json'} and {out / 'report.png'}", flush=True)


def plot(results, path):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)

    # 1. Training loss: logged steps plus per-epoch means, one axis.
    ax = axes[0, 0]
    tr = results["training"]
    if tr["steps"] and tr["steps_per_epoch"]:
        ax.plot([s["global_step"] for s in tr["steps"]], [s["loss"] for s in tr["steps"]],
                color=SERIES[0], lw=1.5, alpha=0.45, marker="o", ms=4, label="Logged step loss")
        ax.plot([m["epoch"] * tr["steps_per_epoch"] for m in tr["epoch_means"]],
                [m["loss"] for m in tr["epoch_means"]], color=SERIES[0], lw=2.5, marker="o",
                ms=8, label="Epoch mean loss")
        ax.legend(frameon=False)
    ax.set_title("Training loss", loc="left", fontweight="bold")
    ax.set_xlabel("Training step")
    ax.grid(axis="y", color=GRID)

    # 2. Macro-F1 per model (single measure, so one hue).
    ax = axes[0, 1]
    names = [m["name"] for m in results["models"]]
    vals = [m["macro_f1"] for m in results["models"]]
    bars = ax.bar(names, vals, color=SERIES[0], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", color=INK)
    ax.set_ylim(0, 1)
    ax.set_title(f"Test macro-F1 ({results['eval_items']:,} messages)", loc="left",
                 fontweight="bold")
    ax.grid(axis="y", color=GRID)

    # 3. Per-label F1, grouped by model (<= 3 series: validated slots 1-3).
    ax = axes[1, 0]
    x = np.arange(len(LABELS))
    w = 0.8 / len(results["models"])
    for i, m in enumerate(results["models"]):
        ax.bar(x + (i - (len(results["models"]) - 1) / 2) * w,
               [m["per_label_f1"][l] for l in LABELS], w * 0.92, color=SERIES[i % 3],
               label=m["name"])
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncols=len(results["models"]), loc="lower left",
              bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title("F1 by label", loc="left", fontweight="bold", pad=26)
    ax.grid(axis="y", color=GRID)

    # 4. Confusion matrix of the last (newest) model, row-normalized, one-hue sequential.
    ax = axes[1, 1]
    cm = np.array(results["models"][-1]["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"]), vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else INK, fontsize=9)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.spines[:].set_visible(False)
    ax.set_title(f"Confusion matrix: {results['models'][-1]['name']}", loc="left",
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
