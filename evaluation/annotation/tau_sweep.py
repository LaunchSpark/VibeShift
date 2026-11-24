"""τ threshold sensitivity sweep (owned by Teammates A & B, with Lucas).

Because prediction files store un-thresholded probabilities, the sweep is a pure
post-processing pass: for each τ in a grid (default 0.15-0.85), apply the
universal gating rule and report Macro-F1 and the JSD between the τ-gated
top-label counts and the gold label mix. Use it to pick ``VIBESHIFT_TAU`` for the
production rollup. Writes a CSV and an Altair chart to eval/report/; drag τ live
in ``pixi run eval-notebook``.

    pixi run eval-tau                        # sweeps the laya predictions, test split
    python -m evaluation.annotation.tau_sweep --model nrc_wordcount --split test
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from evaluation.common import io
from evaluation.eval import charts, frames

REPORT_DIR = io.ROOT / "eval" / "report"


def select_tau_star(table: pl.DataFrame) -> float:
    """Select τ* according to Issue #56:

    τ* = argmax dev macro-F1; tie-break on lower JSD (jsd_gated).
    If multiple rows tie on both, picks the lowest τ.
    """
    sorted_table = table.sort(
        by=["macro_f1", "jsd_gated", "tau"],
        descending=[True, False, False]
    )
    return float(sorted_table["tau"][0])


def update_tau_star_json(model: str, tau_star: float) -> None:
    """Record or update model's tau_star in eval/report/tau_star.json."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORT_DIR / "tau_star.json"
    data = {}
    if out_file.exists():
        try:
            data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[model] = tau_star
    out_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya", help="prediction file stem to sweep, or 'all'")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--window", type=int, help="window size whose gold labels to use")
    ap.add_argument("--grid", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                    help="τ grid (default: 0.0 then 0.15 to 0.85 by 0.05)")
    args = ap.parse_args()

    models = [args.model]
    if args.model == "all":
        pred_files = list(io.PREDICTIONS.glob("*.jsonl"))
        models = [p.stem for p in pred_files]
        if not models:
            raise SystemExit("no prediction files in predictions/")

    windows = io.consensus_windows()
    target_window = args.window
    if target_window is None:
        if 4 in windows:
            target_window = 4
        elif windows:
            target_window = min(windows)

    messages = io.load_messages()
    gold = frames.filter_split(messages, io.load_consensus(window=target_window), args.split)
    if not gold:
        raise SystemExit(
            f"no gold labels for the messages in messages.jsonl (split={args.split}). "
            "Label, export, then run `pixi run label-consensus --window W`."
        )

    taus = frames.TAU_GRID
    if args.grid:
        start, stop, step = args.grid
        taus = tuple(round(t, 4) for t in np.arange(start, stop + 1e-9, step))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for m in models:
        if not io.prediction_path(m).exists():
            print(f"Skipping {m}: no predictions/{m}.jsonl")
            continue

        rows = io.load_predictions(m)
        table = frames.sweep_table(rows, gold, taus)
        tau_star = select_tau_star(table)
        print(f"τ sweep for '{m}' on split={args.split} ({len(gold)} gold messages)")
        print(f"Selected τ* = {tau_star:.2f}\n")
        print(table)

        csv = REPORT_DIR / f"tau_sweep_{m}.csv"
        table.write_csv(csv)
        try:
            png = REPORT_DIR / f"tau_sweep_{m}.png"
            charts.save_chart(
                charts.plot_tau_sweep(table, title=f"{m} — τ sweep"),
                png,
                ppi=200,
            )
            print(f"\nWrote {csv.relative_to(io.ROOT.parent)} and {png.name}")
        except Exception as e:
            print(f"\nWrote {csv.relative_to(io.ROOT.parent)} (chart render skipped: {e})")

        if args.split == "dev":
            update_tau_star_json(m, tau_star)
            print(f"Updated {REPORT_DIR / 'tau_star.json'}: {m} = {tau_star:.2f}")


if __name__ == "__main__":
    main()
