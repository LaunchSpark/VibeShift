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

import numpy as np

from evaluation.common import io
from evaluation.eval import charts, frames

REPORT_DIR = io.ROOT / "eval" / "report"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya", help="prediction file stem to sweep")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--grid", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                    help="τ grid (default 0.15 0.85 0.05)")
    args = ap.parse_args()

    if not io.prediction_path(args.model).exists():
        raise SystemExit(f"no predictions/{args.model}.jsonl — run its model first.")

    messages = io.load_messages()
    gold = frames.filter_split(messages, io.load_consensus(), args.split)
    taus = frames.TAU_GRID
    if args.grid:
        start, stop, step = args.grid
        taus = tuple(round(t, 4) for t in np.arange(start, stop + 1e-9, step))

    table = frames.sweep_table(io.load_predictions(args.model), gold, taus)
    print(f"τ sweep for '{args.model}' on split={args.split} ({len(gold)} gold messages)\n")
    print(table)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv = REPORT_DIR / f"tau_sweep_{args.model}.csv"
    table.write_csv(csv)
    png = REPORT_DIR / f"tau_sweep_{args.model}.png"
    charts.plot_tau_sweep(table, title=f"{args.model} — τ sweep").save(str(png), ppi=200)
    print(f"\nWrote {csv.relative_to(io.ROOT.parent)} and {png.name}")


if __name__ == "__main__":
    main()
