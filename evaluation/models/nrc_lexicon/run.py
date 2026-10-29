"""NRC Emotion Lexicon (EmoLex) baselines (owned by Teammate D).

Two roles, both writing Contract-2 prediction files:

1. ``nrc_wordcount`` — per-message 7-class word-count distribution over the
   message text, with a neutral fallback when no lexicon words match.
2. ``nrc_macro`` — the M=25 rolling macro-window mood vector: count lexicon hits
   across ``preceding_25`` (the context window in messages.jsonl). This vector is
   also the context feature the stacking head concatenates with Laya's probs.

NRC EmoLex has 8 emotions + 2 sentiments. Map to our 7 (drop/ fold `trust` and
`anticipation`, keep `joy/sadness/anger/fear/disgust/surprise`, add `neutral` as
the fallback). Define that mapping in ``NRC_TO_7`` once the lexicon is loaded.

STUB STATE: returns uniform probs so `pixi run eval-nrc` → `eval-report` runs
today. Teammate D: download EmoLex, implement ``_counts(text)``, and fill in both
roles.

    pixi run eval-nrc                        # --role wordcount (default)
    python -m evaluation.models.nrc_lexicon.run --role macro
"""

from __future__ import annotations

import argparse

from evaluation.common import io
from evaluation.models import _common
from evaluation.common.labels import NEUTRAL

# EmoLex emotion -> our 7 labels. Teammate D: complete once the lexicon is loaded.
NRC_TO_7: dict[str, str] = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    # "trust" / "anticipation" have no 7-label home — fold or drop (D's call).
}


def predict_wordcount(texts: list[str]) -> list[_common.Prediction]:
    print("\033[33m[nrc] STUB (wordcount) — uniform probs. "
          "Teammate D: implement EmoLex word counting here.\033[0m")
    return [(_common.uniform(), NEUTRAL) for _ in texts]


def predict_macro(messages: list[dict]) -> list[_common.Prediction]:
    print("\033[33m[nrc] STUB (macro) — uniform probs. "
          "Teammate D: count EmoLex hits across preceding_25 here.\033[0m")
    return [(_common.uniform(), NEUTRAL) for _ in messages]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["wordcount", "macro"], default="wordcount")
    args = ap.parse_args()

    if args.role == "wordcount":
        _common.run("nrc_wordcount", predict_wordcount)
    else:
        # macro needs the preceding_25 window, so it runs over full message rows.
        messages = io.load_messages()
        preds = predict_macro(messages)
        rows = [
            {
                "id": m["id"],
                "model": "nrc_macro",
                "probs": io.normalize_probs(probs),
                "raw_top_label": raw_top,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
            }
            for m, (probs, raw_top) in zip(messages, preds)
        ]
        io.write_predictions("nrc_macro", rows)


if __name__ == "__main__":
    main()
