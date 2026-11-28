"""NRC EmoLex baselines (non-commercial research/educational use only).

EmoLex is copyrighted by National Research Council Canada and may not be
redistributed. Obtain it from the official page and use the local extraction
helper documented in ``evaluation/models/nrc_lexicon/README.md``. The lexicon
file stays ignored and local.

The 8 NRC emotions map to our schema by retaining joy, sadness, anger, fear,
disgust, and surprise; trust and anticipation are dropped because the 7-label
scheme has no matching classes. Polarity columns are unused. Ties use the first
label in ``LABELS`` order.

``nrc_wordcount`` counts hits in W=4 context (three preceding messages plus the
target), with equal weight per token. ``nrc_macro`` counts only
``preceding_25``; the target is excluded so macro mood is independent of the
target classifier input.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

from evaluation.common import io
from evaluation.common.labels import LABELS, NEUTRAL

NRC_TO_7: dict[str, str] = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    # trust and anticipation are deliberately dropped: no schema equivalent.
}

LEXICON_PATH = Path(__file__).resolve().parent / "data" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DISCORD_TAG_RE = re.compile(r"<(?:(?:@!?\d+)|(?:@&\d+)|(?:#\d+)|(?:a?:[A-Za-z0-9_~]+:\d+))>")
_WORD_RE = re.compile(r"\b[^\W_]+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase words after removing URLs and Discord mention/emoji markup."""
    text = _URL_RE.sub(" ", text)
    text = _DISCORD_TAG_RE.sub(" ", text)
    return [token.casefold() for token in _WORD_RE.findall(text)]


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, frozenset[str]]:
    """Load positive word-emotion associations from the official tab-separated file."""
    if not path.exists():
        raise SystemExit(
            f"NRC EmoLex is missing at {path}. Obtain the non-commercial research/educational "
            "lexicon from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm, "
            "then run `python -m evaluation.models.nrc_lexicon.prepare_lexicon <download.zip>`."
        )
    words: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                continue
            word, emotion, association = fields
            label = NRC_TO_7.get(emotion)
            if label and association == "1":
                words.setdefault(word.casefold(), set()).add(label)
    if not words:
        raise ValueError(f"No supported NRC associations found in {path}")
    return {word: frozenset(labels) for word, labels in words.items()}


def _counts(text: str, lexicon: dict[str, frozenset[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in tokenize(text):
        counts.update(lexicon.get(token, ()))
    return counts


def _prediction(counts: Counter[str]) -> tuple[dict[str, float], str]:
    total = sum(counts.values())
    if not total:
        probs = {label: float(label == NEUTRAL) for label in LABELS}
    else:
        probs = {label: counts[label] / total for label in LABELS}
    # max() is stable, so exact ties resolve in the canonical LABELS order.
    return probs, max(LABELS, key=probs.__getitem__)


def predict_wordcount(messages: list[dict], lexicon: dict[str, frozenset[str]] | None = None):
    """Score target plus its preceding three messages; return prediction and ms per row."""
    lexicon = lexicon or load_lexicon()
    predictions = []
    for message in messages:
        context = [*message.get("preceding_3", []), message.get("content", "")]
        start = time.perf_counter()
        counts = Counter()
        for text in context:
            counts.update(_counts(text, lexicon))
        elapsed = (time.perf_counter() - start) * 1000
        predictions.append((*_prediction(counts), elapsed))
    return predictions


def predict_macro(messages: list[dict], lexicon: dict[str, frozenset[str]] | None = None):
    """Score preceding_25 only, excluding the target message."""
    lexicon = lexicon or load_lexicon()
    predictions = []
    for message in messages:
        start = time.perf_counter()
        counts = Counter()
        for text in message.get("preceding_25", []):
            counts.update(_counts(text, lexicon))
        elapsed = (time.perf_counter() - start) * 1000
        predictions.append((*_prediction(counts), elapsed))
    return predictions


def _write_role(model: str, messages: list[dict], results: list[tuple[dict, str, float]]) -> None:
    rows = [
        {
            "id": message["id"],
            "model": model,
            "probs": probs,
            "raw_top_label": raw_top,
            "latency_ms": round(latency, 4),
            "cost_usd": 0.0,
        }
        for message, (probs, raw_top, latency) in zip(messages, results)
    ]
    io.write_predictions(model, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["wordcount", "macro"], default="wordcount")
    args = ap.parse_args()
    messages = io.load_messages()
    lexicon = load_lexicon()
    if args.role == "wordcount":
        _write_role("nrc_wordcount", messages, predict_wordcount(messages, lexicon))
    else:
        _write_role("nrc_macro", messages, predict_macro(messages, lexicon))


if __name__ == "__main__":
    main()
