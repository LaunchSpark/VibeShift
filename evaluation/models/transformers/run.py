"""Pretrained transformer baselines (owned by Teammate C).

Two models, selected with ``--model``:
  * distilroberta  — j-hartmann/emotion-english-distilroberta-base
  * goemotions     — SamLowe/roberta-base-go_emotions (28 classes mapped to 7)

Each runner loads the Hugging Face text-classification pipeline, runs inference,
and maps native labels onto ``evaluation.common.labels.LABELS``.
``io.normalize_probs`` handles renormalization.

    pixi run eval-transformers              # defaults to --model distilroberta
    python -m evaluation.models.transformers.run --model goemotions
"""

from __future__ import annotations

import argparse

from evaluation.models import _common
from evaluation.common.labels import LABELS, NEUTRAL

# ---------------------------------------------------------------------------
# GoEmotions 28 → 7 mapping (from the design doc's grouping table)
# ---------------------------------------------------------------------------
GOEMOTIONS_TO_7: dict[str, str] = {
    # joy group
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "admiration": "joy",
    "desire": "joy",
    "caring": "joy",
    # sadness group
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    # anger group
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # fear group
    "fear": "fear",
    "nervousness": "fear",
    # disgust group
    "disgust": "disgust",
    # surprise group
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    # neutral
    "neutral": "neutral",
}

# Cache pipelines so they're loaded once per process.
_pipeline_cache: dict[str, object] = {}


def _get_pipeline(model_id: str) -> object:
    """Lazily load and cache a Hugging Face text-classification pipeline."""
    if model_id not in _pipeline_cache:
        from transformers import pipeline  # heavy import; keep it lazy
        _pipeline_cache[model_id] = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,         # return all labels with scores
            truncation=True,
            max_length=512,
        )
    return _pipeline_cache[model_id]


# ---------------------------------------------------------------------------
# DistilRoBERTa-Emotion: native 7 labels (Ekman's 6 + neutral)
# ---------------------------------------------------------------------------
_DISTILROBERTA_ID = "j-hartmann/emotion-english-distilroberta-base"

# DistilRoBERTa uses these label names; map to our canonical LABELS.
_DISTILROBERTA_MAP: dict[str, str] = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral",
}


def _distilroberta_probs(scores: list[dict]) -> tuple[dict[str, float], str]:
    """Convert one message's list of {label, score} dicts to (probs, raw_top)."""
    probs = {label: 0.0 for label in LABELS}
    for entry in scores:
        mapped = _DISTILROBERTA_MAP.get(entry["label"])
        if mapped:
            probs[mapped] += entry["score"]
    # raw top label (argmax before normalization)
    raw_top = max(probs, key=probs.__getitem__)
    return probs, raw_top


def predict_distilroberta(messages: list[dict]) -> list[_common.Prediction]:
    pipe = _get_pipeline(_DISTILROBERTA_ID)
    texts = [m["content"] for m in messages]
    all_scores = pipe(texts, batch_size=8)
    return [_distilroberta_probs(scores) for scores in all_scores]


# ---------------------------------------------------------------------------
# RoBERTa-GoEmotions: 28 independent sigmoids → sum within groups → normalize
# ---------------------------------------------------------------------------
_GOEMOTIONS_ID = "SamLowe/roberta-base-go_emotions"


def _goemotions_probs(scores: list[dict]) -> tuple[dict[str, float], str]:
    """Sum sigmoid outputs within each of the 7 groups, then pick argmax."""
    probs = {label: 0.0 for label in LABELS}
    for entry in scores:
        mapped = GOEMOTIONS_TO_7.get(entry["label"])
        if mapped:
            probs[mapped] += entry["score"]
    raw_top = max(probs, key=probs.__getitem__)
    return probs, raw_top


def predict_goemotions(messages: list[dict]) -> list[_common.Prediction]:
    pipe = _get_pipeline(_GOEMOTIONS_ID)
    texts = [m["content"] for m in messages]
    all_scores = pipe(texts, batch_size=8)
    return [_goemotions_probs(scores) for scores in all_scores]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
_RUNNERS = {
    "distilroberta": ("transformers_distilroberta", predict_distilroberta),
    "goemotions": ("transformers_goemotions", predict_goemotions),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(_RUNNERS), default="distilroberta")
    args = ap.parse_args()
    model_name, predict_fn = _RUNNERS[args.model]
    _common.run(model_name, predict_fn, cost_per_msg=0.0)


if __name__ == "__main__":
    main()
