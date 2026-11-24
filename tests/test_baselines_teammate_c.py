"""Tests for Teammate C baseline runners and mappings."""

import pytest

from evaluation.common.labels import LABELS, NEUTRAL
from evaluation.models.transformers.run import (
    GOEMOTIONS_TO_7,
    _distilroberta_probs,
    _goemotions_probs,
)
from evaluation.models.llm.run import (
    _build_prompt,
    _parse_probs,
)

# Canonical 28 GoEmotions labels from the paper / Hugging Face dataset
ALL_28_GOEMOTIONS = {
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
}


def test_goemotions_covers_all_28_labels_exactly_once():
    """Issue #69 acceptance test: mapping covers all 28 labels exactly once."""
    keys = set(GOEMOTIONS_TO_7.keys())
    assert len(GOEMOTIONS_TO_7) == 28, f"Expected 28 keys, got {len(GOEMOTIONS_TO_7)}"
    assert keys == ALL_28_GOEMOTIONS, f"Missing: {ALL_28_GOEMOTIONS - keys}, Extra: {keys - ALL_28_GOEMOTIONS}"

    # Every mapped target must be in canonical LABELS
    for src, dst in GOEMOTIONS_TO_7.items():
        assert dst in LABELS, f"{src} mapped to invalid label {dst}"


def test_goemotions_probs_aggregation():
    """Verify independent sigmoids are summed into their 7 target groups."""
    raw_scores = [
        {"label": "joy", "score": 0.3},
        {"label": "amusement", "score": 0.4},  # joy total = 0.7
        {"label": "anger", "score": 0.2},      # anger total = 0.2
        {"label": "neutral", "score": 0.1},    # neutral total = 0.1
    ]
    probs, raw_top = _goemotions_probs(raw_scores)
    assert raw_top == "joy"
    assert pytest.approx(probs["joy"]) == 0.7
    assert pytest.approx(probs["anger"]) == 0.2
    assert pytest.approx(probs["neutral"]) == 0.1
    assert probs["sadness"] == 0.0
    assert probs["fear"] == 0.0
    assert probs["disgust"] == 0.0
    assert probs["surprise"] == 0.0


def test_distilroberta_probs_mapping():
    """Verify DistilRoBERTa emotion scores map to canonical LABELS."""
    raw_scores = [
        {"label": "joy", "score": 0.6},
        {"label": "sadness", "score": 0.1},
        {"label": "anger", "score": 0.05},
        {"label": "fear", "score": 0.05},
        {"label": "disgust", "score": 0.05},
        {"label": "surprise", "score": 0.05},
        {"label": "neutral", "score": 0.1},
    ]
    probs, raw_top = _distilroberta_probs(raw_scores)
    assert raw_top == "joy"
    assert set(probs.keys()) == set(LABELS)
    assert pytest.approx(probs["joy"]) == 0.6


def test_llm_parse_probs_valid():
    """Test JSON parsing of LLM response."""
    json_text = '{"joy": 0.7, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "disgust": 0.05, "surprise": 0.05, "neutral": 0.05}'
    probs = _parse_probs(json_text)
    assert probs is not None
    assert probs["joy"] == 0.7
    assert set(probs.keys()) == set(LABELS)


def test_llm_parse_probs_with_extra_text():
    """Test extracting JSON from conversational LLM response."""
    text = 'Here is the result:\n```json\n{"joy": 0.1, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "disgust": 0.1, "surprise": 0.1, "neutral": 0.4}\n```\nHope that helps!'
    probs = _parse_probs(text)
    assert probs is not None
    assert probs["neutral"] == 0.4


def test_llm_parse_probs_invalid():
    """Test that malformed responses return None so retry/fallback triggers."""
    assert _parse_probs("Not a json") is None
    assert _parse_probs('{"joy": 0.5}') is None  # missing required labels


def test_llm_prompt_building():
    """Test that prompt includes preceding context and target message."""
    template = "Context:\n{context}\nTarget: {message}"
    msg = {
        "content": "I am so excited!",
        "preceding_3": ["are you coming?", "yes!"]
    }
    prompt = _build_prompt(msg, template)
    assert "are you coming?" in prompt
    assert "yes!" in prompt
    assert "I am so excited!" in prompt
