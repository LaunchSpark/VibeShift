"""Emotion classification with the fine-tuned Laya emotion-tracker model."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import laya
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MODEL_REPO = "LaunchSpark/laya-emotion-tracker"
# The formerly pinned ``v2-7-labels`` ref is no longer present on Hugging Face.
# Pin the available checkpoint commit; its task schema is overridden below to
# the project's seven labels so the bot and evaluation use the same output set.
MODEL_REVISION = "e7fdadbb5de7efe2ac9dad4511509df14a793554"
# Keep checkpoint identity and inference contract together so W=1 scores are
# distinguishable from scores made with the W=5 context prompt.
MODEL_ID = f"{MODEL_REPO}@{MODEL_REVISION}+w5-context-v1"
CONTEXT_INSTRUCTIONS = (
    "Classify the emotion expressed by the TARGET MESSAGE. Use earlier messages "
    "only to understand its context; do not classify their emotions. Choose the "
    "single best category for the target."
)
EMOTION_CRITERIA = {
    "joy": "Happiness, amusement, excitement, gratitude, affection, or positive enjoyment.",
    "sadness": "Sadness, grief, disappointment, loneliness, or emotional distress.",
    "anger": "Anger, irritation, frustration, hostility, or outrage.",
    "fear": "Fear, anxiety, worry, nervousness, or threat.",
    "disgust": "Disgust, revulsion, contempt, or strong aversion.",
    "surprise": "Surprise, astonishment, or unexpectedness without a stronger other emotion.",
    "neutral": "No clear emotion is expressed by the target message in its context.",
}


@cache
def load_model() -> tuple["laya.Agent", dict]:
    """Download (once, then cached on disk) and load the agent and its schema.

    The question schema ships with the model as questions.json, so it always
    matches what the model was trained on.
    """
    model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION)
    agent = laya.Agent(model_dir)
    questions = json.loads((Path(model_dir) / "questions.json").read_text())
    # Laya accepts a question schema at inference time. The available checkpoint
    # has a stale 13-label schema, so use the project's seven labels and the
    # context-aware instruction without modifying its large weight files.
    questions["emotion_tracker"]["instructions"] = CONTEXT_INSTRUCTIONS
    questions["emotion_tracker"]["criteria"] = EMOTION_CRITERIA
    return agent, questions


def emotion_labels() -> list[str]:
    _, questions = load_model()
    return list(questions["emotion_tracker"]["criteria"])


def predict_emotion(message: str) -> str:
    agent, questions = load_model()
    result = laya.decide(agent, message, questions=questions)
    return result["emotion_tracker"]["choice"]


def batch_predict_emotions(input_dict: dict) -> dict:
    """Map {key: text_message} to {key: predicted_emotion}."""
    return {
        key: predict_emotion(message)
        for key, message in tqdm(input_dict.items(), desc="Predicting emotions")
    }


@dataclass
class Score:
    emotion: str
    probabilities: dict[str, float]
    confidence: float


def score_batch(texts: list[str]) -> list[Score]:
    """Score W=5 serialized inputs in one call, retaining full probabilities."""
    if not texts:
        return []
    agent, questions = load_model()
    results = agent.decide_batch(texts, questions=questions, return_details=True)
    scores = []
    for result in results:
        detail = result.values["emotion_tracker"]
        probabilities = {k: float(v) for k, v in detail["probabilities"].items()}
        answer_confidence = detail.get("answer_confidence")
        if answer_confidence is None:
            answer_confidence = max(probabilities.values())
        scores.append(
            Score(
                emotion=max(probabilities, key=probabilities.get),
                probabilities=probabilities,
                confidence=float(answer_confidence),
            )
        )
    return scores
