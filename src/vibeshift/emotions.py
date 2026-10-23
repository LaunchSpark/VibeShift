"""Emotion classification with the fine-tuned Laya emotion-tracker model."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import laya
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MODEL_REPO = "LaunchSpark/laya-emotion-tracker"
# The 7-label checkpoint (joy, sadness, anger, fear, disgust, surprise, neutral), fine-tuned
# on GoEmotions by notebooks/finetune_laya_emotions.py. The 13-label one is "v1-13-labels".
MODEL_REVISION = "v2-7-labels"
# Stored on every scored message, so a message's scores can be traced to the checkpoint.
MODEL_ID = f"{MODEL_REPO}@{MODEL_REVISION}"


@cache
def load_model() -> tuple["laya.Agent", dict]:
    """Download (once, then cached on disk) and load the agent and its schema.

    The question schema ships with the model as questions.json, so it always
    matches what the model was trained on.
    """
    model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION)
    agent = laya.Agent(model_dir)
    questions = json.loads((Path(model_dir) / "questions.json").read_text())
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
    """Score texts in one batched call, keeping the full per-label distribution."""
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
