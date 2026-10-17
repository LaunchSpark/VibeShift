"""Emotion classification with the fine-tuned Laya emotion-tracker model."""

import json
from functools import cache
from pathlib import Path

import laya
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MODEL_REPO = "LaunchSpark/laya-emotion-tracker"


@cache
def load_model() -> tuple["laya.Agent", dict]:
    """Download (once, then cached on disk) and load the agent and its schema.

    The question schema ships with the model as questions.json, so it always
    matches what the model was trained on.
    """
    model_dir = snapshot_download(MODEL_REPO)
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
