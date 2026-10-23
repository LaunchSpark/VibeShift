import pytest

from vibeshift.emotions import Score, emotion_labels, predict_emotion, score_batch

pytestmark = pytest.mark.model


def test_predicts_known_label():
    assert emotion_labels() == ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
    assert predict_emotion("I cannot believe we actually won!") in emotion_labels()


def test_score_batch_shape():
    scores = score_batch(["I cannot believe we actually won!", "this is so annoying"])
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, Score)
        assert set(s.probabilities) == set(emotion_labels())
        assert len(s.probabilities) == 7
        assert abs(sum(s.probabilities.values()) - 1.0) < 0.02
        assert s.emotion == max(s.probabilities, key=s.probabilities.get)
        assert abs(s.confidence - s.probabilities[s.emotion]) < 1e-3


def test_sample_chat_scores():
    from tests.fixtures.chat import parse_chat
    from vibeshift.emotions import emotion_labels, score_batch

    msgs = [m for m in parse_chat() if not m.author.bot]
    scores = score_batch([m.content for m in msgs])
    assert len(scores) == 13
    labels = set(emotion_labels())
    for m, s in zip(msgs, scores):
        print(f"{s.emotion:12s} {s.confidence:.2f} <{m.author.name}> {m.content[:60]}")
        assert len(s.probabilities) == 7
        assert s.emotion in labels
