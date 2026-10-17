import pytest

from vibeshift.emotions import Score, emotion_labels, predict_emotion, score_batch

pytestmark = pytest.mark.model


def test_predicts_known_label():
    assert len(emotion_labels()) == 13
    assert predict_emotion("I cannot believe we actually won!") in emotion_labels()


def test_score_batch_shape():
    scores = score_batch(["I cannot believe we actually won!", "this is so annoying"])
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, Score)
        assert set(s.probabilities) == set(emotion_labels())
        assert len(s.probabilities) == 13
        assert abs(sum(s.probabilities.values()) - 1.0) < 0.02
        assert s.emotion == max(s.probabilities, key=s.probabilities.get)
        assert abs(s.confidence - s.probabilities[s.emotion]) < 1e-3
