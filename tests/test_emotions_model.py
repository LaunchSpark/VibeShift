import pytest

from vibeshift.emotions import emotion_labels, predict_emotion

pytestmark = pytest.mark.model


def test_predicts_known_label():
    assert len(emotion_labels()) == 13
    assert predict_emotion("I cannot believe we actually won!") in emotion_labels()
