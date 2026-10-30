"""Guard against 7-label drift between this tree and the live production model.

Marked ``model`` because it downloads/loads the real checkpoint, so it is skipped
by the default ``pixi run test``. Run it with ``pixi run test-model``; production
pins the 7-label GoEmotions checkpoint (``v2-7-labels``), so it should pass.
"""

import pytest

from evaluation.common.labels import LABELS


@pytest.mark.model
def test_labels_match_production_model():
    from vibeshift.emotions import emotion_labels

    assert tuple(emotion_labels()) == LABELS, (
        "evaluation LABELS and the live model's labels disagree — update "
        "evaluation/common/labels.py or MODEL_REVISION in vibeshift/emotions.py."
    )
