"""The 7 emotion labels — the single source of truth for the whole eval tree.

Every model runner, the NRC mapping, the stacking head, and the evaluation axes
import ``LABELS`` from here. Keep this list and its order fixed: prediction
vectors, confusion matrices, and schema-ordered metrics all rely on it.

The production model's live label list (``vibeshift.emotions.emotion_labels``,
pinned to the ``v2-7-labels`` checkpoint) must agree with this tuple.
``test_labels.py`` asserts that agreement (marked ``model``, so it only runs
when you opt into loading the real model).
"""

LABELS: tuple[str, ...] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "neutral",
)

NEUTRAL = "neutral"

# Index lookups used by the schema-ordered metrics.
INDEX = {label: i for i, label in enumerate(LABELS)}
