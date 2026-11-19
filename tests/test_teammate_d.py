from evaluation.common.labels import LABELS
from evaluation.models.nrc_lexicon import run as nrc


def test_tokenize_removes_urls_mentions_and_custom_emoji():
    assert nrc.tokenize("JOY <@123> <@!456> <@&789> <#42> <:wave:10> <a:dance:11> https://x.test/a") == ["joy"]


def test_nrc_counts_normalize_and_break_ties_in_label_order():
    lexicon = {"glad": frozenset({"joy"}), "upset": frozenset({"anger", "sadness"})}
    probs, top = nrc._prediction(nrc._counts("glad upset", lexicon))
    assert probs["joy"] == 1 / 3
    assert probs["anger"] == probs["sadness"] == 1 / 3
    assert sum(probs.values()) == 1
    assert top == "joy"
    assert list(probs) == list(LABELS)


def test_nrc_no_hits_returns_neutral():
    probs, top = nrc._prediction(nrc._counts("unknown", {}))
    assert top == "neutral"
    assert probs == {label: float(label == "neutral") for label in LABELS}


def test_nrc_roles_use_their_documented_context():
    lexicon = {"joyful": frozenset({"joy"}), "angry": frozenset({"anger"})}
    row = {"content": "angry", "preceding_3": ["joyful"], "preceding_25": ["joyful"]}
    word_probs, _, _ = nrc.predict_wordcount([row], lexicon)[0]
    macro_probs, _, _ = nrc.predict_macro([row], lexicon)[0]
    assert word_probs["joy"] == word_probs["anger"] == 0.5
    assert macro_probs["joy"] == 1.0
    assert macro_probs["anger"] == 0.0


def test_nrc_loads_emotion_associations_only(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text("happy\tjoy\t1\nhappy\tpositive\t1\ncalm\ttrust\t1\n")
    assert nrc.load_lexicon(path) == {"happy": frozenset({"joy"})}


def test_stacking_folds_and_dev_only_search():
    import numpy as np

    from evaluation.models.stacking.train import _folds, _select_hyperparameters

    y = ["joy"] * 5 + ["anger"] * 5
    x = np.asarray([[float(i), float(i % 2)] for i in range(len(y))])
    folds, stratified = _folds(y)
    assert stratified and len(folds) == 5
    c, balanced = _select_hyperparameters(x, y, folds, stratified)
    assert c in (0.01, 0.1, 1.0, 10.0)
    assert isinstance(balanced, bool)


def test_evaluation_includes_per_class_precision_and_recall():
    from evaluation.eval.evaluate import evaluate_model

    gold = {"x": {"label": "joy"}}
    probs = {label: float(label == "joy") for label in LABELS}
    row = {"id": "x", "probs": probs, "raw_top_label": "joy", "latency_ms": 1.0}
    result = evaluate_model([row], gold, 0.0)
    assert result["per_class"]["joy"] == {"precision": 1.0, "recall": 1.0}
    assert result["per_class"]["anger"] == {"precision": 0.0, "recall": 0.0}
