"""Load the dataset, read/write prediction files, and validate both schemas.

If the real ``messages.jsonl`` / ``consensus.jsonl`` don't exist yet (Teammates A
and B are still curating and annotating), the loaders fall back to the committed
10-row sample fixtures and print a one-line warning. That lets Teammates C and D
run the whole pipeline today.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .labels import LABELS, NEUTRAL

ROOT = Path(__file__).resolve().parents[1]  # evaluation/
DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
LABEL_FILES = ANNOTATIONS / "labels"  # per-labeler exports: {labeler}_w{W}.jsonl
PREDICTIONS = ROOT / "predictions"

MESSAGES = DATA / "messages.jsonl"
SAMPLE_MESSAGES = DATA / "sample_messages.jsonl"
CONSENSUS = ANNOTATIONS / "consensus.jsonl"
SAMPLE_CONSENSUS = ANNOTATIONS / "sample_consensus.jsonl"

_PROB_TOL = 1e-3


# --------------------------------------------------------------------------- #
# jsonl helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _warn(msg: str) -> None:
    print(f"\033[33m[evaluation] {msg}\033[0m", file=sys.stderr)


# --------------------------------------------------------------------------- #
# dataset loaders (with sample fallback)
# --------------------------------------------------------------------------- #
def load_messages(path: Path | None = None) -> list[dict]:
    """The 300 messages, or the 10-row sample if the real file is missing."""
    if path is not None:
        return read_jsonl(path)
    if MESSAGES.exists():
        return read_jsonl(MESSAGES)
    _warn(
        f"{MESSAGES.name} not found — using {SAMPLE_MESSAGES.name} (10 rows). "
        "Replace once A & B export the real 300 messages."
    )
    return read_jsonl(SAMPLE_MESSAGES)


def consensus_path(window: int) -> Path:
    """Gold labels made at window size ``window`` (built by `pixi run label-consensus`)."""
    return ANNOTATIONS / f"consensus_w{window}.jsonl"


def overrides_path(window: int) -> Path:
    """Adjudicated tie-breaks for one window size: ``{"<id>": "<label>"}``."""
    return ANNOTATIONS / f"overrides_w{window}.json"


def consensus_windows() -> list[int]:
    """Window sizes that have a consensus file, ascending."""
    sizes = []
    for p in ANNOTATIONS.glob("consensus_w*.jsonl"):
        suffix = p.stem.removeprefix("consensus_w")
        if suffix.isdigit():
            sizes.append(int(suffix))
    return sorted(sizes)


def load_label_files(window: int, directory: Path | None = None) -> dict[str, dict[str, str]]:
    """Every labeler's labels at one window size: labeler -> {message id -> label}."""
    directory = LABEL_FILES if directory is None else directory
    out = {}
    for p in sorted(directory.glob(f"*_w{window}.jsonl")):
        labeler = p.stem.removesuffix(f"_w{window}")
        out[labeler] = {r["id"]: r["label"] for r in read_jsonl(p)}
    return out


def load_consensus(path: Path | None = None, window: int | None = None) -> dict[str, dict]:
    """Gold rows keyed by message id.

    ``window`` picks ``consensus_w{window}.jsonl``. Without it, the single
    consensus file present is used; several present is an error (pass
    ``--window``); none falls back to the legacy file, then the sample fixture.
    """
    if path is None and window is not None:
        path = consensus_path(window)
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found — run `pixi run label-consensus --window {window}` first."
            )
    if path is None:
        windows = consensus_windows()
        if len(windows) == 1:
            path = consensus_path(windows[0])
        elif len(windows) > 1:
            raise SystemExit(
                f"gold labels exist for window sizes {windows}; pass --window to choose one."
            )
    if path is not None:
        rows = read_jsonl(path)
    elif CONSENSUS.exists():
        rows = read_jsonl(CONSENSUS)
    else:
        _warn(
            f"no consensus file found — using {SAMPLE_CONSENSUS.name} (10 rows). "
            "Build real gold with `pixi run label-consensus --window W`."
        )
        rows = read_jsonl(SAMPLE_CONSENSUS)
    return {row["id"]: row for row in rows}


def context_sensitive_ids() -> set[str] | None:
    """Messages whose gold label changes between the smallest and largest window size.

    ``None`` until gold exists for at least two window sizes. This replaces the old
    hand-set ``is_context_dependent`` flag: context mattered if more of it changed
    the label.
    """
    windows = consensus_windows()
    if len(windows) < 2:
        return None
    small = load_consensus(window=windows[0])
    large = load_consensus(window=windows[-1])
    return {i for i in small.keys() & large.keys() if small[i]["label"] != large[i]["label"]}


def using_sample() -> bool:
    """True when the real dataset is not present yet (for banners in runners)."""
    return not MESSAGES.exists()


def split_ids(messages: list[dict], split: str) -> set[str]:
    """Message ids in the requested split ('dev' or 'test')."""
    return {m["id"] for m in messages if m.get("split") == split}


# --------------------------------------------------------------------------- #
# probability vectors
# --------------------------------------------------------------------------- #
def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Project any label->prob mapping onto LABELS (fill 0, renormalize).

    Tolerates a model whose native label set differs from ours (e.g. the live
    13-label checkpoint before the 7-label one is pushed, or RoBERTa's 28
    GoEmotions classes after they're mapped down).
    """
    extra = set(probs) - set(LABELS)
    if extra:
        _warn(f"dropping non-schema labels from probs: {sorted(extra)}")
    vec = {label: float(probs.get(label, 0.0)) for label in LABELS}
    total = sum(vec.values())
    if total <= 0:
        # Degenerate input — fall back to uniform so downstream math stays finite.
        return {label: 1.0 / len(LABELS) for label in LABELS}
    return {label: v / total for label, v in vec.items()}


def probs_vector(probs: dict[str, float]) -> list[float]:
    """probs as a list in canonical LABELS order."""
    return [float(probs[label]) for label in LABELS]


# --------------------------------------------------------------------------- #
# prediction files (Contract 2)
# --------------------------------------------------------------------------- #
def prediction_path(model_name: str) -> Path:
    return PREDICTIONS / f"{model_name}.jsonl"


def validate_prediction_row(row: dict) -> None:
    missing = {"id", "model", "probs", "raw_top_label", "latency_ms"} - set(row)
    if missing:
        raise ValueError(f"prediction row {row.get('id')!r} missing keys: {sorted(missing)}")
    probs = row["probs"]
    if set(probs) != set(LABELS):
        raise ValueError(
            f"prediction row {row['id']!r} probs keys {sorted(probs)} != LABELS {list(LABELS)}"
        )
    total = sum(probs.values())
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(f"prediction row {row['id']!r} probs sum to {total:.4f}, not 1.0")


def write_predictions(model_name: str, rows: list[dict]) -> Path:
    """Validate every row against Contract 2, then write predictions/{model}.jsonl."""
    for row in rows:
        row.setdefault("model", model_name)
        row.setdefault("cost_usd", 0.0)
        validate_prediction_row(row)
    path = prediction_path(model_name)
    write_jsonl(path, rows)
    print(f"[evaluation] wrote {len(rows)} predictions -> {path.relative_to(ROOT.parent)}")
    return path


def load_predictions(model_name: str) -> list[dict]:
    return read_jsonl(prediction_path(model_name))


def list_prediction_models() -> list[str]:
    """Model names with a committed predictions file, sorted."""
    return sorted(p.stem for p in PREDICTIONS.glob("*.jsonl"))
