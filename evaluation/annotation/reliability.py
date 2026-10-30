"""Inter-annotator reliability (owned by Teammates A & B).

Computes Cohen's κ between ``annotator_a.jsonl`` and ``annotator_b.jsonl`` on the
7-class emotion label and on the ``is_context_dependent`` / ``is_sarcastic``
flags, and lists the message ids where A and B disagree (the adjudication
worklist). The adjudicated decisions go into ``consensus.jsonl`` (Contract 1B),
whose ``soft_probs`` should preserve the pre-adjudication 50/50 split.

    pixi run eval-kappa
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score

from evaluation.common import io

A = io.ANNOTATIONS / "annotator_a.jsonl"
B = io.ANNOTATIONS / "annotator_b.jsonl"


def _load(path):
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found. Teammates A & B: produce independent label files "
            f"(Contract 1B) at {path}, then re-run `pixi run eval-kappa`."
        )
    return {r["id"]: r for r in io.read_jsonl(path)}


def main() -> None:
    a, b = _load(A), _load(B)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("no overlapping ids between annotator_a and annotator_b.")

    for field in ("label", "is_context_dependent", "is_sarcastic"):
        ka = [a[i].get(field) for i in ids]
        kb = [b[i].get(field) for i in ids]
        kappa = cohen_kappa_score(ka, kb)
        agree = sum(x == y for x, y in zip(ka, kb)) / len(ids)
        print(f"{field:22s}  κ={kappa:.3f}  raw agreement={agree:.1%}")

    disagree = [i for i in ids if a[i].get("label") != b[i].get("label")]
    print(f"\n{len(disagree)} label disagreements to adjudicate:")
    for i in disagree:
        print(f"  {i}: A={a[i]['label']!r} B={b[i]['label']!r}")


if __name__ == "__main__":
    main()
