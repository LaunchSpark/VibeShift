"""VibeShift evaluation project (CSC 4240/5240).

A research tree that sits alongside the live production system (``src/vibeshift``,
``dashboard``). It curates a 300-message dataset, collects dual human annotations,
runs several emotion-classification baselines, trains a stacking head, and scores
everything with a single evaluation script.

Two files are the contracts that connect the four teammates:

* ``evaluation/data/messages.jsonl`` — the dataset (owned by curation/annotation)
* ``evaluation/predictions/{model}.jsonl`` — one file per model (shared shape)

See ``evaluation/common/schema.md`` for the frozen schemas.
"""
