"""Compare sequential, batched, and threaded emotion prediction throughput.

    uv run python scripts/bench_parallel.py [N ...]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import laya
import torch

from vibeshift.emotions import load_model

SAMPLES = [
    "I can't believe we actually won!",
    "ugh, great, another meeting",
    "I'm so sorry I forgot your birthday",
    "get that spider away from me",
    "wait, what does this error even mean?",
    "I miss you so much",
    "that's honestly disgusting",
    "the meeting is at 3pm",
]


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def main():
    sizes = [int(n) for n in sys.argv[1:]] or [1, 8, 32]
    agent, questions = load_model()
    laya.decide(agent, SAMPLES[0], questions=questions)  # warm-up
    print(f"torch threads: {torch.get_num_threads()}")

    def seq(msgs):
        return [laya.decide(agent, m, questions=questions) for m in msgs]

    def batch(msgs):
        return agent.decide_batch(msgs, questions=questions)

    def threads(msgs, workers=4):
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(lambda m: laya.decide(agent, m, questions=questions), msgs))

    print(f"{'N':>4} {'method':<12} {'total s':>8} {'s/msg':>7} {'speedup':>8}")
    for n in sizes:
        msgs = (SAMPLES * (n // len(SAMPLES) + 1))[:n]
        base, ref = timed(lambda: seq(msgs))
        ref = [r["emotion_tracker"]["choice"] for r in ref]
        for name, fn in [("sequential", seq), ("decide_batch", batch), ("threads x4", threads)]:
            t, out = (base, None) if name == "sequential" else timed(lambda: fn(msgs))
            if out is not None:
                labels = [r["emotion_tracker"]["choice"] for r in out]
                assert labels == ref, f"{name} disagrees with sequential: {labels} vs {ref}"
            print(f"{n:>4} {name:<12} {t:>8.2f} {t / n:>7.2f} {base / t:>7.2f}x")


if __name__ == "__main__":
    main()
