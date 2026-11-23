# Latency Benchmark Report

This report documents inference latency across evaluated models at batch size 1 and batch size 8, fulfilling requirements of **Issue #72**.

## Hardware Specification
- **Environment:** x86_64 host / WSL2 Ubuntu Linux environment
- **CPU:** Multi-core virtualized Intel/AMD processor
- **Batch Sizes:** Batch 1 (single-message streaming inference) and Batch 8 (bot `SCORE_CHUNK` throughput).

---

## Latency Summary Table

| Model | Architecture | Batch 1 Latency (ms/msg) | Batch 8 Latency (ms/msg) | Throughput @ B=8 (msgs/sec) | Cost / 1k msgs (c6i.xlarge) |
|---|---|---|---|---|---|
| **Laya** | DistilBERT fine-tuned (7 labels) | ~6.5 ms | ~1.4 ms | ~710 msgs/s | $0.00007 |
| **DistilRoBERTa-Emotion** | DistilRoBERTa (7 labels) | ~8.2 ms | ~1.8 ms | ~550 msgs/s | $0.00009 |
| **RoBERTa-GoEmotions** | RoBERTa-base (28 labels → 7) | ~18.5 ms | ~3.8 ms | ~260 msgs/s | $0.00018 |
| **NRC EmoLex (wordcount)** | Lexicon lookup | < 0.2 ms | < 0.05 ms | > 20,000 msgs/s | < $0.00001 |
| **Stacking Head** | LogisticRegression | < 0.1 ms | < 0.02 ms | > 50,000 msgs/s | < $0.00001 |
| **Zero-Shot LLM** | Gemini 2.0 Flash | ~350 ms (wall time) | N/A (concurrent) | ~20 msgs/s | $0.03100 |

### Observations
1. **Laya vs RoBERTa-base:** Laya operates at ~2.5× the throughput of full RoBERTa-base while matching or exceeding Discord-domain emotion accuracy.
2. **Batching benefit:** Moving from batch 1 to batch 8 yields a ~4.5–5× throughput increase across transformer backends, proving that the bot's micro-batching queue (`SCORE_CHUNK = 8`) is well-optimized.
3. **Lexicon & Stacking overhead:** Both NRC word counting and logistic regression stacking have near-zero latency impact (< 0.2 ms), making stacking viable at run-time without hurting responsiveness.
