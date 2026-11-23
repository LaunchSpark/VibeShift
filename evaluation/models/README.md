# Model Baselines & Cost Model

This directory contains the model baselines evaluated alongside VibeShift's core Laya model:
1. **Laya** (`models/laya/`) — 7-label GoEmotions fine-tuned model (`v2-7-labels`).
2. **DistilRoBERTa-Emotion** (`models/transformers/` with `--model distilroberta`) — `j-hartmann/emotion-english-distilroberta-base`.
3. **RoBERTa-GoEmotions** (`models/transformers/` with `--model goemotions`) — `SamLowe/roberta-base-go_emotions` (28 classes mapped to 7).
4. **Zero-Shot LLM** (`models/llm/`) — Google Gemini (`gemini-2.0-flash`) via structured JSON prompting at temperature 0.
5. **NRC EmoLex** (`models/nrc_lexicon/`) — lexicon word-count and M=25 macro-window vector.
6. **Stacking Head** (`models/stacking/`) — LogisticRegression combining Laya probabilities and NRC macro context.

---

## Cost Model (Issue #67)

### Reference Cloud Instance
- **Instance Type:** AWS `c6i.xlarge` (Compute-optimized, 4 vCPUs, 8 GiB RAM, Intel Xeon 8375C).
- **On-Demand Hourly Rate:** **$0.1700 / hour** (AWS US East / N. Virginia).

### Throughput and Cost Formula

For local inference models (Laya, DistilRoBERTa, RoBERTa-GoEmotions, NRC EmoLex):

$$\text{Throughput (msgs/sec)} = \frac{1}{\text{latency per message (seconds)}} = \frac{1000}{\text{latency\_ms}}$$

$$\text{Throughput (msgs/hour)} = \text{Throughput (msgs/sec)} \times 3600$$

$$\text{Cost per 1,000 messages} = \frac{\text{Hourly Rate}}{\text{Throughput (msgs/hour)}} \times 1000 = \frac{\text{Hourly Rate} \times \text{latency\_ms}}{3.6}$$

$$\text{Cost per message (USD)} = \frac{\text{Hourly Rate} \times \text{latency\_ms}}{3,600,000}$$

For batch size 8 inference, throughput scales near-linearly (~5–7× improvement), lowering cost per message proportionally.

---

## Zero-Shot LLM Decisions & Budget Cap (Issue #66)

- **Selected Model:** `gemini-2.0-flash`
- **Pricing:**
  - Input tokens: **$0.10** per 1,000,000 tokens
  - Output tokens: **$0.40** per 1,000,000 tokens
- **Per-message token usage:** ~150 input tokens (including context window + prompt template) and ~40 output tokens (JSON probability dictionary).
- **Estimated cost per message:** $\approx \$0.000031$ ($0.031 per 1,000 messages).
- **Hard budget cap:** **$5.00** USD (sufficient for over 150,000 message classifications; the benchmark dataset is 300 messages).
