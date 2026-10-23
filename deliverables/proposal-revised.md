# VibeShift: Emotion Analytics and Multi-Model Benchmarking for Discord Communities

## Project Proposal (Revised) — CSC 4240/5240: Artificial Intelligence

**Team members:** Lucas Starkey, Trigg Lampkins, Trey Gannod, Braylon Trail, Isaiah Chastain

## Motivation

Community managers need to understand how conversations feel as well as how active they are. Message counts and member activity alone cannot reveal whether discussions are becoming more welcoming, confused, or hostile. Written messages lack vocal tone and facial expressions, and Discord messages often depend on informal slang, short replies, or the surrounding thread. That makes their emotional meaning difficult to interpret at scale.

VibeShift aims to make patterns in expressed emotion visible over time. It will help community managers recognize changes in conversation tone and identify discussions that may deserve closer attention. The system will analyze emotions expressed through text, not users' actual internal emotional states.

## Description

VibeShift will use natural language processing (NLP) to analyze Discord messages and present emotion patterns in an interactive dashboard.

**Collection.** A Python bot will collect messages from accessible server text channels, recover available messages missed while offline, and store message metadata and model predictions.

**Labels.** Most everyday Discord messages are conversational or logistical filler, so VibeShift adopts a 7-label classification schema: the six core Ekman emotions (joy, sadness, anger, fear, disgust, surprise) plus *neutral*.

**Laya and the neutral cutoff.** Laya is our own emotion model, which we are retraining on this label schema. An emotion model can still give confident-looking emotion scores to filler, so we will add a configurable confidence cutoff (τ). A message is labeled *neutral* when the model's top label is neutral or its highest emotion score falls below τ. The same rule applies to every model we compare.

**Benchmark.** We will measure Laya's accuracy on Discord text in a head-to-head benchmark against pretrained emotion classifiers (DistilRoBERTa-Emotion and RoBERTa-GoEmotions) and a zero-shot large language model.

**Dashboard.** The dashboard will display aggregate emotion trends over time, allow users to filter results by channel and member, and demonstrate how adjusting τ changes aggregate mood trends. Users can also request an individual prediction with the `!vibe <text>` command.

## Plan of Execution

1. **Collect and store messages.** Configure the bot in a test server with informed participants. Verify live message collection, offline recovery, and duplicate handling. Store message metadata and each message's emotion probability distribution on a local machine. Record when each message is scored so end-to-end processing latency can be measured.
2. **Build the analysis pipeline and dashboard.** Integrate Laya with the neutral cutoff τ, and implement the comparison models. Calculate daily emotion trends for each channel and member, smoothed with a selectable 1-, 7-, 14-, 21-, or 30-day moving average. Add interactive channel, member, and threshold controls to the dashboard.
3. **Evaluate classification and analytics.** Create a held-out dataset of 300 Discord messages: 75 for tuning, 225 kept strictly for final results. Include ordinary conversation and challenging examples such as sarcasm, slang, short replies, and context-dependent messages.

   **Annotation.** Two reviewers will independently label each message with one of the 7 emotion categories while viewing its three preceding messages. They will also flag whether each message is sarcastic and whether it depends on context. We will report inter-annotator agreement (Cohen's κ), establish consensus reference labels, and conduct three core evaluations:

   - **Multi-model benchmark (accuracy, latency, and cost).** Compare Laya (with τ), DistilRoBERTa-Emotion, RoBERTa-GoEmotions, and a zero-shot LLM. Report macro-F1, per-category precision and recall, confusion matrices, inference latency (ms per message), and computational or API cost. Use the annotation flags to report macro-F1 on the sarcastic and context-dependent subsets. Because the dashboard aggregates each message's full probability distribution, not just its top label, we will also score those distributions directly: log-loss and Brier score per message, top-2 accuracy for blended emotions, and Jensen-Shannon divergence between the average predicted distribution and the human label mix.
     - An ablation adding short conversational context (the three preceding messages, and optionally NRC emotion-lexicon features) will measure how much context improves accuracy on ambiguous replies.
     - VADER is retained strictly as a secondary positive/negative/neutral polarity reference.
   - **Neutral cutoff (τ) sensitivity analysis.** Sweep τ from 0.15 to 0.85, choosing each model's cutoff on the tuning split only. Analyze how the choice affects macro-F1 and neutral-class F1, and how far the resulting emotion distributions diverge from the human-labeled distributions (Jensen-Shannon divergence).
   - **System validation.** Measure message capture completeness, offline recovery, and end-to-end processing latency.
4. **Refine and report.** Keep the held-out test data strictly separate from model development and threshold tuning. Use evaluation results and instructor feedback to document errors, model tradeoffs, limitations, team contributions, and reproducible setup instructions. Stretch goals once the core system and comparative evaluation are complete:
   - fine-tuning Laya, or the best-performing emotion transformer, on additional labeled Discord messages
   - exploring whether members' participation is associated with later changes in conversation tone

## Expected Outcome

VibeShift will deliver:

- a working Discord bot
- an interactive emotion analytics dashboard
- a comparative benchmark of Laya, pretrained emotion transformers, and a zero-shot LLM on informal community text

The project will show how neutral calibration affects the emotion trends community managers see. It will also measure the tradeoffs between accuracy, latency, and cost in real-world community monitoring.

## Implementation Resources

- **Laya:** our model, `LaunchSpark/laya-emotion-tracker` on Hugging Face, retrained on the 7-label schema, plus its Colab training and testing notebooks.
- **Pretrained emotion transformers (Hugging Face):** `j-hartmann/emotion-english-distilroberta-base` (the six Ekman emotions plus neutral) and `SamLowe/roberta-base-go_emotions` (a Reddit-trained classifier with 28 labels, mapped to the 7 categories).
- **Zero-shot LLM baseline:** an instruction-tuned LLM via API or local inference (e.g., Gemini Flash or Llama 3).
- **Lexical resources:** the NRC Emotion Lexicon for optional context features, and VADER as the polarity reference.
- **System stack:** discord.py, PocketBase, and a React dashboard.
