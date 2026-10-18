# VibeShift: Emotion Analytics for Discord

**Project Proposal — CSC 4240/5240: Artificial Intelligence**

**Team members:** Lucas Starkey, Trigg Lampkins, Trey Gannod, Braylon Trail, Isaiah Chastain

## Motivation

Community managers need to understand how conversations feel as well as how active they are. Message counts and member activity alone cannot explain whether a discussion is becoming more welcoming, confused, or hostile. Written messages also lack vocal tone and facial expressions, making intent easy to misinterpret. VibeShift aims to make patterns in expressed emotion visible over time, helping managers recognize changes and decide when a conversation deserves closer attention.

## Description

VibeShift will apply existing natural language processing tools to Discord messages and present the results in an interactive dashboard. A Python bot will collect messages from accessible server text channels, recover available messages missed while offline, and use the pretrained Laya emotion-tracker model to assign scores across 13 categories. The dashboard will summarize these predictions over time and support filtering by channel and member. Users will also be able to request an individual prediction with `!vibe <text>`.

We will explore whether particular members’ participation is associated with subsequent changes in conversation tone. These patterns will be presented as exploratory associations, without treating them as proof that a person caused a mood change. Predictions describe the tone expressed in text and may differ from the author’s actual feelings.

## Plan of Execution

1. **Collect and store messages.** Test the bot in a server with informed participants, verify live capture and offline recovery, and store message metadata and model outputs for analysis.
2. **Build the analysis and dashboard.** Integrate emotion scoring, aggregate daily trends, and implement channel and member filters. Explore changes in surrounding conversation tone when a member participates, accounting for message volume and channel differences.
3. **Evaluate the approach.** Have two reviewers independently label a held-out set of test messages, record disagreements, and measure emotion classification with macro-F1 and a confusion matrix. Include slang, sarcasm, and ambiguous replies, and measure processing speed. Separately compare VADER with human positive/negative/neutral labels as a simpler sentiment baseline; its labels will not be treated as equivalent to the 13 emotion categories.
4. **Refine and report.** Use evaluation results and instructor feedback to improve the design. Document limitations, alternatives tried, team contributions, and reproducible setup instructions, then prepare the final report, code, and presentation. Fine-tuning Laya on additional labeled Discord messages is a stretch goal after the core system and evaluation are complete.
