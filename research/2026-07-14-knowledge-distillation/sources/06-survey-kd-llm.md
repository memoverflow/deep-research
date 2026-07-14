---
url: https://arxiv.org/abs/2402.13116
title: "A Survey on Knowledge Distillation of Large Language Models"
type: arxiv_paper
authors: Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, Tianyi Zhou
year: 2024
accessed: 2026-07-14
quality: 4
relevance: supporting
---

Abstract (from search snippet): In the era of Large Language Models (LLMs), Knowledge
Distillation (KD) emerges as a pivotal methodology for transferring advanced capabilities
from leading proprietary LLMs, such as GPT-4, to their open-source counterparts like LLaMA
and Mistral. Additionally, as open-source LLMs flourish, KD plays a crucial role in both
compressing these models, and facilitating their self-improvement by employing themselves
as teachers.

Key taxonomy used by the survey (from snippets):
- White-box KD: student has access to teacher's internal logits/parameters — allows KL-
  based logit matching, hidden-state matching, attention matching.
- Black-box KD: only teacher's textual outputs (API access) available — used for distilling
  reasoning traces, chain-of-thought, and instruction-following behavior from proprietary
  models (GPT-4, Claude) into open student models. This is the dominant recipe behind many
  "GPT-4-distilled" open models (Alpaca-style self-instruct pipelines, WizardLM, Orca).
- Notes remaining open problems: explaining exactly how/why CoT-distillation transfers
  reasoning ability, and how much data is required for effective instruction distillation.
