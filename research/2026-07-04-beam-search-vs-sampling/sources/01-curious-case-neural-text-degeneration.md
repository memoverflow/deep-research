---
url: https://arxiv.org/abs/1904.09751
title: "The Curious Case of Neural Text Degeneration"
type: arxiv_paper
authors: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
year: 2019
accessed: 2026-07-04
quality: 5
relevance: core
---

Abstract: Despite considerable advancements with deep neural language models, the enigma of
neural text degeneration persists when these models are tested as text generators. The
counter-intuitive empirical observation is that even though the use of likelihood as
training objective leads to high quality models for a broad range of language understanding
tasks, using likelihood as a decoding objective leads to text that is bland and strangely
repetitive. This paper reveals surprising distributional differences between human text and
machine text, and proposes Nucleus Sampling (top-p) as a method that samples from the dynamic
"nucleus" of the probability distribution, truncating the unreliable tail while allowing
diversity.

Key takeaway used in article: MAP/beam-search decoding leads to bland, repetitive text; the
fix is a dynamically-sized truncated sampling distribution (top-p), motivating the
divergence between "high likelihood" and "human-like" text.
