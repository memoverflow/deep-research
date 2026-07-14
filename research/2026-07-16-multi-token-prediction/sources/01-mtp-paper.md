---
url: https://arxiv.org/abs/2404.19737
title: "Better & Faster Large Language Models via Multi-token Prediction"
type: arxiv_paper
authors: Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve
year: 2024
accessed: 2026-07-16
quality: 5
relevance: core
---

Abstract: Large language models such as GPT and Llama are trained with a next-token
prediction loss. In this work, we suggest that training language models to predict
multiple future tokens at once results in higher sample efficiency. More specifically,
at each position in the training corpus, we ask the model to predict the following n
tokens using n independent output heads, operating on top of a shared model trunk.
Considering multi-token prediction as an auxiliary training task, we measure improved
downstream capabilities with no overhead in training time for both code and natural
language models. The method is increasingly useful for larger model sizes, and keeps
its appeal when training for multiple epochs. Gains are especially pronounced on
generative benchmarks like coding, where our models consistently outperform strong
baselines by several percentage points. Our 13B parameter models solves 12% more
problems on HumanEval and 17% more on MBPP than comparable next-token models.
Experiments on small algorithmic tasks demonstrate that multi-token prediction is
favorable for the development of induction heads and algorithmic reasoning
capabilities. As an additional benefit, models trained with 4-token prediction are up
to 3 times faster at inference, even with large batch sizes.

## Key content extracted

- Loss: L_n = -sum_t sum_{i=1}^n log P_theta(x_{t+i} | x_{t:1})
- Architecture: shared trunk f_s + n independent output heads f_h1..f_hn + shared
  unembedding f_u. Only head 1 used at inference (standard next-token), others can
  drive self-speculative decoding.
- Memory-efficient implementation: sequential forward/backward per head reduces peak
  GPU memory from O(nV+d) to O(V+d), no runtime overhead.
- Scale effect: MTP hurts small models (300M) but wins from ~3B+ params; advantage
  grows with scale. 13B model: +12% HumanEval, +17% MBPP.
- Byte-level extreme case: 8-byte prediction model solves 67% more MBPP pass@1 than
  next-byte baseline — shows MTP is critical when next-step task carries little
  information.
- Self-speculative decoding: up to 3x inference speedup on code (2.5/3 accepted
  tokens average), 3x on text, 6.4x on 8-byte model.
- Section 5 "Why does it work?":
  - 5.1 Lookahead reinforces choice points: not all tokens equally important;
    "choice points" determine downstream structure. MTP implicitly assigns weight
    n(n+1)/2 to choice-point tokens (via correlated future tokens) vs weight n to
    inconsequential tokens.
  - 5.2 Information-theoretic argument: teacher forcing encourages short-term focus,
    ignoring longer-term structure; MTP loss decomposition shows increased weight on
    mutual information between adjacent future tokens.
- Section 4.1 Induction capability: MTP models form induction heads earlier/more
  reliably, especially for smaller models / lower quality data. Once induction
  capability forms via other means (e.g. better data mix), MTP's advantage on this
  task disappears — suggesting the mechanism is specifically about accelerating
  formation of induction-head-like circuits.
- Section 4.2 Algorithmic reasoning: tested on polynomial ring arithmetic tasks,
  MTP models show stronger in-context algorithmic reasoning.
- Related work: connects to ProphetNet (Qi et al., 2020) which proposed multi-token
  prediction with n-fold residual stream replication (not compute-matched, unlike
  this paper's compute-matched design).

## Key Figures
- Figure 1: architecture overview, shared trunk + n dedicated output heads, only
  next-token head used at inference by default.
- Figure 9: illustration of implicit weighting — "hard" choice point token receives
  higher loss weight via its correlation with subsequent easy-to-predict tokens.
