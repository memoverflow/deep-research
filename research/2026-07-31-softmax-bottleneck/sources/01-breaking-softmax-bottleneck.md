---
url: https://arxiv.org/abs/1711.03953
title: "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model"
type: arxiv_paper
authors: Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen
year: 2017/2018 (ICLR 2018 Oral)
accessed: 2026-07-31
quality: 5
relevance: core
---

## Abstract
We formulate language modeling as a matrix factorization problem, and show that the expressiveness of Softmax-based models (including the majority of neural language models) is limited by a Softmax bottleneck. Given that natural language is highly context-dependent, this further implies that in practice Softmax with distributed word embeddings does not have enough capacity to model natural language. We propose a simple and effective method (Mixture of Softmaxes, MoS) to address this issue, improving perplexities on Penn Treebank (47.69) and WikiText-2 (40.68), and outperforming baseline by 5.6 points on 1B Word dataset.

## Core formalization (Section 2, from PDF full text)
- Language modeling reduces to modeling P(x|c) for context c and next token x.
- Define three matrices:
  - H_θ ∈ R^{N×d}: rows are context vectors h_c (N = number of distinct contexts)
  - W_θ ∈ R^{M×d}: rows are word embeddings w_x (M = vocabulary size)
  - A ∈ R^{N×M}: rows are TRUE log-probabilities log P*(x|c)
- Standard softmax model computes logits as H_θ W_θ^T, i.e. it's an inner product between a rank-d context matrix and rank-d word matrix.
- Key theoretical result: the set F(A) = {A + Λ·J | Λ diagonal} defines all logit matrices consistent with the true distribution (row-wise shift doesn't change softmax output because softmax is shift-invariant per row).
- Property 2: for any two matrices in F(A), their ranks differ by at most 1 — so the "true" rank of the log-probability structure is essentially well defined.
- Central claim: this true matrix A generally has rank much higher than d (the hidden/embedding dimension), especially because natural language is highly context-dependent (same word needs very different distributions depending on context). Since H_θ W_θ^T can have rank at most d, standard softmax LMs are fundamentally rank-limited — this is the "Softmax Bottleneck."
- Proposed fix: Mixture of Softmaxes (MoS) — introduce K discrete latent variables / K sets of context vectors and K prior weights π_k(c), then:
  P_θ(x|c) = Σ_k π_k(c) · softmax(h_{c,k}^T w_x)
  This is a convex combination of K softmax distributions, effectively raising the achievable rank to scale with K rather than being capped at d.
- Empirically: MoS learns matrices with much higher rank (larger normalized singular values) than vanilla Softmax and other baselines (e.g. mixture of contexts) on real language data.
- Results: SOTA perplexity improvements on Penn Treebank, WikiText-2, 1B Word dataset, and a dialogue dataset.
