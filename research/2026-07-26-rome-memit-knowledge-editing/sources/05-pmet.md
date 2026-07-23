---
url: https://arxiv.org/abs/2308.08742
title: "PMET: Precise Model Editing in a Transformer"
type: arxiv_paper
authors: Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, Jie Yu
year: 2023
accessed: 2026-07-26
quality: 4
relevance: supporting (refinement of ROME/MEMIT)
---

Abstract: Model editing techniques modify a minor proportion of knowledge in Large Language Models (LLMs) at a relatively low cost, which have demonstrated notable success. Existing methods assume Transformer Layer (TL) hidden states are values of key-value memories of the Feed-Forward Network (FFN). They usually optimize the TL hidden states to memorize target knowledge and use it to update the weights of the FFN in LLMs. However, the information flow of TL hidden states comes from three parts: Multi-Head Self-Attention (MHSA), FFN, and residual connections. Existing methods neglect the fact that the TL hidden states contains information not specifically required for FFN. Consequently, the performance of model editing decreases. To achieve more precise model editing, we analyze hidden states of MHSA and FFN, finding that MHSA encodes certain general knowledge extraction patterns. This implies that MHSA weights do not require updating when new knowledge is introduced. Based on above findings, we introduce PMET, which simultaneously optimizes Transformer Component (TC, namely MHSA and FFN) hidden states, while only using the optimized TC hidden states of FFN to precisely update FFN weights. Our experiments demonstrate that PMET exhibits state-of-the-art performance on both the COUNTERFACT and zsRE datasets.

## Key content

PMET identifies a subtle flaw in ROME/MEMIT: they optimize the *residual-stream hidden state at a layer* to encode the new fact and then reverse-engineer the FFN weight update needed to reproduce that hidden state. But the residual-stream hidden state is a sum of MHSA output + FFN output + previous residual — not purely FFN output. Using the whole hidden state as the editing "target value" pollutes the update with attention-derived signal that doesn't belong to the FFN's key-value memory.

PMET's fix: separately optimize the MHSA's and FFN's hidden states, but ONLY use the FFN-specific optimized signal to compute the final FFN weight edit. Ablations show MHSA does encode some generic "knowledge extraction pattern" but doesn't need to be edited for a new fact to stick — reinforcing that facts are stored in FFN, not attention. Achieves state-of-the-art on CounterFact/zsRE at time of publication (late 2023).
