---
url: https://proceedings.neurips.cc/paper/2018/file/9dcb88e0137649590b755372b040afad-Paper.pdf
title: "Sigsoftmax: Reanalysis of the Softmax Bottleneck"
type: conference_paper (NeurIPS 2018)
year: 2018
accessed: 2026-07-31
quality: 4
relevance: supporting
---

## Key content (from search snippets)
- Follow-up analysis to Yang et al. 2018, exploring output activation functions composed of ReLU and sigmoid as alternatives to plain softmax.
- Proposes "sigsoftmax": composed of exp(x) * sigmoid(x) instead of just exp(x), before normalizing — breaks the softmax bottleneck WITHOUT adding extra parameters (unlike Mixture of Softmaxes which multiplies cost by K).
- Experiments on language modeling show sigsoftmax and its mixture variant improve perplexity, competitive with MoS at lower cost.
- Also relevant papers found in the same space:
  - "Mixtape: Breaking the Softmax Bottleneck Efficiently" (NeurIPS 2019) — addresses MoS's memory/time cost while retaining higher rank.
  - "Breaking the Softmax Bottleneck via Learnable Monotonic Pointwise Functions" (Ganea et al., ICML 2019) — Linear-Monotonic-Softmax (LMS), learns pointwise monotonic distortion of logits before final softmax.
  - "Deep Residual Output Layers for Neural Language Generation" (Pappas et al., ICML 2019) — alternative output layer parameterization.

## Synthesis: taxonomy of fixes since 2018
1. Mixture-based: MoS (2018), Mixtape (2019) — combine multiple softmax "modes."
2. Activation-based: Sigsoftmax (2018) — change the nonlinearity itself, no extra params.
3. Logit-transform-based: LMS (2019) — learn a monotonic warping of logits pre-softmax.
4. Architectural/scale-based (modern LLM era): weight tying, wider final layers, MoE experts as de facto multiple softmax "modes" — not designed explicitly to fix the bottleneck but empirically relax it.
5. Fully alternative parameterizations: residual output layers, decoupled input/output embeddings (e.g. "Leviathan" 2026 arXiv paper found in search, decoupling embeddings from a compact generator).
