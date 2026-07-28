---
url: https://dev.to/haltonlabs/the-softmax-bottleneck-why-making-llms-bigger-doesnt-always-make-them-smarter-3203
title: "The Softmax Bottleneck: Why Making LLMs Bigger Doesn't Always Make Them Smarter"
type: blog
authors: Vikrant Shukla
year: 2026 (recent)
accessed: 2026-07-31
quality: 3
relevance: supporting
---

## Extracted content (full)
At the final step of a language model, you produce a probability distribution over the vocabulary (30K-200K tokens) by: hidden state h (dim d) × output embedding matrix W (shape V×d) → softmax.

Problem: H·W^T is a rank-d matrix. If the "true" next-token distribution requires higher effective rank than d allows, you can't represent it, no matter how well trained.

Formally: the log-probability matrix across all contexts × all tokens has some ideal rank. If that rank exceeds hidden dim d, softmax layer is a bottleneck — model expresses high-rank function through low-rank projection.

Why it shows up in practice: word "bank" needs very different distributions depending on context ("river bank" vs "financial bank" vs "blood bank" vs "memory bank"). Across all contexts, the full distribution matrix can have very high rank — contexts create nearly linearly independent distributions. A model with hidden dim 4096 can only produce at most 4096 linearly independent output distributions, regardless of total parameter count — the transformer body can be arbitrarily deep/wide, it still funnels down to a d-dim vector at the final step.

Field responses:
1. **Mixture of Softmaxes**: compute K parallel softmax distributions, mix with learned weights → effective rank scales with K·d rather than d. Yang et al.'s own fix; costs ~K× more compute/memory at output layer.
2. **Weight tying** (input=output embedding): can help in some configs because input embeddings encode richer token-token relationships inherited by output projection.
3. **MoE**: different experts activate for different inputs → effective expressiveness scales with number of active experts, partially relaxing rank constraint. Argued as an underappreciated reason MoE models "punch above" their activated-parameter weight.
4. **Wider final layers**: some architectures deliberately widen final transformer blocks / use wider projection head since bottleneck is sharpest at output stage.

Practical signal: if fine-tuning loss plateaus at unreasonably high value, could be architectural bottleneck not data/training issue — "train longer" won't fix it; need bigger d, output factorization, or accept the structural ceiling.
