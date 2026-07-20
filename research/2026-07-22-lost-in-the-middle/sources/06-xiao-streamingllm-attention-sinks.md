---
url: https://arxiv.org/abs/2309.17453
title: "Efficient Streaming Language Models with Attention Sinks"
type: arxiv_paper
authors: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
year: 2023
accessed: 2026-07-22
quality: 5
relevance: supporting
---

Introduces and empirically documents "attention sinks": a small number of initial tokens
(often just the very first token, or the BOS token) absorb a disproportionately large share
of attention weight, across almost all heads and layers, largely regardless of their
semantic content. Removing/evicting these initial tokens (e.g. in a naive sliding-window KV
cache for streaming generation) causes catastrophic perplexity blowup, even though those
tokens carry little "meaning" — because so much of the softmax normalization mass is anchored
on them.

Practical fix (StreamingLLM): always keep the first few tokens ("sink tokens") in the KV
cache permanently, combined with a sliding window for the rest, achieving stable perplexity
for sequences of essentially unbounded length with up to 22.2x speedup vs. naive
recomputation baselines.

Relevance: attention sinks are the "primacy" half of the lost-in-the-middle U-shape made
concrete and exploitable. Later theoretical work (Chowdhury 2026, Wu et al. 2025) argues sinks
are not just a learned trick models pick up, but are geometrically FORCED to occur, at least
in part, by causal masking + depth, even before training. This paper is the empirical/
engineering side of that story — showing sinks exist, matter, and can be engineered around.
