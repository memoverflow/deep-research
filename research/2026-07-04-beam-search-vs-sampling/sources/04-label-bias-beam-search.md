---
url: https://arxiv.org/abs/2005.11009
title: "Investigating Label Bias in Beam Search for Open-ended Text Generation"
type: arxiv_paper
year: 2020
accessed: 2026-07-04
quality: 4
relevance: core
---

Abstract: Beam search in open-ended text generation often produces repetitive/generic text;
sampling methods (top-k, nucleus) are preferred instead. The paper argues label bias — caused
by the locally-normalized probability formulation of standard seq2seq models — is a major
cause of this degenerate behavior. Combining locally-normalized MLE with globally-normalized
sequence-level training reduces label bias with almost no perplexity sacrifice; resulting beam
search outputs become more diverse and meaningful.

Used in article's technical-details section to explain the mechanism of label bias:
local (per-step) softmax normalization can't signal "this whole path is bad," unlike global
normalization over complete sequences.
