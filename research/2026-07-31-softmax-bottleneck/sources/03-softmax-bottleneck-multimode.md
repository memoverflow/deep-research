---
url: https://aclanthology.org/2022.acl-long.554/
title: "Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions"
type: acl_paper
authors: Haw-Shiuan Chang, Andrew McCallum
year: 2022
accessed: 2026-07-31
quality: 4
relevance: supporting
---

## Key content (from search snippets + emergentmind summary)
- Neural LMs like GPT-2 estimate next-word probability via softmax over a single hidden state dotted with word embeddings.
- Because the output is computed from ONE hidden vector per context, softmax can naturally only represent a distribution with a single "mode" of high-probability regions in embedding space — like a single Gaussian bump.
- But real language is often genuinely multi-modal: e.g. after "I want to eat a ___", plausible next words (fruit names, meal names, "sandwich", "pizza") occupy multiple, semantically disjoint clusters in embedding space, not one contiguous region.
- A single linear projection + softmax cannot represent "high probability on cluster A AND cluster B but low probability on the space between them" — because the dot-product-with-single-vector geometry induces one contiguous high-probability region (a half-space intersection / softmax "cone").
- This is a geometric/qualitative companion to Yang et al.'s rank-based bottleneck: even ignoring rank, single-hidden-state softmax has a structural inability to place mass on multiple disjoint locations in the vocabulary embedding space simultaneously.
- Relates to solutions like Mixture of Softmaxes (each softmax component can cover one mode; the mixture covers multiple).
