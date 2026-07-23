---
url: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
type: technical_blog
authors: Anthropic Interpretability Team
year: 2024
accessed: 2026-07-27
quality: 4
relevance: supporting
---

## Key content
Anthropic trained sparse autoencoders (SAEs) on Claude 3 Sonnet's residual stream, extracting
millions of monosemantic (single-concept) features. One famous feature fires specifically on
mentions/depictions of the Golden Gate Bridge. Clamping this feature's activation to ~10x its normal
maximum causes the model to become obsessed with the bridge across unrelated conversations, even to
the point of self-identifying as the bridge — a public demo of the effect was released as "Golden
Gate Claude."

## Relevance to steering-vector literature
This is essentially activation steering / feature clamping performed on SAE-derived directions
rather than difference-in-means directions — showing the same core phenomenon (adding a scaled
direction to the residual stream reliably and specifically shifts model behavior toward a semantic
theme) generalizes from hand-constructed contrastive-pair directions (CAA, refusal direction) to
automatically discovered, more fine-grained SAE features. Also demonstrated clamping a
sycophantic-praise feature causes excessive flattery — directly connecting to CAA's own
demonstrated sycophancy-steering experiments.

## Why it matters for narrative
Provides a vivid, well-known, non-safety-critical example (bridge obsession) that makes the abstract
idea of "you can dial a concept up or down by adding a vector" concrete and memorable before
introducing the higher-stakes refusal-direction case.
