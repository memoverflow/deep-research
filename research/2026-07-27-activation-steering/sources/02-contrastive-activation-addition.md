---
url: https://arxiv.org/abs/2312.06681
title: "Steering Llama 2 via Contrastive Activation Addition"
type: arxiv_paper
authors: Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
year: 2023
accessed: 2026-07-27
quality: 5
relevance: core
---

## Abstract
Introduces Contrastive Activation Addition (CAA): steering vectors are computed by averaging the
difference in residual stream activations between pairs of positive and negative examples of a
target behavior (e.g. factual vs. hallucinatory answers, sycophantic vs. non-sycophantic). At
inference time these vectors are added, with a tunable coefficient, to every token position after
the prompt — giving continuous, dial-like control over the strength of the behavior. Evaluated on
Llama 2 Chat across multiple-choice behavioral datasets and open-ended generation; CAA changes
behavior significantly, works on top of/beyond fine-tuning and system-prompt design, and only
minimally degrades general capabilities.

## Core mechanism
1. Construct a dataset of contrastive pairs — same underlying prompt, differing only in whether the
   completion exhibits behavior A or its opposite (e.g. "I'm not sure, what do you think?" honest
   answer vs. sycophantic agreement with the user's stated wrong belief).
2. Run both completions through the model, record residual stream activations at a chosen layer,
   average the (positive − negative) difference across the whole dataset → one steering vector per
   layer.
3. At generation time, add coefficient × steering_vector to every token's residual stream activation
   at that layer, for all tokens after the user prompt. Positive coefficient pushes toward the
   behavior, negative coefficient pushes away from it.

## Why it matters relative to fine-tuning / prompting
- No gradient updates, no training data beyond a small set of contrastive pairs (tens to low
  hundreds), takes minutes rather than hours/days.
- Effects stack with system prompts and fine-tuning rather than replacing them — CAA is a runtime
  dial that can be applied on top of an already-deployed, already-fine-tuned model.
- Because it operates directly on the geometry of the residual stream, the strength of the effect is
  continuously tunable via the scalar coefficient — unlike prompting, which is binary/discrete.

## Interpretability angle
The paper uses activation-space visualization to show that contrastive pairs for a given behavior
cluster tightly and separate cleanly along the extracted direction — supporting the view that
high-level behavioral concepts are represented approximately linearly in the residual stream, and
that the "difference of means" between contrastive activation clusters recovers a meaningful causal
direction, not just a correlational one.
