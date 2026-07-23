---
url: https://arxiv.org/abs/2406.11717
title: "Refusal in Language Models Is Mediated by a Single Direction"
type: arxiv_paper
authors: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
year: 2024
accessed: 2026-07-27
quality: 5
relevance: core
---

## Abstract
Conversational LLMs are fine-tuned for both instruction-following and safety, producing models
that obey benign requests but refuse harmful ones. This paper shows refusal is mediated by a
**one-dimensional subspace** across 13 open-source chat models up to 72B parameters. For each
model there exists a single direction such that:
- erasing this direction from the residual stream prevents refusal on harmful instructions
- adding this direction elicits refusal even on harmless instructions

The authors use this to build a white-box jailbreak that surgically disables refusal with minimal
effect on other capabilities, and mechanistically study how adversarial suffixes suppress
propagation of the refusal direction.

## Method: extracting the refusal direction (difference-in-means)
For each layer l and post-instruction token position i, compute mean activations over harmful
prompts (µ) and harmless prompts (ν) from a small training set:

  µ_i^(l) = mean over harmful prompts of x_i^(l)(t)
  ν_i^(l) = mean over harmless prompts of x_i^(l)(t)
  r_i^(l) = µ_i^(l) − ν_i^(l)

This "difference-in-means" vector is meaningful both in direction (how harmful/harmless activations
differ) and magnitude (distance between the two clusters). Enumerating all (layer, token position)
pairs gives many candidate vectors; the best one is selected on a validation set by checking that
ablating it bypasses refusal and adding it induces refusal while minimally changing other behavior.

## Two interventions
1. **Activation addition**: x'^(l) ← x^(l) + r^(l) — shifts harmless activations toward the harmful
   cluster mean, inducing refusal even on innocuous prompts.
2. **Directional ablation**: x' ← x − r̂r̂ᵗx — projects out the unit direction r̂ from every residual
   stream activation at every layer/token position, so the model can never represent that direction.
   This is applied to ALL layers and token positions (vs. activation addition, at one layer only).

## Weight orthogonalization (turns runtime intervention into a permanent weight edit)
Directional ablation can be baked directly into the weights. For every matrix W_out that writes to
the residual stream (embedding matrix, positional embedding matrix, attention output matrices, MLP
output matrices):

  W'_out ← W_out − r̂r̂ᵗW_out

This is mathematically equivalent to inference-time directional ablation (proven in appendix E) —
the model's weights are edited once, permanently, with no gradient-based optimization and no
examples of harmful completions needed. This rank-one weight edit is the technical basis of what the
community calls "abliteration."

## Evaluation
Uses HarmBench's standardized ASR (attack success rate) evaluation. The weight-orthogonalization
jailbreak ("ORTHO") is competitive with prompt-specific optimization-based jailbreaks like GCG,
despite being a single, general, training-free operation.

## Key implication
"An understanding of model internals can be leveraged to develop practical methods for controlling
model behavior" — and conversely, current safety fine-tuning is brittle: safety behavior that took
significant RLHF effort to install can be erased by a single vector subtraction identified from
~dozens of contrastive prompt pairs.
