---
url: https://arxiv.org/abs/2310.01405
title: "Representation Engineering: A Top-Down Approach to AI Transparency"
type: arxiv_paper
authors: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
year: 2023
accessed: 2026-07-27
quality: 5
relevance: core
---

## Abstract
Identifies and characterizes Representation Engineering (RepE): an approach to AI transparency
drawing on cognitive neuroscience that treats population-level representations — not individual
neurons or circuits — as the fundamental unit of analysis. RepE gives methods for both monitoring
and manipulating high-level cognitive phenomena (honesty, harmlessness, power-seeking, emotion, and
more) in LLMs, offering "simple yet effective" solutions relative to bottom-up mechanistic
interpretability, and demonstrates safety-relevant applications.

## Framing: top-down vs bottom-up interpretability
- **Bottom-up (mechanistic interpretability)**: start from individual neurons/circuits, try to build
  up understanding of the whole model — analogous to understanding a brain neuron by neuron.
  Precise but extremely slow to scale to frontier models.
- **Top-down (RepE)**: start from population-level directions/subspaces that correspond to
  high-level concepts (honesty, danger, emotion) borrowed from how cognitive neuroscience studies
  human brains via fMRI-style population activity, not single-neuron recordings. Trade some
  mechanistic precision for practical, scalable control.

## Two core techniques
1. **Reading**: linear probes / PCA-style directions extracted from contrastive stimuli reveal
   whether a concept (e.g. "the model believes this statement is true") is currently active in the
   residual stream — used for lie/hallucination detection.
2. **Control**: once a direction for a concept is found, adding or subtracting it (the same
   activation-addition mechanism as CAA) shifts the model's behavior along that concept axis at
   inference time.

## Demonstrated safety-relevant applications
Honesty (detecting/inducing lying), harmlessness/refusal, power-seeking tendencies, emotional tone,
bias, and memorization — showing the same linear-direction machinery generalizes across many
behavioral axes, not just refusal.

## Significance for the steering-vector literature
RepE is the paper that named and systematized the family of techniques that CAA (Panickssery et al.)
and the refusal-direction paper (Arditi et al.) are specific, sharpened instances of. It established
that "compute a direction from a small contrastive dataset, then intervene linearly on activations"
is a general-purpose recipe rather than a one-off trick for any single behavior.
