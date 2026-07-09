---
url: https://arxiv.org/abs/2210.13382
title: "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"
type: arxiv_paper
authors: Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
year: 2022 (ICLR 2023 Oral)
accessed: 2026-07-11
quality: 5
relevance: core
---

## Abstract
Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network and create "latent saliency maps" that can help explain predictions in human terms.

## Key Content
- Model: "Othello-GPT", a GPT-variant (~25M params) trained ONLY to predict the next legal move in Othello games, given as a sequence of tile-coordinates. No access to board state, no rules given explicitly.
- Two training regimes: (1) synthetic dataset of games sampled uniformly at random from the full legal game tree (~20M games); (2) "championship" dataset of human-played games.
- Method: probe classifiers trained on internal activations to see if board state (which of 64 tiles is black/white/empty) can be *decoded* from hidden representations.
  - Linear probes on synthetic-trained model: fail to decode board state well.
  - Non-linear probes (2-layer MLP): succeed, revealing the board state IS encoded, just not in a simple linear way in raw coordinates. (Later Neel Nanda's follow-up work — see source 02 — showed a linear encoding does exist if you use a different coordinate system: "mine/yours" relative to current player, rather than absolute black/white.)
- Causal intervention: authors modify internal representation ("flip" the decoded state of a tile) and observe that the model's subsequent move predictions change consistently with the new (intervened) board state — evidence the representation is *used* causally, not epiphenomenal.
- "Latent saliency maps": using the probe + intervention technique, they can visualize which tiles the model's internal state says matter for a given move decision — akin to explainability tooling grounded in the discovered internal world-model.
- Key philosophical framing: authors position the paper directly against the "just surface statistics" hypothesis, want empirical test using a toy, fully-specified environment (game rules) as ground truth to check what's *really* inside the network.
