---
url: https://arxiv.org/pdf/2507.21513
title: "What Does it Mean for a Neural Network to Learn a 'World Model'?"
type: arxiv_paper
authors: Kenneth Li, Fernanda Viégas, Martin Wattenberg
year: 2025
accessed: 2026-07-11
quality: 5
relevance: core
---

## Key Content
- Meta-level paper by the same Othello-GPT author, trying to formally DEFINE "world model" so debates aren't talking past each other.
- Observes that the debate is stuck because key terms ("world model," "understanding," "common sense") are used informally, sometimes to mean totally different things (cf. LeCun's very different, much richer notion of "world model" vs. what Othello-GPT demonstrates).
- Proposed formal criteria: a network N has learned/uses a "world model" M of the true world/process W if there exist:
  1. An "abstraction" function φ that maps world states W to simplified model states M (e.g., real living room → 2D floorplan with robot position).
  2. Sensing function α mapping W to network input X.
  3. Computation f (the network) that can be FACTORED through M: i.e., f = h∘g where g: X→Z decodes an internal representation of M-like state, and h: Z→A produces the actual output/action — meaning the network's computation genuinely routes through something isomorphic to M, not just correlates with it.
  4. Non-triviality checks — to rule out cases where "world model" claims are vacuous (e.g., trivially true for any sufficiently expressive network, or where the "state space" is just the input itself restated).
- Explicitly cites both LeCun (2022, "informal, richer notion involving action/prediction of future states") and Bender et al. (2021, "stochastic parrot," skeptical baseline) as the two poles the definition needs to mediate between.
- Useful contribution: gives interpretability researchers a checklist to test whether a given probing result (linear or nonlinear) constitutes a genuine "world model" finding versus a more modest "the information is somehow present" finding — important because probing accuracy alone (can you decode X from activations) is a much weaker claim than "the network's actual computation factors through and depends on X."
