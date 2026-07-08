---
url: https://arxiv.org/abs/2210.10760
title: "Scaling Laws for Reward Model Overoptimization"
type: arxiv_paper
authors: Leo Gao, John Schulman, Jacob Hilton
year: 2022
accessed: 2026-07-10
quality: 5
relevance: core
---

Abstract: In reinforcement learning from human feedback, it is common to optimize
against a reward model trained to predict human preferences. Because the reward
model is an imperfect proxy, optimizing its value too much can hinder ground truth
performance, in accordance with Goodhart's law. This effect has been frequently
observed, but not carefully measured due to the expense of collecting human
preference data. In this work, we use a synthetic setup in which a fixed
"gold-standard" reward model plays the role of humans, providing labels used to
train a proxy reward model. We study how the gold reward model score changes as
we optimize against the proxy reward model using either reinforcement learning or
best-of-n sampling. We find that this relationship follows a different functional
form depending on the method of optimization, and that in both cases its
coefficients scale smoothly with the number of reward model parameters. We also
study the effect on this relationship of the size of the reward model dataset,
the number of reward model and policy parameters, and the coefficient of the KL
penalty added to the reward in the reinforcement learning setup.

Key quantitative findings (via Lilian Weng's summary, cross-checked):
- Gold reward as function of KL distance d = sqrt(KL(pi||pi_init)):
  R*_bon(d) = d(alpha_bon - beta_bon * d)  for best-of-n
  R*_RL(d)  = d(alpha_RL - beta_RL * log d) for RL/PPO
- Both curves rise then fall — the RL curve degrades more slowly (log term) than
  the BoN curve (linear term), meaning naive rejection sampling overoptimizes
  faster with search budget than PPO does with training steps, for a matched KL budget.
- Larger policies benefit less from optimization but also overoptimize less.
- More reward-model training data reduces "Goodharting" (higher peak gold reward).
- Adding an explicit KL penalty to the RL objective did NOT help in their main
  experiments — it strictly increased the proxy-gold reward gap in most settings;
  the paper's other main lever is early stopping based on the gold-score curve shape.
- Goodhart's law is explicitly invoked as the mechanism: an imperfect proxy that
  becomes the optimization target eventually stops representing what it's a proxy for.
