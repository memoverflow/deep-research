---
url: https://arxiv.org/abs/2305.18290
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
type: arxiv_paper
authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn
year: 2023 (NeurIPS 2023, v3 updated 2024-07-29)
accessed: 2026-07-10
quality: 5
relevance: core
---

## Abstract
While large-scale unsupervised language models (LMs) learn broad world knowledge and some
reasoning skills, achieving precise control of their behavior is difficult due to the completely
unsupervised nature of their training. Existing methods for gaining such steerability collect
human labels of the relative quality of model generations and fine-tune the unsupervised LM to
align with these preferences, often with reinforcement learning from human feedback (RLHF).
However, RLHF is a complex and often unstable procedure, first fitting a reward model that
reflects the human preferences, and then fine-tuning the large unsupervised LM using
reinforcement learning to maximize this estimated reward without drifting too far from the
original model. In this paper we introduce a new parameterization of the reward model in RLHF
that enables extraction of the corresponding optimal policy in closed form, allowing us to solve
the standard RLHF problem with only a simple classification loss. The resulting algorithm, which
we call Direct Preference Optimization (DPO), is stable, performant, and computationally
lightweight, eliminating the need for sampling from the LM during fine-tuning or performing
significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with
human preferences as well as or better than existing methods. Notably, fine-tuning with DPO
exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves
response quality in summarization and single-turn dialogue while being substantially simpler to
implement and train.

## Key derivation chain (from paper appendix, cross-checked with secondary sources)

**RLHF objective:**
max_π E[r(x,y)] − β D_KL(π(y|x) || π_ref(y|x))

**Step 1 — closed-form optimal policy.** Rearranging the KL term and completing it into a
proper KL divergence against a Boltzmann-reweighted reference distribution:

π*(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x,y)/β)

where Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) is the partition function (intractable to compute
because it sums over all possible completions).

**Step 2 — invert to express reward via policy.** Take log of both sides and rearrange:

r(x,y) = β log[π*(y|x)/π_ref(y|x)] + β log Z(x)

**Step 3 — plug into Bradley-Terry.** BT model: P(y1≻y2|x) = σ(r(x,y1) − r(x,y2)).
Substituting r for both y1 and y2, the β log Z(x) terms are identical and cancel exactly
(this is the key algebraic trick — BT only depends on reward *differences*, and Z(x) depends
only on x, not on y, so it's a constant that cancels in the subtraction).

Result: P(y1≻y2|x) = σ(β log[π*(y1|x)/π_ref(y1|x)] − β log[π*(y2|x)/π_ref(y2|x)])

**Step 4 — MLE loss.** Apply negative log-likelihood over preference dataset:

L_DPO(π_θ; π_ref) = −E_(x,yw,yl)~D [log σ(β log(π_θ(yw|x)/π_ref(yw|x)) − β log(π_θ(yl|x)/π_ref(yl|x)))]

This is the final DPO loss — a binary classification loss computed only from log-probabilities
of the policy model and a frozen reference model, no reward model or RL rollout required.

## DPO gradient (from RLHF book derivation, verified algebra)

∇_θ L_DPO = −β E[σ(r̂_θ(x,yl) − r̂_θ(x,yw)) · (∇_θ log π(yw|x) − ∇_θ log π(yl|x))]

where r̂_θ(x,y) = β log(π_θ(y|x)/π_ref(y|x)) is the "implicit reward."

Interpretation: the σ(...) term is a per-example weight that is LARGE when the model currently
gets the ranking wrong (assigns higher implicit reward to the rejected response than the chosen
one) and SMALL when the model already ranks correctly. This is exactly the self-correcting
weighting behavior that keeps the DPO gradient well-behaved — it automatically focuses learning
effort on mistakes and stops pushing once preferences are satisfied.

## Practical notes from paper
- π_θ initialized from SFT model; π_ref is the frozen SFT model (or the base model)
- No sampling from the policy during training — this is what makes DPO much cheaper than PPO
- β controls how far the policy is allowed to drift from π_ref
- Sensitive to learning rate — community found DPO needs surprisingly low LR (per RLHF book)
