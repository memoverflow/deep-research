---
url: https://rlhfbook.com/c/08-direct-alignment
title: "Direct-Alignment Algorithms — The RLHF Book, Chapter 8"
type: book_chapter
authors: Nathan Lambert
year: 2024-2026 (ongoing)
accessed: 2026-07-10
quality: 5
relevance: core
---

Authoritative textbook-quality treatment of Direct Alignment Algorithms (DAAs), of which DPO is
the most prominent. Includes full gradient derivation.

## History/context
- DPO released May 2023; after a brief period where the community figured out correct
  hyperparameters (notably: DPO needs surprisingly LOW learning rates), it was adopted widely.
- Technically, SLiC-HF (Sequence Likelihood Calibration) was the first modern direct alignment
  algorithm, released before DPO, but didn't catch on.
- DPO/variants used in: Zephyr-β (Oct 2023, kickstarted DPO adoption), Llama 3 Instruct,
  Tülu 2 and 3 (AI2), Nemotron-4 340B.
- Benefit: DAAs need far less compute and are much easier to implement/experiment with than
  PPO-based RLHF — no reward model training, no online rollout/sampling loop.

## DPO loss (matches other sources, notation aligned)
L_DPO(πθ; π_ref) = −E_(x,yc,yr)~D[log σ(β log(πθ(yc|x)/π_ref(yc|x)) − β log(πθ(yr|x)/π_ref(yr|x)))]

Implicit reward: r(x,y) = β log(π_r(y|x)/π_ref(y|x)) + β log Z(x)

## Full gradient derivation (verbatim structure, verified)

Define u(θ) = β log(π_ref(yl|x)/πθ(yl|x)) − β log(π_ref(yw|x)/πθ(yw|x))

Using sigmoid identities σ'(x) = σ(x)(1−σ(x)) and σ(−x) = 1−σ(x), applying chain rule through
log, σ, u:

∇θ L_DPO = −E[(x,yw,yl)~D][σ(−u)·∇θ(u)]

Substituting back r̂θ(x,y) = β log(πθ(y|x)/π_ref(y|x)):

∇θ L_DPO = −β · E[(x,yw,yl)~D][ σ(r̂θ(x,yl) − r̂θ(x,yw)) · (∇θ log π(yw|x) − ∇θ log π(yl|x)) ]

Interpretation (explicit labeling in source):
- σ(r̂θ(x,yl) − r̂θ(x,yw)) = "higher weight when reward estimate is wrong" — this term is close
  to 1 when the model currently ranks the rejected response ABOVE the chosen one (a mistake),
  and close to 0 once the model already has the correct ranking with a comfortable margin.
- ∇θ log π(yw|x) = "increase likelihood of yw" (chosen)
- −∇θ log π(yl|x) = "decrease likelihood of yl" (rejected)

This is the mechanistic explanation for DPO's self-moderating behavior: examples the model
already handles correctly contribute almost no gradient, while mis-ranked examples get pushed
hard — directly analogous to how logistic regression/cross-entropy weighting works.
