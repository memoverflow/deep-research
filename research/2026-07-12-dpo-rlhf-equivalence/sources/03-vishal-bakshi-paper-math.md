---
url: https://vishalbakshi.github.io/blog/posts/2024-04-05-dpo/index.html
title: "Paper Math: DPO (Direct Preference Optimization)"
type: blog_technical
authors: Vishal Bakshi
year: 2024-04-05
accessed: 2026-07-10
quality: 4
relevance: supporting
---

Step-by-step walkthrough of DPO paper's appendix math (A.1, A.2, A.4), useful for
cross-verifying the derivation chain against the fireant.github.io source.

## Reward Modelling Phase (Bradley-Terry)
p*(y1≻y2|x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2)))
             = σ(r*(x,y1) − r*(x,y2))   [equivalent sigmoid form]

Reward model r_φ trained via negative log-likelihood (binary classification framing):
L_R(r_φ, D) = −E[(x,yw,yl)~D][log σ(r_φ(x,yw) − r_φ(x,yl))]

r_φ typically initialized from the SFT model with a linear head producing a scalar.

## RL Fine-Tuning Phase
max_πθ E_x~D,y~πθ(y|x)[r_φ(x,y)] − β D_KL[πθ(y|x) || π_ref(y|x)]

Not differentiable (discrete generation) → standard approach uses PPO with reward
r(x,y) = r_φ(x,y) − β(log πθ(y|x) − log π_ref(y|x))

## Author's key intuition on the DPO loss (paraphrased from paper Section 4)
"As the model gets more likely to pick the preferred response, the gradient magnitude behaves
predictably. As the implicit reward for rejected responses increases (model wrongly favoring the
bad response), the gradient also increases to correct it. These two terms contrast each other,
keeping the gradient from vanishing or exploding" — i.e. the σ(·) weighting term in the DPO
gradient acts as an automatic error-proportional learning-rate modulator.

## Confirms the same derivation chain
Cross-checked against fireant.github.io and RLHF book — same result: closed-form optimal policy
→ invert for reward → substitute into Bradley-Terry → partition function Z(x) cancels →
supervised classification loss.
