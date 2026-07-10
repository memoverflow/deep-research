---
url: https://fireant.github.io/ml/principles/2026/02/11/dpo.html
title: "DPO: From Bradley-Terry to Direct Preference Optimization"
type: blog_technical
authors: Fireant
year: 2026-02-11
accessed: 2026-07-10
quality: 4
relevance: core
---

Fourth post in "ML Principles for Practitioners" series. Walks through the full DPO derivation
in a pedagogically clean way, building from three prior posts (Bradley-Terry, MLE, KL divergence).

## Key content extracted

**RLHF objective and rearrangement into a KL divergence:**
max_π E[r(x,y) − β log(π(y|x)/π_ref(y|x))]

Dividing by β and flipping to min, then absorbing r(x,y)/β into the log via
(1/β)r(x,y) = log exp(r(x,y)/β), we get a ratio whose denominator
π_ref(y|x)·exp(r(x,y)/β) is not a valid distribution (doesn't sum to 1). Introduce partition
function Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) to normalize it. After adding/subtracting log Z(x)
(constant w.r.t. π), the objective becomes:

min_π KL(π(y|x) || (1/Z(x)) π_ref(y|x) exp(r(x,y)/β)) − log Z(x)

Since KL ≥ 0 and equals 0 only when the two distributions match, the optimal π* is exactly the
Boltzmann/energy-based reweighting of π_ref by exp(r/β), normalized by Z(x). Physical analogy:
same functional form as Boltzmann distribution in statistical mechanics — probability of a state
∝ base rate × exp(energy/temperature).

**Why partition function cancels:** Bradley-Terry only compares reward *differences* between two
candidate responses for the same prompt x. Since Z(x) depends only on x (not y), it appears
identically in both terms of the subtraction r(x,y1) − r(x,y2) and cancels exactly. This is
the crux of why DPO is possible: an intractable normalization constant that would otherwise
block using the closed-form policy in practice simply disappears when you only need reward
*differences*, not absolute reward values.

## Where DPO struggles (author's honest assessment)
- **Offline-only data**: DPO only learns from responses that exist in the fixed preference
  dataset; it can't explore new completions during training the way PPO/online RL can. If the
  best response isn't similar to anything in the data, DPO can't discover it.
- **Preference/likelihood displacement**: DPO tends to decrease log-probability of BOTH chosen
  and rejected responses, just decreasing rejected more. It's learning relative preference, not
  absolute frequency — this can push probability mass to responses not even in the training set.
- **Sensitivity to noisy labels**: treats every pair equally; can overfit to annotator
  disagreement, whereas RM-based RL smooths over many comparisons.
- Research shows PPO still wins on complex reasoning tasks (math, code) with a strong reward
  model and enough compute — the gap grows as tasks get harder.

## DPO variant landscape (each changes ONE piece of the framework)
- **IPO** (Identity Preference Optimization): changes the preference model — replaces strict BT
  sigmoid assumption with a more robust loss, less prone to overfitting on deterministic/noisy
  preferences.
- **SimPO**: changes how the score is computed — averages log-probs across tokens (length
  normalization) instead of summing, removing length bias and the need for a reference model.
- **KTO** (Kahneman-Tversky Optimization): changes the data format — works with unpaired binary
  (good/bad) feedback instead of pairwise comparisons, inspired by prospect theory.
- **Online DPO**: changes offline→online — generates fresh completions from the current policy
  during training and re-ranks them, recovering some of PPO's exploration benefit.

## Reference chain
Cites: Rafailov et al. NeurIPS 2023 (original DPO); Nathan Lambert's RLHF Book Ch.8/12;
Azar et al. "A General Theoretical Paradigm to Understand Learning from Human Feedback" (IPO
paper, addresses DPO overfitting).
