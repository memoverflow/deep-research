---
url: https://arxiv.org/abs/2203.15556
title: "Training Compute-Optimal Large Language Models"
type: arxiv_paper
authors: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. (DeepMind)
year: 2022
accessed: 2026-07-08
quality: 5
relevance: core
---

## Abstract
We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4x more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. Chinchilla reaches SOTA average accuracy of 67.5% on MMLU, >7% improvement over Gopher.

## Three Approaches
1. **Approach 1 (minimum over training curves):** vary model size, train, look at loss along whole training curve; fit power law. Result: N_opt ∝ C^0.50, D_opt ∝ C^0.50.
2. **Approach 2 (IsoFLOP profiles):** fix 9 different training FLOP budgets (6e18 to 3e21), vary model size, use cosine schedule length matched to token count. Fit a parabola to each IsoFLOP curve (loss vs log parameter count) to find the minimum. Then fit power law between FLOPs and optimal N/D. Result: a=0.49, b=0.51.
3. **Approach 3 (parametric loss function):** model all final losses as
   L̂(N,D) = E + A/N^α + B/D^β
   - E: irreducible entropy of natural text (ideal generative process loss)
   - A/N^α: loss from finite model capacity (perfectly trained N-param transformer still underperforms ideal process)
   - B/D^β: loss from finite optimization steps / limited data seen
   Fit (A,B,E,α,β) by minimizing Huber loss (δ=1e-3) between predicted and observed log loss via L-BFGS.
   Efficient frontier (via FLOPs(N,D)≈6ND, Kaplan et al. 2020):
   N_opt(C) = G(C/6)^a, D_opt(C) = G^-1(C/6)^b
   where G = (αA/βB)^(1/(α+β)), a = β/(α+β), b = α/(α+β)
   Result: a=0.46, b=0.54. Projected optimal model size at Gopher's FLOP budget: 40B params (they trained 70B Chinchilla, close).

## Table 2 comparison
| Approach | a (N∝C^a) | b (D∝C^b) |
|---|---|---|
| 1. Min over training curves | 0.50 (0.488–0.502) | 0.50 (0.501–0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462–0.534) | 0.51 (0.483–0.529) |
| 3. Parametric loss fit | 0.46 (0.454–0.455) | 0.54 (0.542–0.543) |
| Kaplan et al. 2020 (prior work) | 0.73 | 0.27 |

All three Chinchilla approaches agree: N and D should scale roughly equally with compute (a≈b≈0.5). This sharply contradicts Kaplan et al. (2020), which said model size should grow much faster than data (a=0.73, b=0.27) — implying earlier LLMs (GPT-3, MT-NLG, Gopher) were far too big relative to the data they were trained on, i.e. "undertrained."

## Result: token/parameter ratio
Chinchilla-optimal training implies roughly ~20 tokens of training data per parameter (empirically derived rule of thumb, not an exact universal constant — depends on A,B,α,β fit to DeepMind's specific data/architecture). Chinchilla itself: 70B params, 1.4T tokens (~20:1 ratio), same compute budget as Gopher (280B params, 300B tokens, ~1:1 ratio — badly undertrained by comparison).
