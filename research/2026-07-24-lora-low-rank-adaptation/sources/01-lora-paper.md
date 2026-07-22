---
url: https://arxiv.org/abs/2106.09685
title: "LoRA: Low-Rank Adaptation of Large Language Models"
type: arxiv_paper
authors: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
year: 2021
accessed: 2026-07-24
quality: 5
relevance: core
---

## Abstract
An important paradigm of NLP consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning becomes less feasible. Using GPT-3 175B as an example — deploying independent fine-tuned instances, each with 175B parameters, is prohibitively expensive. LoRA freezes pre-trained weights and injects trainable rank decomposition matrices into each layer of the Transformer, greatly reducing trainable parameters. Compared to GPT-3 175B fine-tuned with Adam, LoRA reduces trainable parameters by 10,000x and GPU memory by 3x. LoRA performs on-par or better than fine-tuning on RoBERTa, DeBERTa, GPT-2, GPT-3, with no additional inference latency.

## Key Content (extracted from full text via ar5iv)

**Core hypothesis:** Inspired by Li et al. 2018 and Aghajanyan et al. 2020 showing pretrained models reside on a low intrinsic dimension, the authors hypothesize the *change* in weights during adaptation (ΔW) also has low "intrinsic rank." Using GPT-3 175B, a rank as low as r=1 or 2 suffices even though full rank d=12,288.

**Key advantages:**
- Frozen shared model + swappable small LoRA modules for different tasks (efficient task switching)
- No need to compute gradients/optimizer states for most parameters → up to 3x less hardware
- Linear design allows merging trainable matrices with frozen weights at deploy time → **zero additional inference latency**, unlike adapter layers

**Optimal rank experiments (GPT-2 Medium, E2E NLG):**
Rank r=1 through 1024 tested. Validation loss plateaus around r=16, BLEU peaks near r=4. "The optimal rank for GPT-2 Medium is between 4 and 16... similar to that for GPT-3 175B." Relationship between model size and optimal rank called "still an open question."

**Subspace similarity experiment (Section 7):** Comparing A_{r=8} and A_{r=64} learned with same pretrained model via SVD + normalized subspace similarity (Grassmann distance). Finding: "increasing r does not cover a more meaningful subspace" — the top singular directions learned at r=8 are largely already captured within the r=64 solution, meaning most of the "useful" direction space is very low-dimensional; adding more rank mostly adds noise directions, not new task-relevant ones.

**Correlation between ΔW and W (Table 7, 48th layer of GPT-3):**
- ||U^T W_q V^T||_F using top-r directions of ΔW vs. top-r of W vs. a random matrix
- r=4: ΔW-derived directions have Frobenius norm 0.32 with random projection vs 21.67 with ΔW's own directions... (comparing correlation strength)
- ||ΔW_q||_F = 6.91 vs ||W_q||_F = 61.95 — ΔW is roughly 1/9 the magnitude of W
- Amplification factor ≈ 21.5 (6.91/0.32) for r=4: ΔW does NOT just repeat W's top directions (that would be redundant), it "amplifies directions that are not emphasized in W" — task-specific directions that exist but are underweighted in the general pretrained model
- Conclusion: "the low-rank adaptation matrix potentially amplifies the important features for specific downstream tasks that were learned but not emphasized in the general pre-training model"

**Which weight matrices to adapt (Table 5, GPT-3 175B, fixed 18M parameter budget):**
Comparing adapting W_q alone (r=8) vs W_q,W_v together (r=4) vs all four Q,K,V,O together (r=2):
WikiSQL accuracy: 70.4 (Wq) / 70.0 (Wk) / 73.0 (Wv) / 73.2 (Wo) / 71.4 (Wq,Wk) / 73.7 (Wq,Wv) / 73.7 (all four)
→ Spreading a fixed parameter budget across more matrix types (even at lower rank each) tends to help — breadth of adaptation across weight types matters more than concentrated depth in one matrix type.

## Key Figures
- Figure 1 (paper): LoRA reparameterization diagram — pretrained weight W frozen, ΔW = BA with B∈R^{d×r}, A∈R^{r×k}, r << min(d,k). Forward: h = Wx + BAx. B initialized to zero, A initialized with random Gaussian, so ΔW=0 at start of training.
