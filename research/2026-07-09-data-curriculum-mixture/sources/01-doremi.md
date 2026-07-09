---
url: https://arxiv.org/abs/2305.10429
title: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
type: arxiv_paper
authors: Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, et al. (Google/Stanford)
year: 2023
accessed: 2026-07-09
quality: 5
relevance: core
---

## Abstract

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.

## Key Points
- 核心机制: 先训练一个小型 proxy 模型，用 Group Distributionally Robust Optimization (Group DRO) 在各个域上做 minimax 博弈——不断上调"损失最差"的域的权重，逼迫模型在所有域上都表现均衡，而不是被大域主导。
- 用小 proxy 模型（280M）得到的域权重，可以直接迁移到大 30 倍的模型（8B）上，说明"最优配比"这个信号在不同规模上有一定可迁移性（类似 scaling law 的思路，但用在数据配比上）。
- 结果：即使某个域被"降权"，该域自身的 perplexity 依然改善——说明均衡配比不是零和博弈，反而整体收益提升。
- 2.6x 训练步数即可达到 baseline（默认配比）的准确率，说明配比优化本质上是"用数据效率换算力效率"。
