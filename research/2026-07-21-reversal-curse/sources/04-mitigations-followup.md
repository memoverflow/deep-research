---
url: multiple (aggregated)
title: "Reversal Curse 缓解方法综述（Reverse Training / BICO / Entity-Preserving Reversal / Bidirectional Model Editing）"
type: aggregated_notes
accessed: 2026-07-21
quality: 4
relevance: supporting
---

## Reverse Training (data augmentation with reversed sequences)
一些工作（如 Lin et al. 系列 follow-up）提出：在训练数据中显式加入"反转"版本的句子/实体片段，让模型在训练阶段就见过两种方向的关联。这类似于"给模型看正着走和倒着走的路"。

- Entity-Preserving Reversal: 通过反转操作 + whole-entity masking 策略实现双向训练，缓解 reversal curse。反转时保持实体（entity）作为整体不被拆散（比如不会把"Tom Cruise"拆成"Cruise Tom"，而是保持实体完整、调整实体间的相对顺序）。
- ETH Zurich 的一篇学位论文 (Interpreting the Reversal Curse of LLMs) 复现并验证：Reverse training 能在合成和真实世界评测上显著缓解 Reversal Curse，且不损害正向性能。

## Bidirectional Model Editing (Untying the Reversal Curse)
论文 "Untying the Reversal Curse via Bidirectional Language Model Editing" 提出：不通过重新训练，而是通过知识编辑 (model editing) 的方式，在编辑一个事实时同时编辑其反向形式，从而保证模型在两个方向上都保持一致。这类方法被后续的 HalluEditBench (ICLR 2025) 一类基准用来评估知识编辑方法对幻觉的实际修正效果。

## Original Paper's Negative Result
需要强调：Berglund et al. (2023) 原论文明确测试过朴素的数据增强不能缓解 Reversal Curse——即简单地把训练数据集"多加一份"不会起作用，缓解方法必须结构性地针对方向不对称设计（比如显式反转、双向 attention 目标、或者事后编辑）。

## Practical Implication for RAG/Knowledge Systems
这类研究直接影响工程实践：如果知识库/微调数据只以一种方向陈述事实（"X 的创始人是 Y"），模型在被反向询问（"Y 创立了什么"）时可靠性会显著下降。这是 RAG 系统设计、SFT 数据构造、知识注入时需要考虑的实际问题——数据构造者需要主动生成正反两个方向的陈述。

## Relevance
提供缓解方案维度的素材，用于文章结尾"这意味着什么"部分，讨论工程启示。
