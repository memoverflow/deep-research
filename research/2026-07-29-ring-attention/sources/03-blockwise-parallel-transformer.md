---
url: https://arxiv.org/abs/2305.19370
title: "Blockwise Parallel Transformer for Large Context Models"
type: arxiv_paper
authors: Hao Liu, Pieter Abbeel
year: 2023
accessed: 2026-07-29
quality: 5
relevance: supporting
---

Ring Attention 的前置工作。提出 Blockwise Parallel Transformer (BPT),核心思路是把自注意力与前馈网络(FFN)都改为分块(blockwise)计算,融合两者以最小化内存开销。相比只做 blockwise attention 的早期方法(如 memory-efficient attention, Rabe & Staats 2021),BPT 进一步把 FFN 激活的峰值内存从 8bsh 降到 2bsh,是 Ring Attention 论文里"内存效率对比表"的关键基线之一。BPT 后续被 Ring Attention 直接采用为内循环的计算单元。
