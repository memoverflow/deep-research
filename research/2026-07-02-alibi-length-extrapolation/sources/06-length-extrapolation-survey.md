---
url: https://arxiv.org/abs/2312.17044
title: "Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding"
type: arxiv_paper
year: 2023
accessed: 2026-07-02
quality: 4
relevance: supporting
---

摘要要点：所有基于 Transformer 的模型（包括 LLM）都存在预设的长度上限，很难从训练时见过的短序列泛化到推理时的长序列，即"长度外推"问题。这严重限制了 LLM 在需要长输入的场景（如长文档、多轮对话历史）中的应用。该综述指出：位置编码(PE)被认为是决定长度外推能力的核心因素，因此从 PE 的视角系统梳理了外推方法：从"可外推 PE"（如 ALiBi、RoPE 变体）开始讲起，到基于位置索引操作的方法（插值类：PI、NTK-aware、YaRN）。

论文还指出一个值得警惕的观察：现实场景中，是否真的存在"外推"仍不确定——很多所谓的"外推方法"实际上是通过插值避免了真正的越界位置(out-of-distribution position)，而不是让模型真正学会处理没见过的位置模式。这提示读者：长度扩展技术的效果评估需要区分"真正泛化"与"精心设计避免了分布外输入"两种情况。
