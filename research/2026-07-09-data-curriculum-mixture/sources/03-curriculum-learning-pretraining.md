---
url: https://arxiv.org/abs/2506.11300
title: "Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning"
type: arxiv_paper
authors: Yang Zhang, Amr Mohamed, Hadi Abdine, Guokan Shang, Michalis Vazirgiannis (Ecole Polytechnique / MBZUAI)
year: 2025 (updated 2026)
accessed: 2026-07-09
quality: 5
relevance: core
---

## Abstract (+ 全文提取要点)

Curriculum learning—organizing training data from easy to hard—has improved efficiency across machine learning domains, yet remains underexplored for language model pretraining. We present the first systematic investigation of curriculum learning in LLM pretraining, with over 200 models trained on up to 100B tokens across three strategies: vanilla curriculum learning, pacing-based sampling, and interleaved curricula, guided by six difficulty metrics spanning linguistic and information-theoretic properties. We evaluate performance on eight benchmarks under three realistic scenarios: limited data, unlimited data, and continual training. Our experiments show that curriculum learning consistently accelerates convergence in early and mid-training phases, reducing training steps by 18-45% to reach baseline performance. When applied as a warmup strategy before standard random sampling, curriculum learning yields sustained improvements up to 3.5%. We identify compression ratio, lexical diversity (MTLD), and readability (Flesch Reading Ease) as the most effective difficulty signals. Our findings demonstrate that data ordering—orthogonal to existing data selection methods—provides a practical mechanism for more efficient LLM pretraining.

## 全文关键内容（PDF 提取）

- 背景：LLM 训练目前几乎都是随机打乱数据顺序训练（i.i.d. sampling），完全忽略了"顺序"这个维度。这与数据选择（filtering）、数据配比（mixing）是两个独立的正交维度。
- 三种课程策略：
  1. **Vanilla curriculum learning**：严格按"难度"从易到难排序整个数据集，固定顺序训练一遍。
  2. **Pacing-based sampling**：用 pacing function（线性、二次、反二次等）控制"引入难样本的速度"——不是硬排序，而是逐渐扩大采样池的难度上限。
  3. **Interleaved curricula**：在每个训练区段内混合不同难度层级的样本，而不是严格分阶段。
- 六个难度度量指标（从 15 个候选里筛出）横跨语言学和信息论两个角度，其中最有效的三个是：**压缩比 (compression ratio)、词汇多样性 (MTLD, Measure of Textual Lexical Diversity)、可读性 (Flesch Reading Ease)**。
- 三种场景测试：数据有限、数据无限、持续训练（continual training）。
- 核心结果：
  - 课程学习在训练早中期持续加速收敛，达到 baseline 表现所需步数减少 18-45%。
  - 最佳用法不是"从头到尾都用课程"，而是**把课程学习当作热身（warmup）阶段**，热身完再切回随机采样——这样能拿到最持久的收益（最多 +3.5%）。
  - 这暗示了课程学习的价值集中在训练早期：帮助模型先建立稳定的表征基础，之后的随机训练收益会更好。
- 引用了 Bengio et al. 2009a 提出的原始 curriculum learning 概念（模仿人类从简单到复杂的学习过程），以及 CV 领域的先驱工作 Kumar et al. 2010, Sinha et al. 2020。
