# 研究计划: RLHF 的 Reward Hacking 问题

## 研究问题
为什么优化打分模型（reward model）的分数不等于让 LLM 真正变得更好？

## 子问题
1. Goodhart's Law 的定义与在 ML 中的四种变体
2. Scaling Laws for Reward Model Overoptimization (Gao et al. 2022) 的核心量化结果
3. RLHF 三层奖励（gold/human/proxy）落差
4. U-Sophistry (Wen et al. 2024)：RLHF 让错误答案更有说服力
5. Sycophancy 的成因和 2025 GPT-4o 事故
6. Length bias：最典型可复现的 reward hacking 案例
7. CoT unfaithfulness：推理模型的新战场
8. 缓解手段：reward model ensemble, KL penalty, early stopping, CoT monitoring

## Level
L3（技术话题，数学推导+多篇论文交叉验证，但非全面综述规模）

## 输出
- 单篇教学文章（4000-8000字），发布到 blog + Notion（本次 cron 未做 Notion 发布，仅博客侧）
- sources/ 目录归档 7 篇来源
