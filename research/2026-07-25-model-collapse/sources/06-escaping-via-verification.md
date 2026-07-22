---
url: https://arxiv.org/abs/2510.16657
title: "Escaping Model Collapse via Synthetic Data Verification: Near-Optimal Rates"
type: arxiv_paper
authors: (2025)
year: 2025
accessed: 2026-07-25
quality: 4
relevance: supporting
---

在线性回归设定下研究：如果对合成数据加一层"验证/过滤"（verifier，类似 reward model 打分或规则校验），能否避免坍缩？

结论：verifier 过滤后的合成数据在短期内能带来性能提升，但长期反复迭代会让参数估计逐渐被"拉向验证器自身的知识中心"（verifier's knowledge center）——即验证器本身的偏差会被放大，成为新的系统性瓶颈。也就是说，"用 AI 过滤 AI 生成的数据"并不能无限循环地免费获得改进，验证器质量决定了整个系统的上限。

与"Escaping Model Collapse"系列论文中提出的推理：直接监督（人类反馈/ground truth 校验）优于纯粹自我验证的循环。VAE 数字生成实验（MNIST 类比）中：40 轮合成数据再训练后，无过滤的分支严重退化（红色曲线），过滤后分支保持较清晰生成（绿色曲线），但仍不及全真实数据训练的上限。
