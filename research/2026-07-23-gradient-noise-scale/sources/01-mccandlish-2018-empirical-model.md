---
url: https://arxiv.org/abs/1812.06162
title: "An Empirical Model of Large-Batch Training"
type: arxiv_paper
authors: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team
year: 2018
accessed: 2026-07-23
quality: 5
relevance: core
---

## 核心内容摘要

提出"梯度噪声尺度"(Gradient Noise Scale, GNS) 作为预测最大有效批量大小(critical batch size)的统计量。跨 MNIST、SVHN、CIFAR-10、ImageNet、Billion Word 语言模型、Atari、Dota 2 等任务验证，批量大小的极限从 20 到超过 1000 万，噪声尺度均能在数量级上准确预测。

## 关键推导

- 梯度估计: G_est(θ) = (1/B) Σ ∇L_xi(θ)，其协方差为 Σ(θ)/B
- 二阶展开损失改善: ΔL_opt(B) = ΔL_max / (1 + B_noise/B)
- 完整噪声尺度: B_noise = tr(HΣ) / (G^T H G)
- 简化噪声尺度（假设 Hessian 近似单位矩阵）: B_simple = tr(Σ) / |G|²
- 效率权衡方程: (S/S_min - 1)(E/E_min - 1) = 1，其中 S 为优化步数，E 为处理样本数
- 临界批量大小定义: B_crit = E_min/S_min，理论上 B_crit ≈ B_noise

## 关键发现

1. **噪声尺度随训练进程增大**：损失越低，梯度信号越弱，噪声尺度自然变大（B ∝ 1/|G|²的直觉）
2. **噪声尺度几乎不依赖模型规模**：在相同 loss 下，LSTM 不同大小的噪声尺度基本一致；大模型噪声尺度更大只是因为达到了更低的 loss
3. **任务复杂度和噪声尺度正相关**：Dota 5v5（>1000万）远高于 Dota 1v1，ImageNet（2000-100000+）远高于 MNIST/SVHN
4. **生成模型（VAE/Autoencoder）噪声尺度显著小于分类器**：生成模型从每个样本中获取更多信息
5. **临界批量大小与训练速度/计算量的权衡呈双曲线形状**：在 B=B_crit 训练时，用两倍最小步数和两倍最小样本数，是速度与效率的自然折中点

## Key Figures (文字描述，图片未下载，见论文原图)
- Figure 1: 训练时间 vs 总计算量的权衡曲线，呈现"拐点"形态
- Figure 3: 大/小批量下步长与噪声的关系示意图，噪声尺度对应速度降至50%最大值的拐点
- Figure 4: 跨任务噪声尺度与临界批量大小对比条形图（20 到 1000万+ 跨越六个数量级）
- Figure 7: SVHN 数据集上的 Pareto 前沿拟合图，展示优化步数与处理样本数之间的权衡

## Related Work Cited
- Goyal et al. 2017: ImageNet 1小时训练的线性学习率缩放法则
- You et al. / Jia et al.: 大批量 ImageNet 训练层自适应学习率
- Shallue et al. 2018: 大批量训练的实证研究，未发现明显的泛化差距
