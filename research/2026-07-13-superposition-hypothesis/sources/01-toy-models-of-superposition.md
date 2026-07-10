---
url: https://transformer-circuits.pub/2022/toy_model/index.html
title: "Toy Models of Superposition"
type: paper (transformer circuits thread)
authors: Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, et al. (Anthropic)
year: 2022
accessed: 2026-07-13
quality: 5
relevance: core
---

核心论点：神经网络的单个神经元经常对多个不相关概念激活（多义性/polysemanticity）。这篇论文提出"叠加"(superposition) 假说来解释这一现象——网络能表示比它维度更多的特征，把它们编码为高维空间里"几乎正交"的方向叠加在一起。

关键结论：
1. Superposition 是真实存在的现象，可以在小型 ReLU 网络（在稀疏合成特征上训练）中完全观察和理解。
2. 单义 (monosemantic) 和多义 (polysemantic) 神经元都可以形成。
3. 至少某些计算可以在叠加态下完成（例如绝对值函数的简单电路）。
4. 特征是否被存入叠加态，由一个"相变"(phase change) 决定——不是连续的渐变，而是不同编码策略之间的离散跳变。
5. Superposition 会把特征组织成几何结构（二边形、三角形、五边形、四面体等——均匀多胞体几何）。

两个对抗性的力：
- **Privileged Basis**：只有某些表示空间有"特权基"，鼓励特征与坐标轴（神经元）对齐。
- **Superposition**：线性表示可以用比维度更多的特征数量，利用高维空间中方向可以"几乎正交"的性质。

Toy model 设置：ReLU(W^T W x - b)，输入特征是稀疏的（大部分时候是 0，偶尔非零），"重要性"是均方误差损失里的一个标量乘子。

相变实验：2 个特征塞进 1 个隐藏维度。三种自然的权重配置：
- W=[1,0]（忽略额外特征）
- W=[0,1]（忽略第一个特征，给额外特征专属维度）
- W=[1,-1]（"antipodal" 反极性叠加，把两个特征都存起来，但代价是失去表示两者同时出现的能力）
实验证明这三种策略之间存在真正的一阶相变——损失函数的导数在临界点处不连续。

几何结构：均匀叠加（所有特征重要性和稀疏度相同）下，特征会自组织成完全对称的几何形态（如五边形、Möbius strip 状的三角形排列等）。这暗示了叠加编码存在某种普适的组合数学规律。

推测性关联（论文明确说是初步/preliminary evidence）：
- 叠加可能与对抗样本 (adversarial examples) 有关——对抗扰动利用了叠加编码里特征之间的干扰。
- 叠加可能与 grokking（训练后期突然泛化）有关。
- 叠加也许能为 Mixture of Experts 模型的表现提供一种理论视角。

作者认为：我们观察到的神经网络，某种意义上是在"含噪声地模拟一个更大、更稀疏的网络"。
