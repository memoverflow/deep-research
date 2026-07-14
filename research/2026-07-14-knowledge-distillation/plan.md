# 研究计划：知识蒸馏 (Knowledge Distillation) 原理

## 选题理由
去重检查后，"知识蒸馏"话题尚未在 topics-published.md 中出现，且与 DeepSeek-R1 等热点强相关，适合作为 LLM 原理深度解析系列第 41 篇。

## 研究范围
1. Hinton 2015 原始 KD 论文：软标签、温度、暗知识
2. LLM 场景下蒸馏的特殊问题：曝光偏差/分布偏移
3. 正向 KL vs 反向 KL：MiniLLM 的论证 + 2024 AKL 论文的修正
4. On-policy 蒸馏：GKD (Google DeepMind)
5. 容量差距与蒸馏 scaling laws (Apple 2025)
6. 白盒 vs 黑盒蒸馏，序列级蒸馏 (Kim & Rush 2016)
7. 真实案例：DeepSeek-R1 蒸馏出的六个小模型
8. 补充视角：label smoothing / self-distillation 作为正则化的另一种解释

## Level
L3-L4（涉及多篇论文的数学论证 + 实际案例）

## 产出
- 1 篇教学文章 (~5500 字), 2 张内嵌 SVG 架构图
- 9 个 sources/*.md 归档文件
- Notion + 博客双发布
