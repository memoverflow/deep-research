# 研究计划：LoRA 低秩适应理论

## 选题理由
话题池中的"LoRA 低秩适应的理论基础（为什么 rank 16 就够了？）"尚未在 topics-published.md 中出现，
是"LLM 原理深度解析"系列合理的第 48 篇——衔接前面已发布的"知识蒸馏"、"叠加态假说"等关于知识压缩/结构化的主题。

## 研究范围
1. LoRA 原始论文（Hu et al. 2021）核心机制、GPT-3 175B 实验（最优 rank、subspace similarity、ΔW/W 相关性）
2. Intrinsic Dimensionality（Aghajanyan et al. 2020）理论前身
3. LoRA+ （Hayou et al. 2024）学习率不对称问题
4. LoRA vs Full FT 几何差异 / intruder dimensions（2024）
5. DoRA 权重分解变体（2024）
6. QLoRA 量化+低秩组合（Dettmers et al. 2023）

## 执行记录
- Level 3：完成 16 次 web_search（多 query 策略：精确论文名、理论关键词、实验数据关键词、site:arxiv.org）
- 提取 6 篇 arxiv 论文摘要 + LoRA 原始论文全文（通过 ar5iv HTML 提取具体实验数据表格与数值）
- web_extract 工具因 DuckDuckGo backend 限制无法直接提取网页，改用 terminal curl 抓取 arxiv abstract 页面和 ar5iv 全文页面，成功绕过限制
- 归档 6 个 sources/*.md 文件，含论文摘要、关键实验数据、核心论点
- 撰写教学文章：从 Intrinsic Dimension 发现引入 → LoRA 核心机制（BA 分解、初始化、合并零延迟）→ ΔW/W 相关性与"选择性放大"直觉 → LoRA+/DoRA/QLoRA/Illusion of Equivalence 串联收尾
- 内嵌 1 张 SVG 对比图（全量微调 vs Adapter vs LoRA 架构对比）
