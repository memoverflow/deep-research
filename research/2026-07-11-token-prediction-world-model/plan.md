# 研究计划：Token 预测与世界模型的哲学关系

## 研究问题
next-token prediction 这个训练目标，是否会迫使语言模型在内部构建出真正意义上的"世界模型"？这个问题涉及机制可解释性的实证证据，也涉及语言哲学层面对"理解/meaning"的定义争论。

## 子问题拆解
1. 实证证据：Othello-GPT / Chess-GPT 等玩具任务上探针实验揭示了什么？线性 vs 非线性表征的争论细节。
2. 更大规模验证：真实 LLM（Llama-2, Claude）上的时空表征、真相几何、稀疏特征研究。
3. 反方论点：Bender & Koller 的"章鱼测试"论文，stochastic parrots 论证的哲学基础。
4. LeCun 式反对：世界模型的"grounded / predictive / persistent" 定义与自回归 LLM 的差距。
5. 元层面：如何精确定义"世界模型"以避免各说各话（Kenneth Li 后续论文）。

## Level
L3（20-40 分钟深度研究）：22+ 次搜索，11 个来源全文/摘要提取并归档，覆盖 arxiv 论文、行业研究（Anthropic）、position paper（Bender & Koller）、访谈材料（Sutskever）。

## 产出
- report/文章：`_research/2026-07-11-token-prediction-world-model.md`
- sources/: 11 个来源文件，含 frontmatter 元数据
- 3 张暗色主题内嵌 SVG 图（pipeline 图、坐标系对比图、两派对比图）
