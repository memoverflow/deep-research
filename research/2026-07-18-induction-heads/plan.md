## 研究计划：归纳头 (Induction Heads) 与上下文学习机制

**话题：** 归纳头 —— transformer 内部实现 in-context learning 的可解释性核心机制

**去重检查：** topics-published.md 已确认没有直接写过"归纳头/induction heads"这个话题。相关但不重复的已发布话题：
- In-Context Learning 理论解释（2025-05-18）— 侧重理论/统计学习角度，不涉及具体电路机制
- Chain-of-Thought 计算复杂性（2025-05-19）— 不同主题
- 秩坍缩/Token Uniformity（2026-07-13）— 涉及 attention 但主题完全不同
- Token 预测与世界模型（2026-07-11）— 不同角度

归纳头是纯粹的机制可解释性(mechanistic interpretability)话题，此前系列从未涉及，属于全新子领域。

**研究等级：** L3（Deep）

**核心来源：**
1. Olsson et al. 2022, "In-context Learning and Induction Heads" (Anthropic, Transformer Circuits Thread) — 原始假说论文
2. Singh et al. 2024 (arXiv:2404.07129), "What needs to go right for an induction head?" — 后续细化研究
3. Sean Trott 2026 博客 — 批判性反思视角

**搜索执行：** 完成 15+ 次 web_search（涵盖英文关键词、arxiv site 搜索、中文搜索尝试、机制细节搜索、批评视角搜索），并通过 terminal curl 绕过 web_extract 限制，成功提取 3 篇核心来源全文/摘要。

**文章结构：**
1. 故事开场：ICL 现象引入 + 问题定义（如何量化 ICL）
2. 归纳头机制讲解：两层 head 接力、previous token head + induction head、K-composition
3. 六条证据详解
4. 后续研究的两条追问：子电路分解 + 哲学层面的解释力质疑
5. So what：能力涌现的相变性质 + 可解释性方法论的价值

**图表：** 2 张内嵌 SVG（归纳任务示意图 + 两层电路示意图）
