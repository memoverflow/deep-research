# 研究计划：ROME/MEMIT 知识编辑

## 研究问题
大语言模型内部的事实性知识存储在哪？能否精确定位并编辑单条事实，而不影响其他知识？

## 子问题
1. 因果追踪（Causal Tracing）方法原理，定位知识存储的两个"热点"
2. ROME：秩一编辑，FFN 作为键值存储的数学表达
3. MEMIT：如何将编辑扩展到数千条事实（跨层分摊更新）
4. 批评视角：Does Localization Inform Editing?（定位≠最优编辑层）
5. 批评视角：RippleEdits（连锁效应未被现有方法捕捉）
6. 相关基础工作：Knowledge Neurons (Dai et al 2021), FFN=KV memory (Geva et al 2021), PMET (2023)

## Level
L3 — 15+ 次搜索,7 个来源全文/摘要提取(ROME, MEMIT, Knowledge Neurons, Geva FFN-KV, PMET, Does Localization Inform Editing, RippleEdits + baulab 项目页)

## 图表
2 张暗色内嵌 SVG：(1) 因果追踪两阶段信息流图, (2) ROME vs MEMIT 单层/多层编辑对比图, (3) 结尾流程图
