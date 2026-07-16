# 研究计划：模型合并 (Model Merging)

## 研究问题
为什么可以把多个微调模型的权重直接相加/平均得到更强的模型？

## 子问题
1. 独立训练模型 vs 同源微调模型 —— 为什么权重平均只对后者有效（linear mode connectivity, 排列对称性）
2. Model Soups —— 最朴素的权重平均方法
3. Task Arithmetic —— 把能力表示为可加减的任务向量
4. TIES-Merging —— 解决多模型合并时的参数冗余与符号冲突
5. DARE —— 随机丢弃大部分delta参数仍保持性能
6. SLERP —— 球面插值保持向量范数

## Level: 3
目标：15+ 搜索，5+ 全文提取，2+ arxiv 论文精读
