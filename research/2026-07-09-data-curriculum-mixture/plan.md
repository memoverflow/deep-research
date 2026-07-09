# 研究计划：数据配比与课程学习 (Data Mixing & Curriculum Learning)

## 选题
话题池选择："数据配比与课程学习（Curriculum Learning）"
去重检查：已发布话题清单中无此话题，Chinchilla/Scaling Laws 已发布但角度不同（计算量分配 vs 数据本身的组成与顺序）

## Level
L3（数学推导适中，多篇论文交叉验证）

## 子问题
1. 数据配比 (Data Mixture)：为什么域权重(domain weights)影响模型性能？DoReMi 怎么用 minimax 优化找到最优配比？
2. Data Mixing Laws：如何用小规模实验预测大规模训练的最优配比？
3. 课程学习 (Curriculum Learning)：训练数据的顺序是否重要？easy-to-hard 策略效果如何？
4. 退火与高质量数据后置：WSD 学习率调度器 + 训练末期切换高质量数据的工程实践（MiniCPM）
5. 数据质量 vs 数据顺序 vs 数据配比 三者关系梳理

## 搜索与提取记录
- web_search 约15次（DoReMi, Data Mixing Laws, curriculum learning, WSD, phi-1.5, 中文课程学习等）
- curl/web_extract 提取 6 篇论文摘要+全文:
  - DoReMi (arXiv 2305.10429)
  - Data Mixing Laws (arXiv 2403.16952)
  - Beyond Random Sampling: Curriculum Learning for LLM Pretraining (arXiv 2506.11300, 全文PDF via PyMuPDF)
  - MiniCPM WSD (arXiv 2404.06395)
  - Understanding WSD - River Valley Loss Landscape (arXiv 2410.05192)
  - Phi-1.5 Textbooks Are All You Need II (arXiv 2309.05463)
