# 研究计划: Lost in the Middle

## 研究问题
为什么大模型在处理长上下文时,信息利用率随位置呈 U 形曲线(开头/结尾高,中间低)?
这是训练数据的问题,还是架构本身的必然结果?能否修复?

## 子问题拆解
1. 现象本身: Liu et al. 2023 "Lost in the Middle" 的实验设计与结果
2. 相关现象: attention sink (Xiao et al. 2023, StreamingLLM)
3. 理论根源: 因果掩码 + 残差连接的几何证明 (Chowdhury 2026, Wu et al. 2025 MIT)
4. 训练后是否保留: Herasimchyk et al. 2026 残差感知理论 + Rutgers 认知科学视角
5. 实践缓解: Hsieh et al. 2024 found-in-the-middle 校准, reordering, reranking

## Level: L3
预期: 8+ web_search, 5+ web_extract/curl 全文提取, 5-6 篇 arxiv 论文
