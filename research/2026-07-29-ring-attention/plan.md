研究主题: Ring Attention 与上下文并行原理
日期: 2026-07-29
级别: L3

## 研究问题
Ring Attention 如何让上下文长度随设备数量线性扩展？其数学原理、局限性（因果负载不均衡）以及与其他上下文并行方案（DeepSpeed Ulysses）的对比。

## 搜索策略
- Ring Attention 原论文（arxiv 2310.01889）全文提取（通过 arxiv HTML + PDF 双重获取）
- Blockwise Parallel Transformer 前置工作（arxiv 2305.19370）
- Striped Attention 负载均衡改进（arxiv 2311.09431）全文 PDF 提取
- DeepSpeed Ulysses 对比方案（GitHub README + HuggingFace blog）
- Large World Model 应用案例（arxiv 2402.08268）
- 中文技术博客交叉验证（segmentfault, 站内笔记）

## 搜索次数
web_search: 12次（英文8次，中文2次，其余为验证性搜索）
web_extract/curl 全文提取: 5篇（2篇通过 PDF 转文本，2篇通过 arxiv HTML）

## 关键结论
1. Ring Attention 用环形拓扑 + 计算通信重叠，把每层激活内存开销从与序列长度相关（2bsh）降到只与块大小相关（6bch），实现序列长度随设备数线性扩展。
2. 通信被计算掩盖的充分条件: 块大小 c ≥ F/B（算力/带宽比）。
3. Striped Attention 指出并修复了 Ring Attention 在因果注意力场景下的负载不均衡问题，通过打乱 token 分配顺序（而非连续切分），使每设备负载均衡，实测提速 1.45x-1.65x。
4. DeepSpeed Ulysses 是另一条路线（按头切分+all-to-all），与 Ring Attention 各有工程权衡。
