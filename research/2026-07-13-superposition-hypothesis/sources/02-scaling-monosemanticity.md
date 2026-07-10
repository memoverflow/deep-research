---
url: https://transformer-circuits.pub/2024/scaling-monosemanticity/
title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
type: paper (transformer circuits thread)
authors: Adly Templeton, Tom Conerly, Jonathan Marcus, et al. (Anthropic)
year: 2024
accessed: 2026-07-13
quality: 5
relevance: core
---

这篇论文把 2022 年"Toy Models of Superposition"里的理论工具（稀疏字典学习/稀疏自编码器 SAE）第一次成功放大到生产级大模型——Claude 3 Sonnet（Anthropic 的中型生产模型）上。

方法：在模型的激活值上训练一个稀疏自编码器 (SAE)。SAE 由两层组成：
- Encoder：把激活映射到一个远比原始维度更高的层，再接 ReLU 非线性
- Decoder：把这个高维稀疏表示映射回原始激活空间
理论基础：linear representation hypothesis（特征是激活空间里的方向）+ superposition hypothesis（网络利用高维空间"几乎正交"的性质存下比维度更多的特征）。

关键结果：
1. SAE 能从大模型里提取出高质量、可解释的特征。
2. 可以用 scaling law 来指导 SAE 训练（多大的字典、多少数据）。
3. 提取出的特征高度抽象：多语言（同一概念在不同语言里激活同一特征）、多模态（文本和图像触发同一特征）、能跨越具体与抽象的表达层级。
4. 概念出现频率和需要多大字典才能"分辨"出对应特征之间存在系统性关系（越稀有的概念需要越大的字典才能被单独抽出来，这正好呼应了 Toy Model 里"稀疏度决定是否被单独表示"的相变理论）。
5. **特征可以用来对模型行为进行"steering"（引导/操控）**——这是这篇论文最出名的部分。

Golden Gate Bridge 案例：他们找到一个特征（编号 34M/31164353），专门对"金门大桥"的描述和引用激活。通过人为放大这个特征在模型激活中的强度，可以让 Claude 在几乎所有对话里都不由自主地提到金门大桥（"Golden Gate Claude" demo）。这直接证明了：特征不仅是"可以被读出的"（用于解释模型在想什么），也是"可以被写入的"（用于操控模型说什么）。

安全相关特征：论文还找到了一批与安全直接相关的特征簇——欺骗(deception)、权力寻求(power-seeking)、讨好式奉承(sycophancy)、代码中的安全漏洞和后门、偏见(bias)、危险/犯罪内容等。论文强调要谨慎解读这些发现的含义——"知道谎言的存在"、"有能力说谎"、"真的在现实世界撒谎"是三件不同的事，这项研究非常初步。

Case Study：他们展示了一个具体案例，能通过检测和主动纠正某个"欺骗特征"的激活，来改变模型的输出行为，证明这些特征确实是模型行为背后的因果性中间变量，不只是相关的旁观者。
