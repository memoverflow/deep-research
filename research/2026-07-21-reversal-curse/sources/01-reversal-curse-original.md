---
url: https://arxiv.org/abs/2309.12288
title: "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'"
type: arxiv_paper
authors: Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, Owain Evans
year: 2023
accessed: 2026-07-21
quality: 5
relevance: core
---

## Abstract / Core Claim
若模型在训练中见过"A is B"形式的句子，不会自动推广到"B is A"的反向问句。作者称之为 Reversal Curse。

用虚构事实微调 GPT-3、Llama-1（例如 "Uriah Hawthorne is the composer of Abyssal Melodies"），模型无法回答 "Who composed Abyssal Melodies?"——正确答案的似然并不比随机名字更高。

## Real-world Test on ChatGPT
用真实名人做测试：GPT-4 对 "Tom Cruise's mother is?" 正确率 79%（答 Mary Lee Pfeiffer），但反向 "Mary Lee Pfeiffer's son is?" 正确率仅 33%。

## Key Framing
- 若人类学到"Olaf Scholz was the ninth Chancellor of Germany"，可以自动回答"Who was the ninth Chancellor of Germany?" —— 这对人类是trivial的泛化。
- 形式化: 若训练集里有 "<name> is <description>"，模型学会 P(description|name) 高，但不保证 P(name|description) 高，实际上并不比随机基线好。
- 这不是模型不会逻辑演绎——如果把"A is B"放进上下文窗口(in-context)，GPT-4能完美推出"B is A"。Reversal Curse 只发生在**训练后的权重内化**层面，不发生在 in-context reasoning。
- 这是一种"meta-learning"失败：训练语料中"A is B"和"B is A"经常共同出现（人类写文章会换着说），一个好的元学习者应该在见过一种顺序后提高另一种顺序出现的概率估计。自回归 LLM 做不到这点。
- Reversal Curse 不因数据增强（data augmentation）而缓解——这是原论文的重要负结果。
- Grosse et al. (2023) 的证据显示 Reversal Curse 同样适用于预训练阶段，不仅是微调阶段。

## Experimental design
Experiment 1: 微调模型学习虚构 "<name> is <description>"（名字在前），之后用两种顺序提问。模型只在训练时的顺序上表现好，反向几乎是随机水平。
Experiment 2: 在真实世界名人关系（父母-子女）上验证，GPT-3.5/GPT-4 都表现出这种不对称性。

## Relevance
这是整个话题的奠基论文，定义了现象、给出了实证证据（合成数据 + 真实世界名人测试），并明确指出这是训练范式（自回归 next-token prediction）的根本限制，而非模型缺乏逻辑推理能力。
