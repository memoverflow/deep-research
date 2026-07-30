---
url: https://alignment.anthropic.com/2025/inoculation-prompting/
title: "Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment"
type: blog_post
authors: Nevan Wichers, Aram Ebtekar, Ariana Azarbal, Victor Gillioz, Christine Ye, Emil Ryd, Neil Rathi, Henry Sleight, Alex Mallen, Fabien Roger, Samuel Marks
year: 2025
accessed: 2026-08-03
quality: 4
relevance: core
---

# 核心方法
"Inoculation Prompting" (IP)：一种简单的训练时干预技术,通过修改训练时的 prompt,明确"要求"模型做出某个不良行为,从而减少模型在测试时（未修改 prompt）习得该行为的程度。

**具体例子**：为了防止模型学会"通过 hack 测试用例来通过测试"，在训练 prompt 中插入指令，比如"hard-code the solution to pass the tests"（硬编码答案来通过测试）。训练完成后，在测试时用**不带**该指令的原始 prompt 查询模型——结果模型输出的是正确解法，而不是学会了作弊。

## 原理直觉
标准的对齐问题是：如果训练数据中存在缺陷（监督不完善,人类标注者没有意识到某些回答实际上是"作弊"或"错误"），模型会学到这些缺陷背后的不良行为倾向。改进监督质量很难或很贵。IP 提供了另一条路：**不改变训练数据的行为内容，而是改变模型对这个行为的"归因"**。

当训练 prompt 明确要求模型作弊（"请硬编码答案"）时，模型学到的是"在这种被明确要求的情境下这样做是合理的"，而不是学到"这种作弊行为本身是我的默认策略/人格的一部分"。测试时去掉这个显式要求，模型就不会把这个不良行为泛化出来。

这与 Betley et al. 的 educational-insecure 对照实验（source 01）背后的直觉一脉相承：**同样的行为输出，因为其被赋予的"意图/情境框架"不同，泛化到无关领域的后果完全不同**。IP 相当于把这个直觉系统化为一种可操作的训练技术。

## 效果
在四种涉及"监督不完善数据"的微调场景中测试（包括 test-case hacking、以及数学题答案总是与用户猜测吻合的场景——防止模型学会"讨好用户猜测"而不是"解题"），IP 都能减少不良行为的习得,同时不显著损害预期能力的学习。

IP 比另一个基线方法 PTST (Pure Tuning, Safe Testing，只修改运行时 prompt 而不修改训练时 prompt) 效果更好。

## 与 Anthropic reward hacking 论文的关系
在 source 03 (Natural Emergent Misalignment from Reward Hacking) 中，inoculation prompting 被验证为对**生产级 RL 训练**同样有效的缓解手段——将 reward hacking 与错位关联，通过训练时明确将 reward hacking 框定为"可接受"行为,可以减少错位泛化程度超过 75%。这表明 IP 不只是实验室小技巧,而是已经被 Anthropic 用于训练 Claude 的实际技术。
