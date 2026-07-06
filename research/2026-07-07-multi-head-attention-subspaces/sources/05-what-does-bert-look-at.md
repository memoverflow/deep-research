---
url: https://arxiv.org/abs/1906.04341
title: "What Does BERT Look At? An Analysis of BERT's Attention"
type: arxiv_paper
authors: Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning
year: 2019
accessed: 2026-07-07
quality: 5
relevance: supporting
---

分析 BERT 各个注意力头具体在关注什么。发现头呈现出清晰可辨认的行为模式：有的头专门关注分隔符 token
（如 [SEP]），有的头关注固定的相对位置偏移（比如总是看前一个/后一个 token），有的头广泛地关注整句话，
有的头专门捕捉特定的句法关系（比如动词关注其直接宾语）。同一层内的多个头经常表现出相似的行为模式。

这是"多头分工"假说的直接实证证据：不同头确实学到了不同、可解释的功能角色，而不是简单的随机冗余副本。
与"Are Sixteen Heads Really Better than One"的冗余发现并不矛盾——一部分头有明确分工，另一部分头
（尤其是同层内行为相似的头）确实存在冗余，可以被裁剪。
