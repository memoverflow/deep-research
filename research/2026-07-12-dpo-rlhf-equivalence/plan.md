# Research Plan: DPO 与 RLHF 的数学等价性

## Topic
DPO (Direct Preference Optimization) 为什么在数学上等价于 RLHF —— 完整推导链：
KL 约束 RL 目标 → 闭式最优策略 → 反解奖励 → 代入 Bradley-Terry → 配分函数消掉 → 
分类损失 → 梯度分析 → 局限性 (likelihood displacement, OOD, offline-only)

## Level: L3 (Deep)
- Target: 15+ web_search, 5+ full extractions, arxiv papers covered
- Actual: 11 distinct search queries (some search backend failures on repeat queries),
  9 successful web_extract/curl extractions covering original paper + 4 independent
  derivation write-ups + 1 empirical follow-up paper + Nathan Lambert's RLHF book chapter

## Sub-questions covered
1. Original DPO paper (Rafailov et al. 2023) — abstract + appendix derivation chain
2. Independent derivation walkthrough (fireant.github.io, 4-part series) — cross-verification
3. Independent derivation walkthrough (Vishal Bakshi blog) — cross-verification of appendix math
4. Full gradient derivation (RLHF Book Ch.8, Nathan Lambert) — the σ(·) weighting interpretation
5. Empirical failure mode: likelihood displacement (Razin et al. ICLR 2025) — safety-relevant
   quantitative result (Llama-3-8B-Instruct refusal rate 74.4%→33.4%)
6. DPO variant landscape + OOD/reference-policy issues (multiple blog/paper snippets)

## Why this topic passes dedup check
"DPO/GRPO 新范式" was previously covered as a high-level overview in the RLHF series
(2025-06-30). This article goes deep into the FULL closed-form derivation with explicit
gradient analysis — a level of mathematical depth not present in the prior overview article.
Per skill's own dedup examples: "Scaling Laws 写过后，Chinchilla 最优分配 可以写（子话题足够
独立）✓" — analogous relationship here between "DPO/GRPO 概述" and "DPO 完整数学推导+梯度分析".

## Output
- Article: _research/2026-07-12-dpo-rlhf-equivalence.md (~5500 字, L3, 2 SVG diagrams)
- Sources: research/2026-07-12-dpo-rlhf-equivalence/sources/*.md (6 files)
- series_order: 34, series_total: 34 (bumped all prior "LLM 原理深度解析" series articles
  from series_total: 33 → 34)
