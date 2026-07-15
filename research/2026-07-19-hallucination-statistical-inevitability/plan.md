# Research Plan: 幻觉的统计学必然性

## Topic
OpenAI 2025 论文《Why Language Models Hallucinate》(Kalai, Nachum, Vempala, Zhang) 的核心论证:
幻觉源于预训练目标下的统计学分类误差 + 后训练评测体系奖励猜测而非弃权。

## Level
L3 (deep research, single core paper + supporting theory + reception)

## Sub-questions
1. IIV (Is-It-Valid) reduction: generation-to-classification 数学关系是什么?
2. 校准 (calibration) 与错误率之间的张力是什么? 为什么交叉熵训练自然趋向校准?
3. Singleton rate / Good-Turing missing mass 如何解释生日类事实幻觉?
4. 为什么后训练/RLHF 没有消除幻觉? 二元评分机制的激励分析
5. 真实世界证据: GPT-5 系统卡幻觉率下降, sycophancy/reward hacking 相关性
6. 前作 Kalai & Vempala 2024 (STOC) 与本文关系; 相关理论 (Kalavasis et al 2025, Kleinberg & Mullainathan 2024)

## Sources collected
- arxiv.org/abs/2509.04664 (核心论文, 全文 PDF 提取, 832 行)
- arxiv.org/abs/2311.14648 (前作, STOC 2024)
- 补充: GPT-5 system card 幻觉率报道, sycophancy/calibration collapse 相关文章, benchmark 元分析
