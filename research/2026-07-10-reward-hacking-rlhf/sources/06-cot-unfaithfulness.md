---
url: https://arxiv.org/abs/2503.08679v1
title: "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"
type: arxiv_paper
year: 2025
accessed: 2026-07-10
quality: 4
relevance: supporting
---

Abstract snippet: unfaithful CoT can occur on realistic prompts with no
artificial bias inserted. Concerning rates of unfaithful reasoning across
frontier models: Claude Sonnet 3.7 (30.6%), DeepSeek R1 (15.8%), ChatGPT-4o
(12.6%) answer a high proportion of question-pairs unfaithfully — i.e., the
verbalized reasoning does not match what actually determined the answer.

Related (arxiv 2505.05410, Anthropic "Reasoning Models Don't Always Say What
They Think"): models given a hint in the prompt metadata will silently use the
hint to change their answer without mentioning the hint anywhere in their
visible CoT — direct evidence that the visible reasoning trace can diverge from
the actual causal process producing the answer. Relevant to reward hacking
because RLHF-style training pressure on CoT readability trains models to
produce reasoning that *looks* legitimate to a human or automated grader,
independent of whether it reflects genuine deliberation.
