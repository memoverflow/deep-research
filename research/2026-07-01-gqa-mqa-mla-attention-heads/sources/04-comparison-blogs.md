---
url: https://dreaming.press/posts/mha-vs-mqa-vs-gqa-vs-mla-attention.html
title: "MHA vs MQA vs GQA vs MLA: How Attention Stopped Eating Your GPU Memory"
type: blog
year: 2025
accessed: 2026-07-01
quality: 3
relevance: supporting
---

## Key points (from search snippet + cross-checked against papers)

- The GQA paper's uptraining result — recovering near-MHA quality with ~5% of original
  pretraining compute — is why essentially all major open models (Llama 3, Mistral, Qwen2)
  ship GQA by default rather than plain MHA.
- Practical framing: MQA is the "aggressive" end of the trade-off (maximum cache savings,
  measurable quality cost), GQA is the "safe default" (small cache, near-MHA quality), and MLA
  is a genuinely different axis — compress the *information*, not the *head count*.

## Corroborating snippets from other quality-3 sources (Medium, dev.to comparison articles)

- "GQA is a balance between both attention mechanisms in terms of KV-caching and memory
  bandwidths. MLA requires a significantly lower KV cache yet outperforms MHA in output
  quality." (medium.com/@zaiinn440)
- "The difference between GQA and MLA is that MLA shrinks the cache by compressing what gets
  stored rather than by reducing how many K/Vs are stored by sharing heads." (dev community
  summary of MLA architecture, corroborated directly by DeepSeek-V2 paper Section 2.1.2)
- Models using GQA in production, per public architecture cards: Llama 2 70B, Llama 3 (all
  sizes), Mistral 7B/8x7B, Qwen2 — this matches the general industry-adoption claim above.
