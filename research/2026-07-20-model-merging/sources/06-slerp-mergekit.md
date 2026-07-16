---
url: https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md
title: "mergekit merge methods documentation (SLERP, linear, TIES, DARE, task arithmetic)"
type: technical_documentation
authors: Arcee AI
year: 2024
accessed: 2026-07-16
quality: 4
relevance: supporting
---

mergekit is the de facto open-source library (Arcee AI) implementing model merging methods used widely by open-source LLM community. Supports:
- linear (model soup style averaging)
- slerp — spherical interpolation between two models along great-circle arc on unit hypersphere
- task_arithmetic, ties, dare_ties, dare_linear — the methods discussed in this article
- Layer-wise/component-wise merge configuration via YAML

SLERP formula: for vectors v1, v2 with angle Ω between them, Slerp(v1,v2;t) = [sin((1-t)Ω)/sinΩ]v1 + [sin(tΩ)/sinΩ]v2. Preserves vector norm along interpolation path unlike naive LERP (linear interpolation), which shrinks the resulting vector's magnitude when v1, v2 point in different directions.

mergekit has been used to merge thousands of open community models, producing many Open LLM Leaderboard SOTA checkpoints (referenced in arxiv 2403.13257 "Arcee's MergeKit" paper).
