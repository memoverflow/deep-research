---
url: https://arxiv.org/abs/2502.08606
title: "Distillation Scaling Laws"
type: arxiv_paper
authors: Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb (Apple)
year: 2025
accessed: 2026-07-14
quality: 5
relevance: core
---

Abstract: We propose a distillation scaling law that estimates distilled model
performance based on a compute budget and its allocation between the student and teacher.
Our findings mitigate the risks associated with large-scale distillation by enabling
compute-optimal allocation for both the teacher and student to maximize student
performance. We provide compute-optimal distillation recipes for two key scenarios: when a
teacher already exists, and when a teacher needs training. In settings involving many
students or an existing teacher, distillation outperforms supervised learning up to a
compute level that scales predictably with student size. Conversely, if only one student is
to be distilled and a teacher also requires training, supervised learning is generally
preferable.

Key ideas (from abstract + snippet):
- Names and formalizes the "capacity gap" phenomenon: a stronger teacher can produce a
  *worse* student — teacher cross-entropy vs. student loss follows a power law that
  transitions between two regimes depending on relative student/teacher capacity.
- Gives compute-optimal recipes: whether to spend a fixed compute budget training a big
  teacher + distilling vs. just supervised-training the student directly depends on how
  many students will be produced from one teacher (amortization) and whether the teacher
  already exists.
- If only one student and teacher must be trained from scratch: supervised learning on raw
  data is usually better than distillation. If teacher exists/reused across many students:
  distillation wins up to a predictable compute threshold.
