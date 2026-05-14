---
layout: default
title: Home
---

# 🔬 Deep Research Archive

穷尽式深度研究归档。每篇研究包含完整的原始材料、论文提取、多来源交叉验证。

---

{% assign sorted = site.research | sort: 'date' | reverse %}
{% for post in sorted %}
<div class="research-card">
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <div class="card-meta">
    <span class="date">{{ post.date | date: "%Y-%m-%d" }}</span>
    <span class="level">Level {{ post.level }}</span>
    <span class="sources">{{ post.sources_count }} sources</span>
  </div>
  {% if post.summary %}<p>{{ post.summary }}</p>{% endif %}
</div>
{% endfor %}
