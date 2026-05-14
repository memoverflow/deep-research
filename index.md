---
layout: default
title: Home
---

<section class="hero">
  <h1>Deep Research Archive</h1>
  <p class="hero-desc">穷尽式深度研究，以教学系列文章的形式呈现。<br>每个系列从零开始讲透一个领域。</p>
</section>

{% assign sorted = site.research | sort: 'series_order' %}

{% if sorted.size > 0 %}

<section class="series-section">
  <h2 class="series-heading">理解 Attention 与 Transformer</h2>
  <p class="series-desc">8 篇系列教程 · 从 2014 年 Attention 的诞生到 2025 年的架构大融合</p>

  <div class="article-list">
  {% for post in sorted %}
    <a href="{{ post.url | relative_url }}" class="article-card">
      <span class="article-num">{{ post.series_order }}</span>
      <div class="article-info">
        <h3 class="article-title">{{ post.title }}</h3>
        {% if post.summary %}<p class="article-summary">{{ post.summary }}</p>{% endif %}
        <div class="article-meta">
          {% for tag in post.tags limit:3 %}<span class="article-tag">{{ tag }}</span>{% endfor %}
        </div>
      </div>
      <span class="article-arrow">→</span>
    </a>
  {% endfor %}
  </div>
</section>

{% endif %}
