---
layout: default
title: Home
---

{% assign all_posts = site.research | sort: 'date' | reverse %}
{% assign series_names = all_posts | map: 'series' | uniq %}

<!-- Tag filter bar -->
<div class="tag-filter">
  <button class="filter-btn active" data-tag="all">全部</button>
  {% assign all_tags = "" | split: "" %}
  {% for post in all_posts %}{% for tag in post.tags %}{% unless all_tags contains tag %}{% assign all_tags = all_tags | push: tag %}{% endunless %}{% endfor %}{% endfor %}
  {% for tag in all_tags %}
  <button class="filter-btn" data-tag="{{ tag }}">{{ tag }}</button>
  {% endfor %}
</div>

<!-- Series sections -->
{% for series_name in series_names %}
{% if series_name %}
{% assign series_posts = all_posts | where: 'series', series_name | sort: 'series_order' %}
{% assign first = series_posts | first %}
{% assign series_tags = "" %}{% for p in series_posts %}{% for t in p.tags %}{% assign series_tags = series_tags | append: t | append: " " %}{% endfor %}{% endfor %}

<section class="series-section" data-tags="{{ series_tags }}">
  <h2 class="series-heading">{{ series_name }}</h2>
  <p class="series-desc">{{ series_posts.size }} 篇系列教程 · Level {{ first.level }}</p>

  <div class="article-list">
  {% for post in series_posts %}
    <a href="{{ post.url | relative_url }}" class="article-card" data-tags="{% for t in post.tags %}{{ t }} {% endfor %}">
      <span class="article-num">{{ post.series_order }}</span>
      <div class="article-info">
        <h3 class="article-title">{{ post.title }}</h3>
        {% if post.summary %}<p class="article-summary">{{ post.summary }}</p>{% endif %}
        <div class="article-meta">
          {% for tag in post.tags limit:3 %}<span class="article-tag">{{ tag }}</span>{% endfor %}
        </div>
      </div>
      <span class="article-arrow">&rarr;</span>
    </a>
  {% endfor %}
  </div>
</section>

{% endif %}
{% endfor %}

<script>
// Tag filtering
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    const tag = this.dataset.tag;
    document.querySelectorAll('.series-section').forEach(section => {
      if (tag === 'all') {
        section.style.display = '';
        section.querySelectorAll('.article-card').forEach(c => c.style.display = '');
      } else {
        const cards = section.querySelectorAll('.article-card');
        let anyVisible = false;
        cards.forEach(card => {
          if (card.dataset.tags.includes(tag)) {
            card.style.display = '';
            anyVisible = true;
          } else {
            card.style.display = 'none';
          }
        });
        section.style.display = anyVisible ? '' : 'none';
      }
    });
  });
});
</script>
