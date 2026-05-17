import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Dark theme settings
plt.style.use('dark_background')
BG_COLOR = '#1a1a2e'
GREEN = '#4ecca3'
RED = '#e94560'
BLUE = '#3282b8'
ORANGE = '#f39c12'
PURPLE = '#9b59b6'
YELLOW = '#f1c40f'

output_dir = os.path.expanduser('~/Documents/cc/deep-research/assets')

# ===== Chart 1: Stock Price (52-week area chart) =====
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Simulated 52-week price data based on research:
# 52-week range: $23.97 - $87.30, current ~$29.92
months = ['Jun25','Jul25','Aug25','Sep25','Oct25','Nov25','Dec25','Jan26','Feb26','Mar26','Apr26','May26']
prices = [47, 55, 62, 70, 78, 87, 65, 45, 38, 24, 32, 30]

ax.fill_between(range(len(prices)), prices, alpha=0.3, color=GREEN)
ax.plot(prices, color=GREEN, linewidth=2.5)
ax.axhline(y=87.30, color=RED, linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=23.97, color=BLUE, linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=29.92, color=YELLOW, linestyle='--', alpha=0.7, linewidth=1.5)

ax.annotate('52W High: $87.30', xy=(5, 87.30), fontsize=9, color=RED, ha='center', va='bottom')
ax.annotate('52W Low: $23.97', xy=(9, 23.97), fontsize=9, color=BLUE, ha='center', va='top')
ax.annotate('Current: $29.92', xy=(11, 29.92), fontsize=9, color=YELLOW, ha='right', va='bottom')

ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, rotation=45, fontsize=8, color='white')
ax.set_ylabel('Stock Price ($)', color='white', fontsize=10)
ax.set_title('UPST — 52-Week Stock Price', color='white', fontsize=14, fontweight='bold')
ax.grid(alpha=0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'upst-price-chart.png'), dpi=150, facecolor=BG_COLOR)
plt.close()

# ===== Chart 2: Quarterly Revenue Bar Chart =====
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

quarters = ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025', 'Q1 2026']
revenue = [213.4, 257.3, 270, 296.1, 308.2]  # millions
yoy_growth = ['+67%', '+102%', '+50%', '+35%', '+44%']

bars = ax.bar(quarters, revenue, color=GREEN, alpha=0.8, width=0.6)

for i, (bar, growth) in enumerate(zip(bars, yoy_growth)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            growth, ha='center', va='bottom', color=ORANGE, fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
            f'${revenue[i]:.1f}M', ha='center', va='center', color='white', fontsize=10)

ax.set_ylabel('Revenue ($M)', color='white', fontsize=10)
ax.set_title('UPST — Quarterly Revenue & YoY Growth', color='white', fontsize=14, fontweight='bold')
ax.set_ylim(0, 380)
ax.grid(alpha=0.1, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'upst-revenue-chart.png'), dpi=150, facecolor=BG_COLOR)
plt.close()

# ===== Chart 3: Valuation Comparison (horizontal bar) =====
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

companies = ['UPST\n(Upstart)', 'SOFI\n(SoFi)', 'AFRM\n(Affirm)', 'PGY\n(Pagaya)', 'LC\n(LendingClub)']
# P/S ratios approximate
ps_ratios = [2.8, 4.5, 6.2, 1.5, 2.1]
colors = [GREEN, BLUE, BLUE, BLUE, BLUE]

bars = ax.barh(companies, ps_ratios, color=colors, alpha=0.8, height=0.5)
ax.axvline(x=2.8, color=GREEN, linestyle='--', alpha=0.5)

for bar, val in zip(bars, ps_ratios):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}x', va='center', color='white', fontsize=11)

ax.set_xlabel('Price/Sales Ratio', color='white', fontsize=10)
ax.set_title('UPST vs Peers — P/S Valuation Comparison', color='white', fontsize=14, fontweight='bold')
ax.grid(alpha=0.1, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'upst-valuation-chart.png'), dpi=150, facecolor=BG_COLOR)
plt.close()

# ===== Chart 4: Revenue Composition Pie =====
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

segments = ['Personal Loans\n(Core)', 'Auto Loans', 'Home Equity\n(HELOC)', 'Small Dollar\nLoans']
sizes = [65, 20, 10, 5]
colors_pie = [GREEN, BLUE, ORANGE, PURPLE]
explode = (0.05, 0, 0, 0)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=segments, 
                                   colors=colors_pie, autopct='%1.0f%%',
                                   shadow=True, startangle=90,
                                   textprops={'color': 'white', 'fontsize': 11})
for autotext in autotexts:
    autotext.set_fontweight('bold')

ax.set_title('UPST — Revenue Composition by Product', color='white', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'upst-revenue-mix.png'), dpi=150, facecolor=BG_COLOR)
plt.close()

print("All charts generated successfully!")
