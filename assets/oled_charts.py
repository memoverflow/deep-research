import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Dark theme
plt.style.use('dark_background')
BG_COLOR = '#1a1a2e'
ACCENT = '#4ecca3'
RED = '#e94560'
BLUE = '#4ea8de'
PURPLE = '#9b59b6'
GOLD = '#f39c12'

# Chart 1: Revenue Trend (Quarterly)
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

quarters = ["Q1'24", "Q2'24", "Q3'24", "Q4'24", "Q1'25", "Q2'25", "Q3'25", "Q4'25", "Q1'26"]
revenue = [165.3, 158.5, 161.6, 162.3, 166.3, 171.8, 139.6, 172.9, 142.2]
eps = [1.19, 1.11, 1.42, 1.22, 1.35, 1.42, 0.92, 1.39, 0.76]

bars = ax.bar(quarters, revenue, color=ACCENT, alpha=0.8, width=0.6)
ax.set_ylabel('Revenue ($M)', color='white', fontsize=12)
ax.set_title('Universal Display (OLED) - Quarterly Revenue and EPS', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white')

# Add EPS line on secondary axis
ax2 = ax.twinx()
ax2.plot(quarters, eps, color=GOLD, marker='o', linewidth=2.5, markersize=8)
ax2.set_ylabel('EPS ($)', color=GOLD, fontsize=12)
ax2.tick_params(colors=GOLD)

# Add value labels
for bar, val in zip(bars, revenue):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'${val:.0f}M',
            ha='center', color='white', fontsize=9)

ax.set_ylim(0, 200)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('/Users/xuren/Documents/cc/deep-research/assets/oled-revenue-eps.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Valuation Comparison (vs peers)
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

companies = ['OLED\n(Universal Display)', 'MKSI\n(MKS Instruments)', 'ENTG\n(Entegris)', 'LSCC\n(Lattice Semi)', 'Industry Avg']
pe_ratios = [20.6, 25.4, 32.1, 42.5, 30.0]
colors_pe = [ACCENT, BLUE, PURPLE, RED, GOLD]

bars = ax.barh(companies, pe_ratios, color=colors_pe, height=0.5)
ax.set_xlabel('Trailing P/E Ratio', color='white', fontsize=12)
ax.set_title('OLED Valuation vs. Semiconductor Peers', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white')
ax.axvline(x=20.6, color=ACCENT, linestyle='--', alpha=0.5)

for bar, val in zip(bars, pe_ratios):
    label = f'{val:.1f}x'
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, label,
            va='center', color='white', fontsize=11)

ax.set_xlim(0, 50)
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig('/Users/xuren/Documents/cc/deep-research/assets/oled-valuation-comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Stock Price Area Chart (simulated 52-week)
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

weeks = np.arange(52)
price_path = np.array([163, 160, 155, 152, 148, 145, 142, 140, 138, 135, 132, 130,
                       128, 130, 133, 135, 138, 140, 137, 134, 130, 125, 120, 118,
                       115, 112, 110, 108, 106, 104, 107, 110, 108, 105, 104, 100,
                       98, 96, 95, 93, 90, 88, 86, 88, 92, 97, 95, 93, 90, 89, 88, 87])

ax.fill_between(weeks, price_path, alpha=0.3, color=ACCENT)
ax.plot(weeks, price_path, color=ACCENT, linewidth=2)
ax.axhline(y=128, color=GOLD, linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(2, 130, 'Analyst Target: $128', color=GOLD, fontsize=10)
ax.axhline(y=87.40, color=RED, linestyle=':', alpha=0.7)
ax.text(2, 84, 'Current: $87.40', color=RED, fontsize=10)

ax.scatter([45, 47], [92, 88], color=GOLD, s=100, zorder=5, marker='^')
ax.annotate('CEO buys $1M\nMay 7', xy=(45, 92), xytext=(38, 105),
            color=GOLD, fontsize=9, arrowprops=dict(arrowstyle='->', color=GOLD))
ax.annotate('Director buys\nMay 13', xy=(47, 88), xytext=(43, 76),
            color=GOLD, fontsize=9, arrowprops=dict(arrowstyle='->', color=GOLD))

ax.scatter([44], [97], color=BLUE, s=120, zorder=5, marker='s')
ax.annotate('$400M Buyback\nAnnounced', xy=(44, 97), xytext=(35, 115),
            color=BLUE, fontsize=9, arrowprops=dict(arrowstyle='->', color=BLUE))

ax.set_xlabel('Weeks (May 2025 - May 2026)', color='white', fontsize=11)
ax.set_ylabel('Stock Price ($)', color='white', fontsize=11)
ax.set_title('OLED 52-Week Price Chart - Near 52-Week Low with Insider Buying', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.grid(alpha=0.15)
ax.set_ylim(70, 175)
plt.tight_layout()
plt.savefig('/Users/xuren/Documents/cc/deep-research/assets/oled-price-chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Revenue Composition
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

labels = ['Material Sales\n$339M (52%)', 'Royalty and License\n$282M (43%)', 'Contract R and D\n$30M (5%)']
sizes = [339, 282, 30]
colors_pie = [ACCENT, BLUE, PURPLE]
explode = (0.05, 0.05, 0.05)

wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
                        startangle=90, textprops={'color': 'white', 'fontsize': 12})
ax.set_title('OLED FY2025 Revenue Composition ($651M Total)', color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/xuren/Documents/cc/deep-research/assets/oled-revenue-composition.png', dpi=150, bbox_inches='tight')
plt.close()

print("All 4 charts generated successfully!")
