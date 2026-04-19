import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats

eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])

# Filter to only modern models (post-2022) to match Epoch's view
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# Color map for top orgs only
top_orgs = ['OpenAI', 'Google', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
org_colors = {
    'OpenAI': '#F472B6',
    'Google': '#34D399',
    'Google DeepMind': '#34D399',
    'Meta AI': '#FB923C',
    'Anthropic': '#A78BFA',
    'xAI': '#60A5FA',
}
eci_df['color'] = eci_df['Organization'].apply(
    lambda o: org_colors.get(o, '#A8A29E')
)

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot dots by org for legend
for org in top_orgs:
    subset = eci_df[eci_df['Organization'] == org]
    if len(subset):
        ax.scatter(subset['Release date'], subset['ECI Score'],
                   color=org_colors[org], s=55, alpha=0.85,
                   label=org, zorder=3, linewidths=0)

# Plot "Other" orgs
other = eci_df[~eci_df['Organization'].isin(top_orgs)]
ax.scatter(other['Release date'], other['ECI Score'],
           color='#A8A29E', s=55, alpha=0.7, label='Other', zorder=2, linewidths=0)

# Frontier trend line — rolling max (upper envelope), not OLS
# eci_df_sorted = eci_df.sort_values('Release date')
# eci_df_sorted['frontier'] = eci_df_sorted['ECI Score'].cummax()
# ax.plot(eci_df_sorted['Release date'], eci_df_sorted['frontier'],
#         color='#6B7280', linewidth=1.2, linestyle='-', alpha=0.6, zorder=1)

# Axis formatting to match Epoch
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=0, fontsize=10)
ax.set_ylim(55, 165)
ax.set_ylabel("Score", fontsize=11, color='#374151')
ax.set_xlabel("Release date", fontsize=11, color='#374151')
ax.set_title("Epoch Capabilities Index (ECI)", fontsize=14, fontweight='500',
             loc='left', pad=12, color='#111827')

# Light grid like Epoch
ax.grid(True, axis='y', color='#E5E7EB', linewidth=0.8, linestyle='-')
ax.grid(True, axis='x', color='#E5E7EB', linewidth=0.8, linestyle='-')
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

# Legend top right, compact
legend = ax.legend(frameon=False, fontsize=9, loc='upper left',
                   bbox_to_anchor=(1.01, 1), borderaxespad=0,
                   markerscale=1.2, handletextpad=0.4)

plt.tight_layout()
plt.savefig("eci_recreated.png", dpi=150, bbox_inches='tight')
plt.show()