# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import acf

# eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
# eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
# eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
# eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# # Fit a simple linear trend and get residuals
# eci_df['date_numeric'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
# slope, intercept = np.polyfit(eci_df['date_numeric'], eci_df['ECI Score'], 1)
# eci_df['residuals'] = eci_df['ECI Score'] - (slope * eci_df['date_numeric'] + intercept)
# coeffs = np.polyfit(eci_df['date_numeric'], eci_df['ECI Score'], 1)
# eci_df['residuals'] = eci_df['ECI Score'] - np.polyval(coeffs, eci_df['date_numeric'])

# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
# fig.patch.set_facecolor('white')

# # ACF on raw ECI scores
# plot_acf(eci_df['ECI Score'], lags=30, ax=axes[0], color='#A78BFA', 
#          vlines_kwargs={'colors': '#A78BFA'})
# axes[0].set_title("ACF — Raw ECI Scores", fontsize=12, color='#111827', loc='left')
# axes[0].set_facecolor('white')
# axes[0].grid(True, color='#E5E7EB', linewidth=0.8)
# for spine in axes[0].spines.values():
#     spine.set_visible(False)

# # ACF on residuals (after removing linear trend)
# plot_acf(eci_df['residuals'], lags=30, ax=axes[1], color='#F472B6',
#          vlines_kwargs={'colors': '#F472B6'})
# axes[1].set_title("ACF — Residuals (after removing linear trend)", fontsize=12, color='#111827', loc='left')
# axes[1].set_facecolor('white')
# axes[1].grid(True, color='#E5E7EB', linewidth=0.8)
# for spine in axes[1].spines.values():
#     spine.set_visible(False)

# plt.suptitle("Autocorrelation Check — ECI Scores", fontsize=14, fontweight='500', 
#              color='#111827', y=1.01)
# plt.tight_layout()
# plt.savefig("acf_plot.png", dpi=150, bbox_inches='tight')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

eci_df['date_numeric'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
coeffs = np.polyfit(eci_df['date_numeric'], eci_df['ECI Score'], 1)
eci_df['residuals'] = eci_df['ECI Score'] - np.polyval(coeffs, eci_df['date_numeric'])

from scipy import stats

# OLS stats
x = eci_df['date_numeric'].values
y = eci_df['ECI Score'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

stats_output = f"""OLS Regression — ECI Score vs Time
=====================================
Beta (slope):        {slope:.6f}  (ECI points per day)
Intercept:           {intercept:.4f}
R-squared:           {r_value**2:.4f}
P-value:             {p_value:.2e}
Std Error (slope):   {std_err:.6f}

Interpretation
--------------
Beta annualized:     {slope * 365:.2f} ECI points per year
N models:            {len(eci_df)}
Date range:          {eci_df['Release date'].min().date()} to {eci_df['Release date'].max().date()}

ACF Diagnosis
-------------
Lag-1 autocorr (residuals): {eci_df['residuals'].autocorr(lag=1):.4f}
Lag-2 autocorr (residuals): {eci_df['residuals'].autocorr(lag=2):.4f}
Lag-3 autocorr (residuals): {eci_df['residuals'].autocorr(lag=3):.4f}

Note: Significant residual autocorrelation means OLS standard errors
are underestimated — this is the core frequentist flaw this project corrects.
"""

print(stats_output)

with open("ols_stats.txt", "w") as f:
    f.write(stats_output)

# --- Plot 1: ECI scatter with frontier line ---
top_orgs = ['OpenAI', 'Google', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
org_colors = {
    'OpenAI': '#F472B6',
    'Google': '#34D399',
    'Google DeepMind': '#34D399',
    'Meta AI': '#FB923C',
    'Anthropic': '#A78BFA',
    'xAI': '#60A5FA',
}
eci_df['color'] = eci_df['Organization'].apply(lambda o: org_colors.get(o, '#A8A29E'))

fig1, ax1 = plt.subplots(figsize=(13, 7))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

for org in top_orgs:
    subset = eci_df[eci_df['Organization'] == org]
    if len(subset):
        ax1.scatter(subset['Release date'], subset['ECI Score'],
                    color=org_colors[org], s=55, alpha=0.85,
                    label=org, zorder=3, linewidths=0)

other = eci_df[~eci_df['Organization'].isin(top_orgs)]
ax1.scatter(other['Release date'], other['ECI Score'],
            color='#A8A29E', s=55, alpha=0.7, label='Other', zorder=2, linewidths=0)

x_line = np.linspace(eci_df['date_numeric'].min(), eci_df['date_numeric'].max(), 300)
y_line = np.polyval(coeffs, x_line)
date_line = pd.to_datetime(eci_df['Release date'].min()) + pd.to_timedelta(x_line, unit='D')
ax1.plot(date_line, y_line, color='red', linewidth=1.5, alpha=0.7,
         label='OLS trend', zorder=1)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.set_ylim(55, 165)
ax1.set_ylabel("Score", fontsize=11, color='#374151')
ax1.set_xlabel("Release date", fontsize=11, color='#374151')
ax1.set_title("Epoch Capabilities Index (ECI)", fontsize=14,
               fontweight='500', loc='left', pad=12, color='#111827')
ax1.grid(True, axis='y', color='#E5E7EB', linewidth=0.8)
ax1.grid(True, axis='x', color='#E5E7EB', linewidth=0.8)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.legend(frameon=False, fontsize=9, loc='upper left',
           bbox_to_anchor=(1.01, 1), borderaxespad=0)

plt.tight_layout()
plt.savefig("eci_plot_with_frontier.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: ACF ---
fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
fig2.patch.set_facecolor('white')
fig2.suptitle("Autocorrelation Check — ECI Scores", fontsize=14,
               fontweight='500', color='#111827', y=0.98)  # fixed: y=0.98 not 1.01

plot_acf(eci_df['ECI Score'], lags=30, ax=axes[0],
         color='#A78BFA', vlines_kwargs={'colors': '#A78BFA'})
axes[0].set_title("ACF — Raw ECI Scores", fontsize=12, color='#111827', loc='left')
axes[0].set_facecolor('white')
axes[0].grid(True, color='#E5E7EB', linewidth=0.8)
for spine in axes[0].spines.values():
    spine.set_visible(False)

plot_acf(eci_df['residuals'], lags=30, ax=axes[1],
         color='#F472B6', vlines_kwargs={'colors': '#F472B6'})
axes[1].set_title("ACF — Residuals (after removing linear trend)",
                   fontsize=12, color='#111827', loc='left')
axes[1].set_facecolor('white')
axes[1].grid(True, color='#E5E7EB', linewidth=0.8)
for spine in axes[1].spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("acf_plot.png", dpi=150, bbox_inches='tight')
plt.show()