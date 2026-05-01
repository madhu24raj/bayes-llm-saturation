# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np
# from statsmodels.graphics.tsaplots import plot_acf

# eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
# eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
# eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
# eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# eci_df['date_numeric'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
# coeffs = np.polyfit(eci_df['date_numeric'], eci_df['ECI Score'], 1)
# eci_df['residuals'] = eci_df['ECI Score'] - np.polyval(coeffs, eci_df['date_numeric'])

# from scipy import stats

# # OLS stats
# x = eci_df['date_numeric'].values
# y = eci_df['ECI Score'].values
# slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# stats_output = f"""OLS Regression — ECI Score vs Time
# =====================================
# Beta (slope):        {slope:.6f}  (ECI points per day)
# Intercept:           {intercept:.4f}
# R-squared:           {r_value**2:.4f}
# P-value:             {p_value:.2e}
# Std Error (slope):   {std_err:.6f}

# Interpretation
# --------------
# Beta annualized:     {slope * 365:.2f} ECI points per year
# N models:            {len(eci_df)}
# Date range:          {eci_df['Release date'].min().date()} to {eci_df['Release date'].max().date()}

# ACF Diagnosis
# -------------
# Lag-1 autocorr (residuals): {eci_df['residuals'].autocorr(lag=1):.4f}
# Lag-2 autocorr (residuals): {eci_df['residuals'].autocorr(lag=2):.4f}
# Lag-3 autocorr (residuals): {eci_df['residuals'].autocorr(lag=3):.4f}

# Note: Significant residual autocorrelation means OLS standard errors
# are underestimated — this is the core frequentist flaw this project corrects.
# """

# print(stats_output)

# with open("ols_stats.txt", "w") as f:
#     f.write(stats_output)

# # --- Plot 1: ECI scatter with frontier line ---
# top_orgs = ['OpenAI', 'Google', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
# org_colors = {
#     'OpenAI': '#F472B6',
#     'Google': '#34D399',
#     'Google DeepMind': '#34D399',
#     'Meta AI': '#FB923C',
#     'Anthropic': '#A78BFA',
#     'xAI': '#60A5FA',
# }
# eci_df['color'] = eci_df['Organization'].apply(lambda o: org_colors.get(o, '#A8A29E'))

# fig1, ax1 = plt.subplots(figsize=(13, 7))
# fig1.patch.set_facecolor('white')
# ax1.set_facecolor('white')

# for org in top_orgs:
#     subset = eci_df[eci_df['Organization'] == org]
#     if len(subset):
#         ax1.scatter(subset['Release date'], subset['ECI Score'],
#                     color=org_colors[org], s=55, alpha=0.85,
#                     label=org, zorder=3, linewidths=0)

# other = eci_df[~eci_df['Organization'].isin(top_orgs)]
# ax1.scatter(other['Release date'], other['ECI Score'],
#             color='#A8A29E', s=55, alpha=0.7, label='Other', zorder=2, linewidths=0)

# x_line = np.linspace(eci_df['date_numeric'].min(), eci_df['date_numeric'].max(), 300)
# y_line = np.polyval(coeffs, x_line)
# date_line = pd.to_datetime(eci_df['Release date'].min()) + pd.to_timedelta(x_line, unit='D')
# ax1.plot(date_line, y_line, color='red', linewidth=1.5, alpha=0.7,
#          label='OLS trend', zorder=1)

# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
# ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
# ax1.set_ylim(55, 165)
# ax1.set_ylabel("Score", fontsize=11, color='#374151')
# ax1.set_xlabel("Release date", fontsize=11, color='#374151')
# ax1.set_title("Epoch Capabilities Index (ECI)", fontsize=14,
#                fontweight='500', loc='left', pad=12, color='#111827')
# ax1.grid(True, axis='y', color='#E5E7EB', linewidth=0.8)
# ax1.grid(True, axis='x', color='#E5E7EB', linewidth=0.8)
# ax1.set_axisbelow(True)
# for spine in ax1.spines.values():
#     spine.set_visible(False)
# ax1.legend(frameon=False, fontsize=9, loc='upper left',
#            bbox_to_anchor=(1.01, 1), borderaxespad=0)

# plt.tight_layout()
# plt.savefig("eci_plot_with_frontier.png", dpi=150, bbox_inches='tight')
# plt.show()

# # --- Plot 2: ACF ---
# fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
# fig2.patch.set_facecolor('white')
# fig2.suptitle("Autocorrelation Check — ECI Scores", fontsize=14,
#                fontweight='500', color='#111827', y=0.98)  # fixed: y=0.98 not 1.01

# plot_acf(eci_df['ECI Score'], lags=30, ax=axes[0],
#          color='#A78BFA', vlines_kwargs={'colors': '#A78BFA'})
# axes[0].set_title("ACF — Raw ECI Scores", fontsize=12, color='#111827', loc='left')
# axes[0].set_facecolor('white')
# axes[0].grid(True, color='#E5E7EB', linewidth=0.8)
# for spine in axes[0].spines.values():
#     spine.set_visible(False)

# plot_acf(eci_df['residuals'], lags=30, ax=axes[1],
#          color='#F472B6', vlines_kwargs={'colors': '#F472B6'})
# axes[1].set_title("ACF — Residuals (after removing linear trend)",
#                    fontsize=12, color='#111827', loc='left')
# axes[1].set_facecolor('white')
# axes[1].grid(True, color='#E5E7EB', linewidth=0.8)
# for spine in axes[1].spines.values():
#     spine.set_visible(False)

# plt.tight_layout()
# plt.savefig("acf_plot.png", dpi=150, bbox_inches='tight')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

# ==========================================
# Data Loading and Preprocessing
# ==========================================
eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

eci_df['date_numeric'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days

# ==========================================
# ITEM 3: Statistical Output (Overall & Groups)
# ==========================================
# Overall OLS
coeffs = np.polyfit(eci_df['date_numeric'], eci_df['ECI Score'], 1)
eci_df['overall_residuals'] = eci_df['ECI Score'] - np.polyval(coeffs, eci_df['date_numeric'])

x = eci_df['date_numeric'].values
y = eci_df['ECI Score'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

stats_output = f"""Overall OLS Regression — ECI Score vs Time
==========================================
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

ACF Diagnosis (Overall)
-----------------------
Lag-1 autocorr (residuals): {eci_df['overall_residuals'].autocorr(lag=1):.4f}
Lag-2 autocorr (residuals): {eci_df['overall_residuals'].autocorr(lag=2):.4f}
Lag-3 autocorr (residuals): {eci_df['overall_residuals'].autocorr(lag=3):.4f}

Group-Specific OLS Regression
==========================================
"""

top_orgs = ['OpenAI', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
org_colors = {
    'OpenAI': '#F472B6',
    'Google DeepMind': '#34D399',
    'Meta AI': '#FB923C',
    'Anthropic': '#A78BFA',
    'xAI': '#60A5FA',
    'Other': '#A8A29E'
}

# Group mapping
eci_df['Group'] = eci_df['Organization'].apply(lambda o: o if o in top_orgs else 'Other')
groups = top_orgs + ['Other']
group_models = {}

# Group OLS
for group in groups:
    subset = eci_df[eci_df['Group'] == group].copy()
    if len(subset) < 2:
        continue 
        
    x_g = subset['date_numeric'].values
    y_g = subset['ECI Score'].values
    slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(x_g, y_g)
    
    # Calculate group-specific residuals
    subset['group_residuals'] = y_g - (slope_g * x_g + intercept_g)
    
    group_models[group] = {
        'slope': slope_g,
        'intercept': intercept_g,
        'subset': subset,
        'color': org_colors[group]
    }
    
    stats_output += f"{group.ljust(16)} | Beta (ann.): {slope_g * 365:6.2f} ECI/yr | R2: {r_value_g**2:.4f} | p-val: {p_value_g:.2e} | N: {len(subset):2d}\n"

print(stats_output)
with open("ols_stats_and_pvals.txt", "w") as f:
    f.write(stats_output)

# Common styling variables for plots
x_line_all = np.linspace(eci_df['date_numeric'].min(), eci_df['date_numeric'].max(), 300)
y_line_all = np.polyval(coeffs, x_line_all)
date_line_all = pd.to_datetime(eci_df['Release date'].min()) + pd.to_timedelta(x_line_all, unit='D')

# ==========================================
# ITEM 1: Regular plot with Overall OLS ONLY
# ==========================================
fig1, ax1 = plt.subplots(figsize=(13, 7))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Scatter all points (grouped for coloring)
for group in groups:
    if group in group_models:
        subset = group_models[group]['subset']
        color = group_models[group]['color']
        ax1.scatter(subset['Release date'], subset['ECI Score'], color=color, 
                    s=55, alpha=0.85 if group != 'Other' else 0.5, label=group, linewidths=0)

# Overall OLS Line
ax1.plot(date_line_all, y_line_all, color='black', linewidth=2, label='Overall OLS trend', zorder=5)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.set_ylim(55, 165)
ax1.set_ylabel("Score", fontsize=11)
ax1.set_xlabel("Release date", fontsize=11)
ax1.set_title("ECI Scores with Overall OLS Trend", fontsize=14, fontweight='500', loc='left')
ax1.grid(True, color='#E5E7EB', linewidth=0.8)
for spine in ax1.spines.values(): spine.set_visible(False)
ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))

plt.tight_layout()
plt.savefig("1_eci_overall_ols_only.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# ITEM 2: Regular plot with Overall AND Group OLS
# ==========================================
fig2, ax2 = plt.subplots(figsize=(13, 7))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

# Scatter points and Group OLS lines
for group in groups:
    if group in group_models:
        model = group_models[group]
        subset = model['subset']
        color = model['color']
        
        ax2.scatter(subset['Release date'], subset['ECI Score'], color=color, 
                    s=55, alpha=0.85 if group != 'Other' else 0.5, label=group, linewidths=0)
        
        x_line_g = np.linspace(subset['date_numeric'].min(), subset['date_numeric'].max(), 100)
        y_line_g = model['slope'] * x_line_g + model['intercept']
        date_line_g = pd.to_datetime(eci_df['Release date'].min()) + pd.to_timedelta(x_line_g, unit='D')
        ax2.plot(date_line_g, y_line_g, color=color, linewidth=2.5, alpha=0.9)

# Overall OLS Line (Dashed)
ax2.plot(date_line_all, y_line_all, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Overall OLS trend', zorder=1)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax2.set_ylim(55, 165)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_xlabel("Release date", fontsize=11)
ax2.set_title("ECI Scores with Group-Specific OLS Trends", fontsize=14, fontweight='500', loc='left')
ax2.grid(True, color='#E5E7EB', linewidth=0.8)
for spine in ax2.spines.values(): spine.set_visible(False)
ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))

plt.tight_layout()
plt.savefig("2_eci_group_and_overall_ols.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# ITEM 4: ACF Plot for Overall Raw & Overall Residuals
# ==========================================
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
fig3.patch.set_facecolor('white')
fig3.suptitle("Autocorrelation Check — Overall Model", fontsize=14, fontweight='500', y=0.98)

# Fix: Passing axes3 and axes3 individually instead of the whole axes3 array
plot_acf(eci_df['ECI Score'], lags=30, ax=axes3, color='#A78BFA', vlines_kwargs={'colors': '#A78BFA'})
axes3.set_title("ACF — Raw Overall ECI Scores", fontsize=12, loc='left')
axes3.grid(True, color='#E5E7EB', linewidth=0.8)

plot_acf(eci_df['overall_residuals'], lags=30, ax=axes3, color='#F472B6', vlines_kwargs={'colors': '#F472B6'})
axes3.set_title("ACF — Overall Residuals (After Linear Trend Removal)", fontsize=12, loc='left')
axes3.grid(True, color='#E5E7EB', linewidth=0.8)

for ax in axes3:
    ax.set_facecolor('white')
    for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig("3_acf_overall.png", dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# ITEM 5: ACF Plot for Group-Specific Residuals
# ==========================================
num_groups = len(group_models)
cols = 2
rows = int(np.ceil(num_groups / cols))

fig4, axes4 = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
fig4.patch.set_facecolor('white')
fig4.suptitle("Autocorrelation Check — Group-Specific Residuals", fontsize=14, fontweight='500', y=1.02)
axes4 = axes4.flatten()

for idx, group in enumerate(group_models.keys()):
    ax = axes4[idx]
    model = group_models[group]
    residuals = model['subset']['group_residuals']
    color = model['color']
    
    max_lags = min(20, len(residuals) - 2) 
    
    if max_lags > 0:
        plot_acf(residuals, lags=max_lags, ax=ax, color=color, vlines_kwargs={'colors': color})
        ax.set_title(f"ACF — {group} Residuals (N={len(residuals)})", fontsize=11, loc='left')
    else:
        ax.text(0.5, 0.5, f"Not enough data for ACF (N={len(residuals)})", 
                ha='center', va='center', color='#6B7280', transform=ax.transAxes)
        ax.set_title(f"ACF — {group} Residuals", fontsize=11, loc='left')
        
    ax.set_facecolor('white')
    ax.grid(True, color='#E5E7EB', linewidth=0.8)
    for spine in ax.spines.values(): spine.set_visible(False)

# Hide any empty subplots
for i in range(idx + 1, len(axes4)):
    axes4[i].axis('off')

plt.tight_layout()
plt.savefig("4_acf_groups.png", dpi=150, bbox_inches='tight')
plt.show()