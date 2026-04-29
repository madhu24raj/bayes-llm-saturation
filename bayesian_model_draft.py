import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── 1. LOAD & CENTER ──────────────────────────────────────────────────────────
eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# Center X_ij — removes intercept/slope correlation
eci_df['date_numeric'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
x_mean = eci_df['date_numeric'].mean()
eci_df['X_centered'] = eci_df['date_numeric'] - x_mean

# ── 2. OLS PER ORGANIZATION ───────────────────────────────────────────────────
top_orgs = ['OpenAI', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
org_colors = {
    'OpenAI': '#F472B6',
    'Google DeepMind': '#2DD4BF',
    'Meta AI': '#FB923C',
    'Anthropic': '#A78BFA',
    'xAI': '#60A5FA',
}

print("=" * 50)
print("OLS PER ORGANIZATION (centered X)")
print("=" * 50)

org_results = {}
for org in top_orgs:
    subset = eci_df[eci_df['Organization'] == org].copy()
    if len(subset) < 4:
        continue
    slope, intercept, r, p, se = stats.linregress(
        subset['X_centered'], subset['ECI Score']
    )
    subset['residual'] = subset['ECI Score'] - (
        slope * subset['X_centered'] + intercept
    )
    lag1 = subset['residual'].autocorr(lag=1)
    org_results[org] = {
        'slope': slope, 'intercept': intercept,
        'r2': r**2, 'p': p, 'se': se,
        'n': len(subset), 'lag1_acf': lag1,
        'residuals': subset['residual'].values
    }
    print(f"\n{org} (n={len(subset)})")
    print(f"  Slope:      {slope:.5f} ECI/day  ({slope*365:.2f}/yr)")
    print(f"  Intercept:  {intercept:.3f}")
    print(f"  R²:         {r**2:.3f}")
    print(f"  P-value:    {p:.2e}")
    print(f"  Lag-1 ACF:  {lag1:.3f}")

# ── 3. PLOT PER-ORG RESIDUALS ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(org_results), 1,
                          figsize=(12, 2.8 * len(org_results)))
fig.patch.set_facecolor('white')
fig.suptitle("OLS Residuals per Organization",
             fontsize=13, fontweight='bold', color='#111827', y=1.01)

for ax, (org, res) in zip(axes, org_results.items()):
    ax.axhline(0, color='#9CA3AF', linewidth=0.8, linestyle='--')
    ax.bar(range(len(res['residuals'])), res['residuals'],
           color=org_colors.get(org, '#C4A882'), alpha=0.7, width=0.6)
    ax.set_title(
        f"{org}  |  β₁={res['slope']*365:.2f}/yr  "
        f"R²={res['r2']:.2f}  Lag-1 ACF={res['lag1_acf']:.2f}",
        fontsize=9, loc='left', color='#374151'
    )
    ax.set_ylabel("Residual", fontsize=8, color='#6B7280')
    ax.set_facecolor('white')
    ax.grid(True, color='#E5E7EB', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig("org_residuals.png", dpi=150, bbox_inches='tight')
plt.show()

# ── 4. BAYESIAN SETUP ─────────────────────────────────────────────────────────
# Starting point: global model with priors on beta_0, beta_1, sigma^2
# Y_i = beta_0 + beta_1 * X_centered_i + epsilon_i
# Prior: beta_0, beta_1 ~ N(0, 100)   (weakly informative)
#        sigma^2 ~ Inverse-Gamma(a0, b0)

y = eci_df['ECI Score'].values
x = eci_df['X_centered'].values
n = len(y)

# Weakly informative priors
mu_0    = 0       # prior mean on betas
tau_0   = 100     # prior variance on betas (wide = weakly informative)
a_0     = 1       # Inverse-Gamma shape for sigma^2
b_0     = 1       # Inverse-Gamma scale for sigma^2

print("\n" + "=" * 50)
print("BAYESIAN PRIOR SETUP")
print("=" * 50)
print(f"  beta_0 prior: N({mu_0}, {tau_0})")
print(f"  beta_1 prior: N({mu_0}, {tau_0})")
print(f"  sigma^2 prior: Inv-Gamma({a_0}, {b_0})")
print(f"  n observations: {n}")
print(f"  X centered at: day {x_mean:.1f}")
print("\n  Next step: Gibbs sampler for beta_0, beta_1, sigma^2")
print("  Then: add rho ~ Uniform(-1, 1) with Metropolis step")