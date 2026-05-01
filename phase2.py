import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
print("Loading and grouping data...")
eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# Centering X
eci_df['days'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
X_raw = eci_df['days'].values
X_mean = np.mean(X_raw)
eci_df['X_centered'] = X_raw - X_mean
eci_df['Y'] = eci_df['ECI Score'].values

# Create groups
top_orgs = ['OpenAI', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
eci_df['Group'] = eci_df['Organization'].apply(lambda x: x if x in top_orgs else 'Other')

# Map groups to integer indices
group_names = eci_df['Group'].unique()
group_mapping = {name: idx for idx, name in enumerate(group_names)}
eci_df['Group_ID'] = eci_df['Group'].map(group_mapping)
J = len(group_names)

# Pre-group the data to speed up the likelihood calculation loop
grouped_data = [eci_df[eci_df['Group_ID'] == j].sort_values('Release date') for j in range(J)]

print(f"Data prepared. N={len(eci_df)} models across J={J} groups.")

# ==========================================
# 2. Define Log-Posterior Functions
# ==========================================
def log_prior(theta):
    # theta array: [beta0, beta1, rho, sigma, tau0, b0_1, b0_2, ..., b0_J]
    beta0, beta1, rho, sigma, tau0 = theta[:5]
    b0 = theta[5:]
    
    # Constraints
    if not (-1 < rho < 1): return -np.inf
    if sigma <= 0: return -np.inf
    if tau0 <= 0: return -np.inf
        
    # Global Priors
    log_p_beta0 = stats.norm.logpdf(beta0, loc=0, scale=100)
    log_p_beta1 = stats.norm.logpdf(beta1, loc=0, scale=100)
    log_p_sigma = stats.invgamma.logpdf(sigma**2, a=2, scale=2)
    log_p_tau0  = stats.invgamma.logpdf(tau0**2, a=2, scale=2)
    
    # Hierarchical Prior for Group Intercepts: b0_j ~ N(0, tau0^2)
    log_p_b0 = np.sum(stats.norm.logpdf(b0, loc=0, scale=tau0))
    
    return log_p_beta0 + log_p_beta1 + log_p_sigma + log_p_tau0 + log_p_b0

def log_likelihood(theta, grouped_data):
    beta0, beta1, rho, sigma, tau0 = theta[:5]
    b0 = theta[5:]
    
    ll = 0
    for j in range(len(grouped_data)):
        df_j = grouped_data[j]
        X_j = df_j['X_centered'].values
        Y_j = df_j['Y'].values
        
        eps = Y_j - (beta0 + b0[j] + beta1 * X_j)
        
        if len(eps) > 1:
            ll += np.sum(stats.norm.logpdf(eps[1:], loc=rho * eps[:-1], scale=sigma))
        if len(eps) > 0:
            ll += stats.norm.logpdf(eps[0], loc=0, scale=sigma / np.sqrt(1 - rho**2))
            
    return ll

def log_posterior(theta, grouped_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, grouped_data)

# ==========================================
# 3. Random Walk Metropolis Sampler
# ==========================================
def metropolis_sampler_hierarchical(grouped_data, iters=25000, burn_in=5000):
    print(f"Starting Phase 2 Sampler ({iters} iterations)...")
    
    n_params = 5 + J
    chain = np.zeros((iters, n_params))
    
    X_all = pd.concat([df['X_centered'] for df in grouped_data]).values
    Y_all = pd.concat([df['Y'] for df in grouped_data]).values
    slope_init, intercept_init, _, _, _ = stats.linregress(X_all, Y_all)
    
    current_theta = np.zeros(n_params)
    current_theta[0] = intercept_init  # beta0
    current_theta[1] = slope_init      # beta1
    current_theta[2] = 0.1             # rho
    current_theta[3] = 5.0             # sigma
    current_theta[4] = 5.0             # tau0
    # b0[0..J-1] stay 0.0
    
    step_sizes = np.array([0.5, 0.002, 0.05, 0.1, 0.1] + [0.5]*J)
    
    current_lp = log_posterior(current_theta, grouped_data)
    accepted = 0
    
    for i in range(iters):
        proposal_theta = current_theta + np.random.normal(0, step_sizes, n_params)
        proposal_lp = log_posterior(proposal_theta, grouped_data)
        log_alpha = proposal_lp - current_lp
        
        if np.log(np.random.rand()) < log_alpha:
            current_theta = proposal_theta
            current_lp = proposal_lp
            if i > burn_in:
                accepted += 1
                
        chain[i, :] = current_theta
        
        if i > 0 and i % 5000 == 0:
            print(f"Iteration {i} complete...")
            
    acc_rate = accepted / (iters - burn_in)
    print(f"Sampling complete. Acceptance rate (post burn-in): {acc_rate:.2%}")
    return chain[burn_in:, :]

# Run sampler (increased iterations to handle hierarchical structure)
trace = metropolis_sampler_hierarchical(grouped_data, iters=30000, burn_in=10000)

# ==========================================
# 4. Plotting Global Posteriors & Varying Intercepts
# ==========================================
global_names = [r'Global Intercept ($\beta_0$)', r'Global Slope ($\beta_1$)', 
                r'AR(1) Coef ($\rho$)', r'Residual Std ($\sigma$)', r'Varying Int. Std ($\tau_0$)']

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.patch.set_facecolor('white')
fig.suptitle("Phase 2 Posteriors: Global Parameters", fontsize=16, y=0.98)
axes = axes.flatten()

for i in range(5):
    axes[i].hist(trace[:, i], bins=50, color='#A78BFA', density=True, alpha=0.8, edgecolor='white')
    axes[i].set_title(global_names[i])
    median_val = np.median(trace[:, i])
    axes[i].axvline(median_val, color='#111827', linestyle='dashed', linewidth=1.5, 
                    label=f'Median: {median_val:.4f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[5].axis('off') # Hide the empty 6th subplot
plt.tight_layout()
plt.savefig("phase2_global_posteriors.png", dpi=150)
plt.show()

# --- Plot Group-Specific Intercepts (Caterpillar Plot) ---
b0_samples = trace[:, 5:]
b0_means = np.mean(b0_samples, axis=0)
b0_ci = np.percentile(b0_samples, [2.5, 97.5], axis=0)

# Sort by mean capability for better visualization
sorted_idx = np.argsort(b0_means)
sorted_names = [group_names[i] for i in sorted_idx]
sorted_means = b0_means[sorted_idx]
sorted_errors = np.vstack([sorted_means - b0_ci[0, sorted_idx], 
                           b0_ci[1, sorted_idx] - sorted_means])

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

ax2.errorbar(sorted_means, range(J), xerr=sorted_errors, fmt='o', 
             color='#FB923C', ecolor='#FCA5A5', elinewidth=2, capsize=4, markersize=8)
ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_yticks(range(J))
ax2.set_yticklabels(sorted_names)
ax2.set_xlabel("Deviation from Global Intercept ($b_{0j}$)")
ax2.set_title("Varying Intercepts by Organization (95% CI)", fontsize=14)
ax2.grid(True, alpha=0.3)
for spine in ax2.spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig("phase2_varying_intercepts.png", dpi=150)
plt.show()

# ==========================================
# 5. Output Summary
# ==========================================
print("\n=== Phase 2 Posterior Summary ===")
for i, name in enumerate(['beta0', 'beta1', 'rho', 'sigma', 'tau0']):
    mean_val = np.mean(trace[:, i])
    ci_lower, ci_upper = np.percentile(trace[:, i], [2.5, 97.5])
    print(f"{name:5}: Mean = {mean_val:8.4f} | 95% CI: [{ci_lower:8.4f}, {ci_upper:8.4f}]")

annual_slope = np.mean(trace[:, 1]) * 365
print(f"\nGlobal Annualized progress (beta1 * 365): {annual_slope:.2f} ECI points/year")

print("\n--- Group-Specific Baseline Advantages (b0_j) ---")
for i in sorted_idx[::-1]: # Print highest to lowest
    name = group_names[i]
    mean_val = b0_means[i]
    ci_l, ci_u = b0_ci[0, i], b0_ci[1, i]
    print(f"{name:16}: {mean_val:6.2f} ECI points (95% CI: [{ci_l:6.2f}, {ci_u:6.2f}])")


# ==========================================
# 6. Phase 2 Residual Diagnostics (by group)
# ==========================================
from statsmodels.graphics.tsaplots import plot_acf

beta0_hat = np.mean(trace[:, 0])
beta1_hat = np.mean(trace[:, 1])
rho_hat   = np.mean(trace[:, 2])
b0_means  = np.mean(trace[:, 5:], axis=0)

group_colors = {
    'OpenAI': '#60A5FA', 'Anthropic': '#F87171', 
    'Google DeepMind': '#34D399', 'xAI': '#FBBF24', 
    'Meta AI': '#A78BFA', 'Other': '#94A3B8'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 2 Residual Diagnostics", fontsize=14)

all_raw, all_ar1, all_dates, all_groups = [], [], [], []

for j, df_j in enumerate(grouped_data):
    X_j = df_j['X_centered'].values
    Y_j = df_j['Y'].values
    name = group_names[j]
    
    raw = Y_j - (beta0_hat + b0_means[j] + beta1_hat * X_j)
    ar1 = raw[1:] - rho_hat * raw[:-1]
    
    all_raw.extend(raw)
    all_ar1.extend(ar1)
    all_dates.extend(df_j['Release date'].values)
    all_groups.extend([name] * len(raw))
    
    color = group_colors.get(name, '#94A3B8')
    axes[1, 0].scatter(df_j['Release date'].values, raw, 
                       alpha=0.6, color=color, s=25, label=name)
    axes[1, 1].scatter(df_j['Release date'].values[1:], ar1, 
                       alpha=0.6, color=color, s=25, label=name)

# ACF plots on pooled residuals
plot_acf(all_raw, lags=20, ax=axes[0, 0], color='#60A5FA')
axes[0, 0].set_title("ACF: Raw Residuals (Phase 2)")

plot_acf(all_ar1, lags=20, ax=axes[0, 1], color='#34D399')
axes[0, 1].set_title("ACF: AR(1) Corrected Residuals (Phase 2)")

axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_title("Raw Residuals over Time (by org)")
axes[1, 0].legend(fontsize=8, loc='upper right')

axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].set_title("AR(1) Corrected Residuals over Time (by org)")
axes[1, 1].legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig("phase2_residual_diagnostics.png", dpi=150)
plt.show()

# Ljung-Box
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_raw = acorr_ljungbox(all_raw, lags=[10], return_df=True)
lb_ar1 = acorr_ljungbox(all_ar1, lags=[10], return_df=True)
print(f"Ljung-Box p-value — Raw:         {lb_raw['lb_pvalue'].values[0]:.4f}")
print(f"Ljung-Box p-value — AR(1):       {lb_ar1['lb_pvalue'].values[0]:.4f}")


# Trace plots - global params only (don't need all b0/b1 traces)
param_labels_p2 = [r'$\beta_0$', r'$\beta_1$', r'$\rho$', r'$\sigma$', r'$\tau_0$']
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Phase 2 Trace Plots (Global Parameters)", fontsize=14)
for i in range(5):
    axes[i].plot(trace[:, i], color='#A78BFA', alpha=0.6, linewidth=0.5)
    axes[i].set_ylabel(param_labels_p2[i])
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("Post-Burnin Iteration")
plt.tight_layout()
plt.savefig("phase2_traces.png", dpi=150)

# Add SD column to your summary print
print("\n=== Phase 2 Posterior Summary with SDs ===")
p2_names = ['beta0', 'beta1', 'rho', 'sigma', 'tau0']
for i, name in enumerate(p2_names):
    mean_val = np.mean(trace[:, i])
    sd_val   = np.std(trace[:, i])
    ci_l, ci_u = np.percentile(trace[:, i], [2.5, 97.5])
    print(f"{name:6}: Mean={mean_val:8.4f} SD={sd_val:.4f} | 95% CI:[{ci_l:.4f}, {ci_u:.4f}]")