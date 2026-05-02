import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
print("Loading and grouping data for Phase 3...")
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

# Define focal groups vs Other
focal_orgs = ['OpenAI', 'Google DeepMind', 'Meta AI', 'Anthropic', 'xAI']
eci_df['Group'] = eci_df['Organization'].apply(lambda x: x if x in focal_orgs else 'Other')

# Split data: Focal groups get hierarchical treatment, "Other" gets global only
focal_df = eci_df[eci_df['Group'] != 'Other'].copy()
other_df = eci_df[eci_df['Group'] == 'Other'].copy().sort_values('Release date')

# Map focal groups to integer indices
group_names = focal_df['Group'].unique()
group_mapping = {name: idx for idx, name in enumerate(group_names)}
focal_df['Group_ID'] = focal_df['Group'].map(group_mapping)
J_focal = len(group_names)

# Pre-group focal data
grouped_focal = [focal_df[focal_df['Group_ID'] == j].sort_values('Release date') for j in range(J_focal)]

print(f"Data prepared. N_focal={len(focal_df)} models across {J_focal} groups. N_other={len(other_df)}.")

# ==========================================
# 2. Define Log-Posterior Functions
# ==========================================
def log_prior(theta):
    # theta array: [beta0, beta1, rho, sigma, tau0, tau1, b0_0..b0_J, b1_0..b1_J]
    beta0, beta1, rho, sigma, tau0, tau1 = theta[:6]
    b0 = theta[6 : 6+J_focal]
    b1 = theta[6+J_focal : 6+2*J_focal]
    
    # Constraints
    if not (-1 < rho < 1): return -np.inf
    if sigma <= 0 or tau0 <= 0 or tau1 <= 0: return -np.inf
        
    # Global Priors
    log_p_beta0 = stats.norm.logpdf(beta0, loc=0, scale=100)
    log_p_beta1 = stats.norm.logpdf(beta1, loc=0, scale=100)
    log_p_sigma = stats.invgamma.logpdf(sigma**2, a=2, scale=2)
    
    # Hierarchical Variance Priors (using Half-Normal for better mixing on variance components)
    log_p_tau0  = stats.halfnorm.logpdf(tau0, loc=0, scale=10)
    log_p_tau1  = stats.halfnorm.logpdf(tau1, loc=0, scale=0.1) # Slopes are small (daily)
    
    # Hierarchical Priors for Focal Groups
    log_p_b0 = np.sum(stats.norm.logpdf(b0, loc=0, scale=tau0))
    log_p_b1 = np.sum(stats.norm.logpdf(b1, loc=0, scale=tau1))
    
    return log_p_beta0 + log_p_beta1 + log_p_sigma + log_p_tau0 + log_p_tau1 + log_p_b0 + log_p_b1

def log_likelihood(theta, grouped_focal, other_df):
    beta0, beta1, rho, sigma = theta[:4]
    b0 = theta[6 : 6+J_focal]
    b1 = theta[6+J_focal : 6+2*J_focal]
    
    ll = 0
    
    # 1. Likelihood for Focal Groups (Global + Random Effects)
    for j in range(J_focal):
        df_j = grouped_focal[j]
        X_j = df_j['X_centered'].values
        Y_j = df_j['Y'].values
        
        # Varying intercept and varying slope applied here
        eps = Y_j - ((beta0 + b0[j]) + (beta1 + b1[j]) * X_j)
        
        if len(eps) > 1:
            ll += np.sum(stats.norm.logpdf(eps[1:], loc=rho * eps[:-1], scale=sigma))
        if len(eps) > 0:
            ll += stats.norm.logpdf(eps[0], loc=0, scale=sigma / np.sqrt(1 - rho**2))
            
    # 2. Likelihood for "Other" Group (Global only)
    if len(other_df) > 0:
        X_o = other_df['X_centered'].values
        Y_o = other_df['Y'].values
        
        # Only global intercept and slope applied here
        eps_o = Y_o - (beta0 + beta1 * X_o)
        
        if len(eps_o) > 1:
            ll += np.sum(stats.norm.logpdf(eps_o[1:], loc=rho * eps_o[:-1], scale=sigma))
        if len(eps_o) > 0:
            ll += stats.norm.logpdf(eps_o[0], loc=0, scale=sigma / np.sqrt(1 - rho**2))
            
    return ll

def log_posterior(theta, grouped_focal, other_df):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, grouped_focal, other_df)

# ==========================================
# 3. Random Walk Metropolis Sampler
# ==========================================
def metropolis_sampler_phase3(grouped_focal, other_df, start_theta, iters=35000, burn_in=10000, chain_id=1):
    print(f"  Starting Phase 3 Chain {chain_id} ({iters} iterations)...")
    
    n_params = 6 + (2 * J_focal)
    chain = np.zeros((iters, n_params))
    current_theta = start_theta.copy()
    
    step_sizes = np.array(
        [0.5, 0.002, 0.05, 0.1, 0.1, 0.005] +
        [0.5] * J_focal +
        [0.002] * J_focal
    )
    
    current_lp = log_posterior(current_theta, grouped_focal, other_df)
    accepted = 0
    
    for i in range(iters):
        proposal_theta = current_theta + np.random.normal(0, step_sizes, n_params)
        proposal_lp = log_posterior(proposal_theta, grouped_focal, other_df)
        log_alpha = proposal_lp - current_lp
        
        if np.log(np.random.rand()) < log_alpha:
            current_theta = proposal_theta
            current_lp = proposal_lp
            if i > burn_in:
                accepted += 1
                
        chain[i, :] = current_theta
        
        if i > 0 and i % 5000 == 0:
            print(f"  Chain {chain_id}: Iteration {i} complete...")
            
    acc_rate = accepted / (iters - burn_in)
    print(f"  Chain {chain_id} complete. Acceptance rate: {acc_rate:.2%}")
    return chain[burn_in:, :]



def run_multiple_chains_phase3(grouped_focal, other_df, n_chains=4, iters=40000, burn_in=15000):
    print(f"\nStarting {n_chains} chains ({iters} iterations each)...")
    
    slope_init, intercept_init, _, _, _ = stats.linregress(eci_df['X_centered'], eci_df['Y'])
    
    chains = []
    for c in range(n_chains):
        start_theta = np.zeros(6 + 2 * J_focal)
        start_theta[0] = intercept_init + np.random.normal(0, 10)    # beta0
        start_theta[1] = slope_init     + np.random.normal(0, 0.02)  # beta1
        start_theta[2] = np.random.uniform(-0.3, 0.3)                # rho
        start_theta[3] = np.random.uniform(2.0, 8.0)                 # sigma
        start_theta[4] = np.random.uniform(2.0, 8.0)                 # tau0
        start_theta[5] = np.random.uniform(0.005, 0.02)              # tau1 (small — daily slopes)
        start_theta[6 : 6+J_focal]          = np.random.normal(0, 5, J_focal)    # b0_j
        start_theta[6+J_focal : 6+2*J_focal] = np.random.normal(0, 0.005, J_focal) # b1_j

        chain = metropolis_sampler_phase3(
            grouped_focal, other_df, start_theta,
            iters=iters, burn_in=burn_in, chain_id=c+1
        )
        chains.append(chain)
    
    return np.array(chains)  # (n_chains, n_samples, n_params)


def calculate_gelman_rubin(chains):
    M, N, num_params = chains.shape
    r_hats = []
    for p in range(num_params):
        chain_samples = chains[:, :, p]
        W = np.mean(np.var(chain_samples, axis=1, ddof=1))
        chain_means = np.mean(chain_samples, axis=1)
        B = N / (M - 1) * np.sum((chain_means - np.mean(chain_means))**2)
        V = (N - 1) / N * W + B / N
        r_hats.append(np.sqrt(V / W))
    return np.array(r_hats)


multi_chain_trace = run_multiple_chains_phase3(grouped_focal, other_df, n_chains=4, iters=40000, burn_in=15000)
r_hat_stats = calculate_gelman_rubin(multi_chain_trace)
trace = multi_chain_trace.reshape(-1, multi_chain_trace.shape[2])
# ==========================================
# Gelman-Rubin Diagnostics
# ==========================================
global_param_names = ['beta0', 'beta1', 'rho', 'sigma', 'tau0', 'tau1']
b0_param_names = [f'b0_{group_names[j]}' for j in range(J_focal)]
b1_param_names = [f'b1_{group_names[j]}' for j in range(J_focal)]
all_param_names = global_param_names + b0_param_names + b1_param_names

print("\n=== Phase 3 Convergence Diagnostics (Gelman-Rubin) ===")
any_bad = False
for i, name in enumerate(all_param_names):
    status = "✅ Converged" if r_hat_stats[i] < 1.05 else "❌ Needs attention"
    if r_hat_stats[i] >= 1.05:
        any_bad = True
    print(f"{name:25} R-hat: {r_hat_stats[i]:.4f}  {status}")

if not any_bad:
    print("\n✅ All parameters converged (R-hat < 1.05)")
else:
    print("\n⚠️  Some parameters need attention — consider more iterations or tuning step sizes")
# ==========================================
# 4. Extracting and Summarizing Results
# ==========================================
global_samples = trace[:, :6]
b0_samples = trace[:, 6:6+J_focal]
b1_samples = trace[:, 6+J_focal:]

print("\n=== Phase 3 Global Posterior Summary ===")
names = ['beta0', 'beta1', 'rho', 'sigma', 'tau0', 'tau1']
for i, name in enumerate(names):
    mean_val = np.mean(global_samples[:, i])
    ci_l, ci_u = np.percentile(global_samples[:, i], [2.5, 97.5])
    print(f"{name:6}: Mean = {mean_val:8.4f} | 95% CI: [{ci_l:8.4f}, {ci_u:8.4f}]")

annual_global = np.mean(global_samples[:, 1]) * 365
print(f"\nGlobal Annualized progress (beta1 * 365): {annual_global:.2f} ECI points/year")
print("Note: The 'Other' category progresses exactly at this rate.")

print("\n--- Focal Group Specific Progress (Annualized) ---")
b1_annual_means = np.mean(b1_samples, axis=0) * 365
b1_annual_ci = np.percentile(b1_samples, [2.5, 97.5], axis=0) * 365
for i, name in enumerate(group_names):
    total_annual = annual_global + b1_annual_means[i]
    ci_l = annual_global + b1_annual_ci[0, i]
    ci_u = annual_global + b1_annual_ci[1, i]
    diff = b1_annual_means[i]
    print(f"{name:16}: {total_annual:6.2f} ECI/yr (Diff from global: {diff:+5.2f}) | 95% CI: [{ci_l:6.2f}, {ci_u:6.2f}]")

# ==========================================
# 5. Plotting Hierarchical Effects
# ==========================================
b0_means = np.mean(b0_samples, axis=0)
b0_ci = np.percentile(b0_samples, [2.5, 97.5], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Intercepts
sorted_idx_0 = np.argsort(b0_means)
axes[0].errorbar(b0_means[sorted_idx_0], range(J_focal), 
                 xerr=np.vstack([b0_means[sorted_idx_0] - b0_ci[0, sorted_idx_0], 
                                 b0_ci[1, sorted_idx_0] - b0_means[sorted_idx_0]]), 
                 fmt='o', color='#3B82F6', elinewidth=2, capsize=4, markersize=8)
axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
axes[0].set_yticks(range(J_focal))
axes[0].set_yticklabels([group_names[i] for i in sorted_idx_0])
axes[0].set_title("Varying Intercepts ($b_{0j}$)", fontsize=14)
axes[0].set_xlabel("Deviation from Global Baseline")
axes[0].grid(True, alpha=0.3)

# Slopes (Annualized)
sorted_idx_1 = np.argsort(b1_annual_means)
axes[1].errorbar(b1_annual_means[sorted_idx_1], range(J_focal), 
                 xerr=np.vstack([b1_annual_means[sorted_idx_1] - b1_annual_ci[0, sorted_idx_1], 
                                 b1_annual_ci[1, sorted_idx_1] - b1_annual_means[sorted_idx_1]]), 
                 fmt='o', color='#10B981', elinewidth=2, capsize=4, markersize=8)
axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
axes[1].set_yticks(range(J_focal))
axes[1].set_yticklabels([group_names[i] for i in sorted_idx_1])
axes[1].set_title("Varying Slopes (Annualized Diff from Global)", fontsize=14)
axes[1].set_xlabel("Deviation from Global Progress Rate (ECI/yr)")
axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.set_facecolor('white')
    for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig("phase3_varying_effects.png", dpi=150)
plt.show()

# ==========================================
# 6. Residual Diagnostics
# ==========================================
beta0_hat = np.mean(global_samples[:, 0])
beta1_hat = np.mean(global_samples[:, 1])
rho_hat   = np.mean(global_samples[:, 2])
b1_means  = np.mean(b1_samples, axis=0)

all_raw, all_ar1 = [], []

# Focal groups residuals
for j, df_j in enumerate(grouped_focal):
    X_j = df_j['X_centered'].values
    Y_j = df_j['Y'].values
    raw = Y_j - ((beta0_hat + b0_means[j]) + (beta1_hat + b1_means[j]) * X_j)
    ar1 = raw[1:] - rho_hat * raw[:-1]
    all_raw.extend(raw)
    all_ar1.extend(ar1)

# Other group residuals (Global only)
if len(other_df) > 0:
    X_o = other_df['X_centered'].values
    Y_o = other_df['Y'].values
    raw_o = Y_o - (beta0_hat + beta1_hat * X_o)
    ar1_o = raw_o[1:] - rho_hat * raw_o[:-1]
    all_raw.extend(raw_o)
    all_ar1.extend(ar1_o)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('white')
fig2.suptitle("Phase 3 Residual Autocorrelation", fontsize=14)

plot_acf(all_raw, lags=20, ax=axes2[0], color='#60A5FA')
axes2[0].set_title("ACF: Raw Residuals (Full Hierarchical)")
axes2[0].grid(True, alpha=0.3)

plot_acf(all_ar1, lags=20, ax=axes2[1], color='#34D399')
axes2[1].set_title("ACF: AR(1) Corrected Residuals")
axes2[1].grid(True, alpha=0.3)

for ax in axes2:
    ax.set_facecolor('white')
    for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig("phase3_acf_diagnostics.png", dpi=150)
plt.show()

lb_raw = acorr_ljungbox(all_raw, lags=[10], return_df=True)
lb_ar1 = acorr_ljungbox(all_ar1, lags=[10], return_df=True)
print(f"Ljung-Box p-value — Raw Phase 3:   {lb_raw['lb_pvalue'].values[0]:.4f}")
print(f"Ljung-Box p-value — AR(1) Phase 3: {lb_ar1['lb_pvalue'].values[0]:.4f}")

# Add this to phase3.py after data prep
focal_all = pd.concat(grouped_focal)
slope_ols, intercept_ols, r, p, se = stats.linregress(
    focal_all['X_centered'], focal_all['Y']
)
print(f"OLS focal-only: {slope_ols*365:.2f} ECI/yr (SE: {se*365:.2f})")
print(f"Bayesian hierarchical global: {annual_global:.2f} ECI/yr")
print(f"Inflation estimate: {slope_ols*365 - annual_global:.2f} ECI/yr")

# Add this after your focal group summary
print("\n=== Key Comparison for Research Question ===")
print(f"OLS focal-only (pooled, no org structure): {slope_ols*365:.2f} ECI/yr")
print(f"\nBayesian org-specific rates:")
for i, name in enumerate(group_names):
    total = annual_global + b1_annual_means[i]
    print(f"  {name:16}: {total:.2f} ECI/yr")

# Weighted average of org-specific Bayesian rates
org_sizes = [len(grouped_focal[j]) for j in range(J_focal)]
weighted_bayes = np.average(
    [annual_global + b1_annual_means[i] for i in range(J_focal)],
    weights=org_sizes
)
print(f"\nWeighted avg Bayesian org rate: {weighted_bayes:.2f} ECI/yr")
print(f"OLS focal pooled:               {slope_ols*365:.2f} ECI/yr")
print(f"Inflation (OLS - weighted avg): {slope_ols*365 - weighted_bayes:.2f} ECI/yr")

# Decompose the inflation sources
print("\n=== Inflation Decomposition ===")
print("How much does each org's baseline advantage distort pooled OLS?\n")

beta0_hat = np.mean(global_samples[:, 0])
b0_means  = np.mean(b0_samples, axis=0)

for i, name in enumerate(group_names):
    n_j = len(grouped_focal[i])
    share = n_j / len(focal_df)
    baseline_advantage = b0_means[i]
    slope_diff = b1_annual_means[i]
    print(f"{name:16}: n={n_j:3d} ({share:.0%}) | "
          f"baseline b0={baseline_advantage:+.1f} | "
          f"slope diff={slope_diff:+.1f} ECI/yr")

# Correlation between baseline and slope deviations
print(f"\nCorrelation(b0, b1): {np.corrcoef(b0_means, np.mean(b1_samples,axis=0)*365)[0,1]:.3f}")
print("Negative = orgs with higher baselines have slower slopes (OLS conflates these)")

# Posterior probability that b1_OpenAI < 0 (slope genuinely below global)
openai_idx = list(group_names).index('OpenAI')
anthropic_idx = list(group_names).index('Anthropic')

p_openai_below = np.mean(b1_samples[:, openai_idx] < 0)
p_anthropic_below = np.mean(b1_samples[:, anthropic_idx] < 0)

print(f"P(OpenAI slope < global):    {p_openai_below:.3f}")
print(f"P(Anthropic slope < global): {p_anthropic_below:.3f}")


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

# Phase 3 - 6 global params
param_labels_p3 = [r'$\beta_0$', r'$\beta_1$', r'$\rho$', r'$\sigma$', r'$\tau_0$', r'$\tau_1$']
fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
fig.suptitle("Phase 3 Trace Plots (Global Parameters)", fontsize=14)
for i in range(6):
    axes[i].plot(trace[:, i], color='#60A5FA', alpha=0.6, linewidth=0.5)
    axes[i].set_ylabel(param_labels_p3[i])
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("Post-Burnin Iteration")
plt.tight_layout()
plt.savefig("phase3_traces.png", dpi=150)

# Add SD column to your summary print
for i, name in enumerate(names):
    mean_val = np.mean(global_samples[:, i])
    sd_val   = np.std(global_samples[:, i])
    ci_l, ci_u = np.percentile(global_samples[:, i], [2.5, 97.5])
    print(f"{name:6}: Mean={mean_val:8.4f} SD={sd_val:.4f} | 95% CI:[{ci_l:.4f}, {ci_u:.4f}]")