import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
print("Loading and centering data...")
eci_df = pd.read_csv("benchmark_data/epoch_capabilities_index.csv")
eci_df['Release date'] = pd.to_datetime(eci_df['Release date'], errors='coerce')
eci_df = eci_df.dropna(subset=['Release date', 'ECI Score'])
eci_df = eci_df[eci_df['Release date'] >= '2022-01-01'].sort_values('Release date')

# Calculate days since start, then strictly CENTER it
eci_df['days'] = (eci_df['Release date'] - eci_df['Release date'].min()).dt.days
X_raw = eci_df['days'].values
X = X_raw - np.mean(X_raw)  # Centering X as recommended
Y = eci_df['ECI Score'].values

# ==========================================
# 2. Define Log-Posterior Functions
# ==========================================
def log_prior(theta):
    beta0, beta1, rho, sigma = theta
    
    if not (-1 < rho < 1): return -np.inf
    if sigma <= 0: return -np.inf
        
    log_p_beta0 = stats.norm.logpdf(beta0, loc=0, scale=100)
    log_p_beta1 = stats.norm.logpdf(beta1, loc=0, scale=100)
    
    sigma_sq = sigma**2
    log_p_sigma = stats.invgamma.logpdf(sigma_sq, a=2, scale=2)
    
    return log_p_beta0 + log_p_beta1 + log_p_sigma

def log_likelihood(theta, X, Y):
    beta0, beta1, rho, sigma = theta
    eps = Y - (beta0 + beta1 * X)

    eps_t = eps[1:]
    eps_t_minus_1 = eps[:-1]
    ll_transitions = np.sum(stats.norm.logpdf(eps_t, loc=rho * eps_t_minus_1, scale=sigma))
    ll_first = np.sum(stats.norm.logpdf(eps, loc=0, scale=sigma / np.sqrt(1 - rho**2)))

    return ll_transitions + ll_first

def log_posterior(theta, X, Y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, X, Y)

# ==========================================
# 3. Random Walk Metropolis Sampler (Modified for Multi-Chain)
# ==========================================
# Note: Added start_theta parameter so we can jitter initial positions
def metropolis_sampler(X, Y, start_theta, iters=10000, burn_in=2000, chain_id=1):
    n_params = 4
    chain = np.zeros((iters, n_params))
    
    current_theta = start_theta.copy()
    step_sizes = np.array([0.25, 0.002, 0.025, 0.1])
    
    current_lp = log_posterior(current_theta, X, Y)
    accepted = 0
    
    for i in range(iters):
        proposal_theta = current_theta + np.random.normal(0, step_sizes, n_params)
        proposal_lp = log_posterior(proposal_theta, X, Y)
        log_alpha = proposal_lp - current_lp
        
        if np.log(np.random.rand()) < log_alpha:
            current_theta = proposal_theta
            current_lp = proposal_lp
            if i >= burn_in:
                accepted += 1
                
        chain[i, :] = current_theta
        
        if i > 0 and i % 5000 == 0:
            print(f"  Chain {chain_id}: Iteration {i} complete...")
            
    acc_rate = accepted / (iters - burn_in)
    print(f"  Chain {chain_id} complete. Post-burn-in acceptance rate: {acc_rate:.2%}")
    
    return chain[burn_in:, :]

def run_multiple_chains(X, Y, n_chains=4, iters=15000, burn_in=3000):
    print(f"\nStarting {n_chains} chains ({iters} iterations each)...")
    chains = []
    
    # Base starting point off OLS
    slope_init, intercept_init, _, _, _ = stats.linregress(X, Y)
    
    for c in range(n_chains):
        # Create "overdispersed" starting values for each chain to strictly test convergence
        start_theta = np.array([
            intercept_init + np.random.normal(0, 10),      # beta0
            slope_init + np.random.normal(0, 0.02),        # beta1
            np.random.uniform(-0.3, 0.3),                  # rho
            np.random.uniform(2.0, 8.0)                    # sigma
        ])
        
        chain = metropolis_sampler(X, Y, start_theta, iters=iters, burn_in=burn_in, chain_id=c+1)
        chains.append(chain)
        
    return np.array(chains) # Shape: (M_chains, N_samples, Num_Params)

# ==========================================
# 4. Gelman-Rubin Diagnostic
# ==========================================
def calculate_gelman_rubin(chains):
    """
    Calculates the R-hat statistic for each parameter.
    chains shape must be: (M_chains, N_samples, num_parameters)
    """
    M, N, num_params = chains.shape
    r_hats = []
    
    for p in range(num_params):
        chain_samples = chains[:, :, p]
        
        # 1. Within-chain variance (W)
        chain_vars = np.var(chain_samples, axis=1, ddof=1)
        W = np.mean(chain_vars)
        
        # 2. Between-chain variance (B)
        chain_means = np.mean(chain_samples, axis=1)
        overall_mean = np.mean(chain_means)
        B = N / (M - 1) * np.sum((chain_means - overall_mean)**2)
        
        # 3. Estimated marginal variance (V)
        V = (N - 1) / N * W + B / N
        
        # 4. R-hat
        R_hat = np.sqrt(V / W)
        r_hats.append(R_hat)
        
    return np.array(r_hats)

# Run the multi-chain sampler
multi_chain_trace = run_multiple_chains(X, Y, n_chains=4, iters=15000, burn_in=3000)

# Calculate Gelman-Rubin stats
r_hat_stats = calculate_gelman_rubin(multi_chain_trace)

# Combine chains for final posterior estimation and plotting
# Shape goes from (4, 12000, 4) -> (48000, 4)
combined_trace = multi_chain_trace.reshape(-1, multi_chain_trace.shape[2])

# ==========================================
# 5. Output Summary & Plotting
# ==========================================
param_names = ['beta0', 'beta1', 'rho', 'sigma']
param_labels = [r'Intercept ($\beta_0$)', r'Slope ($\beta_1$)', r'AR(1) Coef ($\rho$)', r'Residual Std ($\sigma$)']

print("\n=== Phase 1 Convergence Diagnostics ===")
for i, name in enumerate(param_names):
    status = "✅ Converged" if r_hat_stats[i] < 1.05 else "❌ Needs more iterations/tuning"
    print(f"{name:5} R-hat: {r_hat_stats[i]:.4f}  {status}")

print("\n=== Phase 1 Posterior Summary (Combined Chains) ===")
for i, name in enumerate(param_names):
    mean_val = np.mean(combined_trace[:, i])
    ci_lower, ci_upper = np.percentile(combined_trace[:, i], [2.5, 97.5])
    print(f"{name:5}: Mean = {mean_val:8.4f} | 95% CI: [{ci_lower:8.4f}, {ci_upper:8.4f}]")

annual_slope = np.mean(combined_trace[:, 1]) * 365
print(f"\nAnnualized capability progress (beta1 * 365): {annual_slope:.2f} ECI points/year")


# Plotting Trace and Posteriors
colors = ['#60A5FA', '#34D399', '#F472B6', '#FBBF24']

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
fig.patch.set_facecolor('white')

for i in range(4):
    # Plot trace for EACH chain independently to visualize mixing
    for c in range(multi_chain_trace.shape[0]):
        axes[i, 0].plot(multi_chain_trace[c, :, i], alpha=0.6, color=colors[c])
        
    axes[i, 0].set_title(f'Trace (4 Chains): {param_labels[i]}')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Histogram / Posterior (combined chains)
    axes[i, 1].hist(combined_trace[:, i], bins=50, color='#94A3B8', density=True, alpha=0.8, edgecolor='white')
    axes[i, 1].set_title(f'Posterior: {param_labels[i]}')
    
    median_val = np.median(combined_trace[:, i])
    axes[i, 1].axvline(median_val, color='red', linestyle='dashed', linewidth=1.5, 
                       label=f'Median: {median_val:.4f}')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase1_bayesian_pooled_ar1_multichain.png", dpi=150)
plt.show()

# ==========================================
# 6. Residual Diagnostics (Using combined trace)
# ==========================================
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

beta0_hat = np.mean(combined_trace[:, 0])
beta1_hat = np.mean(combined_trace[:, 1])
rho_hat   = np.mean(combined_trace[:, 2])

raw_residuals = Y - (beta0_hat + beta1_hat * X)
ar1_residuals = raw_residuals[1:] - rho_hat * raw_residuals[:-1]

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("Residual Diagnostics: OLS vs Bayesian AR(1)", fontsize=14)

plot_acf(raw_residuals, lags=20, ax=axes2[0, 0], color='#60A5FA')
axes2[0, 0].set_title("ACF: Raw Residuals (= OLS residuals)")
axes2[0, 0].set_xlabel("Lag")

plot_acf(ar1_residuals, lags=20, ax=axes2[0, 1], color='#34D399')
axes2[0, 1].set_title("ACF: AR(1) Corrected Residuals (η_t)")
axes2[0, 1].set_xlabel("Lag")

axes2[1, 0].scatter(eci_df['Release date'].values, raw_residuals, alpha=0.6, color='#60A5FA', s=30)
axes2[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes2[1, 0].set_title("Raw Residuals over Time")

axes2[1, 1].scatter(eci_df['Release date'].values[1:], ar1_residuals, alpha=0.6, color='#34D399', s=30)
axes2[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes2[1, 1].set_title("AR(1) Corrected Residuals over Time")

plt.tight_layout()
plt.savefig("phase1_residual_diagnostics_multichain.png", dpi=150)
plt.show()


lb_raw = acorr_ljungbox(raw_residuals, lags=10, return_df=True)
lb_ar1 = acorr_ljungbox(ar1_residuals, lags=10, return_df=True)
print(f"Ljung-Box p-value — Raw residuals:         {lb_raw['lb_pvalue'].values[-1]:.4f}")
print(f"Ljung-Box p-value — AR(1) corrected:       {lb_ar1['lb_pvalue'].values[-1]:.4f}")