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
    
    # 1. Uniform prior on rho: must be between -1 and 1 for stationarity
    if not (-1 < rho < 1):
        return -np.inf
        
    # 2. Strict positivity for standard deviation
    if sigma <= 0:
        return -np.inf
        
    # Normal priors for betas
    log_p_beta0 = stats.norm.logpdf(beta0, loc=0, scale=100)
    log_p_beta1 = stats.norm.logpdf(beta1, loc=0, scale=100)
    
    # Inverse Gamma prior for sigma^2 (alpha=2, beta=2) translated to sigma
    # Note: If sigma^2 ~ InvGamma(a, b), the log pdf involves terms of sigma
    sigma_sq = sigma**2
    log_p_sigma = stats.invgamma.logpdf(sigma_sq, a=2, scale=2)
    
    # Uniform prior for rho contributes 0 to log-density (constant) within [-1, 1]
    
    return log_p_beta0 + log_p_beta1 + log_p_sigma

def log_likelihood(theta, X, Y):
    beta0, beta1, rho, sigma = theta

    eps = Y - (beta0 + beta1 * X)

    # AR(1) transition likelihoods for t = 2 to N
    eps_t = eps[1:]
    eps_t_minus_1 = eps[:-1]
    ll_transitions = np.sum(stats.norm.logpdf(eps_t, loc=rho * eps_t_minus_1, scale=sigma))

    # Stationary distribution likelihood for the FIRST observation only
    ll_first = stats.norm.logpdf(eps[0], loc=0, scale=sigma / np.sqrt(1 - rho**2))  # <-- eps[0], not eps

    return ll_transitions + ll_first

def log_posterior(theta, X, Y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, X, Y)

# ==========================================
# 3. Random Walk Metropolis Sampler
# ==========================================
def metropolis_sampler(X, Y, iters=10000, burn_in=2000):
    print(f"Starting Metropolis Sampler ({iters} iterations)...")
    
    # Number of parameters: [beta0, beta1, rho, sigma]
    n_params = 4
    chain = np.zeros((iters, n_params))
    
    # Intial guesses (OLS estimates are a good place to start)
    slope_init, intercept_init, _, _, _ = stats.linregress(X, Y)
    current_theta = np.array([intercept_init, slope_init, 0.1, 5.0]) 
    
    # Step sizes (tuning parameters) - adjusted for centered X
    step_sizes = np.array([0.5, 0.005, 0.05, 0.2]) 
    
    current_lp = log_posterior(current_theta, X, Y)
    accepted = 0
    
    for i in range(iters):
        # 1. Propose a new state by taking a random normal step
        proposal_theta = current_theta + np.random.normal(0, step_sizes, n_params)
        
        # 2. Calculate log-posterior of the proposed state
        proposal_lp = log_posterior(proposal_theta, X, Y)
        
        # 3. Acceptance ratio (in log space)
        log_alpha = proposal_lp - current_lp
        
        # 4. Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            current_theta = proposal_theta
            current_lp = proposal_lp
            if i > burn_in:
                accepted += 1
                
        chain[i, :] = current_theta
        
        if i % 2000 == 0:
            print(f"Iteration {i} complete...")
            
    # Calculate acceptance rate (target is usually 20-40% for RWM)
    acc_rate = accepted / (iters - burn_in)
    print(f"Sampling complete. Acceptance rate (post burn-in): {acc_rate:.2%}")
    
    return chain[burn_in:, :]

# Run the sampler
trace = metropolis_sampler(X, Y, iters=15000, burn_in=3000)

# ==========================================
# 4. Plotting Trace and Posteriors
# ==========================================
param_names = [r'Intercept ($\beta_0$)', r'Slope ($\beta_1$)', r'AR(1) Coef ($\rho$)', r'Residual Std ($\sigma$)']

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
fig.patch.set_facecolor('white')

for i in range(4):
    # Trace plot (left column)
    axes[i, 0].plot(trace[:, i], color='#60A5FA', alpha=0.7)
    axes[i, 0].set_title(f'Trace: {param_names[i]}')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Histogram / Posterior (right column)
    axes[i, 1].hist(trace[:, i], bins=40, color='#34D399', density=True, alpha=0.8, edgecolor='white')
    axes[i, 1].set_title(f'Posterior: {param_names[i]}')
    
    # Add median line
    median_val = np.median(trace[:, i])
    axes[i, 1].axvline(median_val, color='red', linestyle='dashed', linewidth=1.5, 
                       label=f'Median: {median_val:.4f}')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase1_bayesian_pooled_ar1.png", dpi=150)
plt.show()

# Print summary statistics
print("\n=== Phase 1 Posterior Summary ===")
for i, name in enumerate(['beta0', 'beta1', 'rho', 'sigma']):
    mean_val = np.mean(trace[:, i])
    ci_lower, ci_upper = np.percentile(trace[:, i], [2.5, 97.5])
    print(f"{name:5}: Mean = {mean_val:8.4f} | 95% CI: [{ci_lower:8.4f}, {ci_upper:8.4f}]")

# Print un-centered annualized slope for interpretation
annual_slope = np.mean(trace[:, 1]) * 365
print(f"\nAnnualized capability progress (beta1 * 365): {annual_slope:.2f} ECI points/year")

# ==========================================
# 5. Residual Diagnostics
# ==========================================
from statsmodels.graphics.tsaplots import plot_acf

# Posterior means for fitted line
beta0_hat = np.mean(trace[:, 0])
beta1_hat = np.mean(trace[:, 1])
rho_hat   = np.mean(trace[:, 2])

# Raw residuals (same as OLS would give you)
raw_residuals = Y - (beta0_hat + beta1_hat * X)

# AR(1) corrected residuals: eta_t = eps_t - rho * eps_{t-1}
ar1_residuals = raw_residuals[1:] - rho_hat * raw_residuals[:-1]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Residual Diagnostics: OLS vs Bayesian AR(1)", fontsize=14)

# --- OLS raw residuals ACF ---
plot_acf(raw_residuals, lags=20, ax=axes[0, 0], color='#60A5FA')
axes[0, 0].set_title("ACF: Raw Residuals (= OLS residuals)")
axes[0, 0].set_xlabel("Lag")

# --- AR(1) corrected residuals ACF ---
plot_acf(ar1_residuals, lags=20, ax=axes[0, 1], color='#34D399')
axes[0, 1].set_title("ACF: AR(1) Corrected Residuals (η_t)")
axes[0, 1].set_xlabel("Lag")

# --- Raw residuals over time ---
axes[1, 0].scatter(eci_df['Release date'].values, raw_residuals, 
                   alpha=0.6, color='#60A5FA', s=30)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_title("Raw Residuals over Time")
axes[1, 0].set_xlabel("Release Date")
axes[1, 0].set_ylabel("Residual")

# --- AR(1) corrected residuals over time ---
axes[1, 1].scatter(eci_df['Release date'].values[1:], ar1_residuals, 
                   alpha=0.6, color='#34D399', s=30)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].set_title("AR(1) Corrected Residuals over Time")
axes[1, 1].set_xlabel("Release Date")
axes[1, 1].set_ylabel("η_t")

plt.tight_layout()
plt.savefig("phase1_residual_diagnostics.png", dpi=150)
plt.show()

# Print Ljung-Box test for remaining autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_raw = acorr_ljungbox(raw_residuals, lags=[10], return_df=True)
lb_ar1 = acorr_ljungbox(ar1_residuals, lags=[10], return_df=True)
print(f"Ljung-Box p-value — Raw residuals:         {lb_raw['lb_pvalue'].values[0]:.4f}")
print(f"Ljung-Box p-value — AR(1) corrected:       {lb_ar1['lb_pvalue'].values[0]:.4f}")
print("(p > 0.05 = no significant remaining autocorrelation = white noise ✅)")