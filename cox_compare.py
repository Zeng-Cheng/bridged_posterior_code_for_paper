# %%
import numpy as np
import matplotlib.pyplot as plt
from src.utils import acc_rate
from tqdm.notebook import trange
from scipy.special import logsumexp

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# %%
np.random.seed(42)

n = 500
x = np.random.randn(n)
# For simulation, create a baseline hazard: piecewise constant on 5 intervals
event_times = np.linspace(0, 10, n)
true_hazards = np.random.gamma(shape=2, scale=1, size=n)
lambda_true = 0.8

# Simulate survival times (using inverse hazard for piecewise constant hazard)
def simulate_time(xi, basehaz, event_times):
    # For simplicity, use a large enough time
    t = 0
    for k in range(len(basehaz)):
        rate = np.exp(lambda_true * xi) * basehaz[k]
        delta = event_times[k+1] - t if k < len(basehaz)-1 else 10 - t
        u = np.random.uniform()
        t_would_end = -np.log(u) / rate
        if t_would_end <= delta:
            t += t_would_end
            return t
        t += delta
    return t

y = np.array([simulate_time(x[i], true_hazards, event_times) for i in range(n)])

# %%

prior_mu = 0.0
prior_sigma = 5.0

def log_posterior(lam, x, y):
    log_prior = -0.5 * np.log(2 * np.pi * prior_sigma**2)
    log_prior = log_prior - 0.5 * ((lam - prior_mu) / prior_sigma)**2
    loglik = 0.
    for i in range(len(y)):
        riskset = x[y >= y[i]]
        numer = lam * x[i]
        denom = logsumexp(lam * riskset)
        loglik += numer - denom
    return loglik + log_prior

# RWMH sampler for lambda
n_iters = 10000
burn_in = 2000
lam_current = 1.0
logp_current = log_posterior(lam_current, x, y)
step_size = 0.15
samples = []
accepts = np.zeros(n_iters)

for i in trange(n_iters):
    lam_prop = lam_current + np.random.normal(0, step_size)
    logp_prop = log_posterior(lam_prop, x, y)
    if np.log(np.random.uniform()) < logp_prop - logp_current:
        logp_current = logp_prop
        lam_current = lam_prop
        accepts[i] = 1
    if (i % 200 == 0) and (i < burn_in):
        # adjust the step size
        if acc_rate(accepts, i + 1) > 0.4:
            step_size *= 1.5
        elif acc_rate(accepts, i + 1) < 0.3:
            step_size *= 0.3
    
    samples.append(lam_current)
    if i % 1000 == 0:
        print(f"Iteration {i}, lambda: {lam_current:.4f}, acceptance rate: {acc_rate(accepts, i + 1):.3f}")

# %%

samples = np.array(samples)
plt.hist(samples, bins=40, density=True, alpha=0.7)
plt.xlabel("$\lambda$")
plt.ylabel("Posterior density")
plt.title("Random Walk MH Posterior of $\lambda$\nusing Cox partial likelihood")
plt.show()
print("Posterior mean:", samples.mean())
print("Posterior SD:", samples.std())

np.savetxt("output/res_cox/cox_samples_bridged.txt", samples)


# %%

x = jnp.array(x)
y = jnp.array(y)

def model(x, y):
    lambda_prior = dist.Normal(0.0, 10.0)
    hazard_prior = dist.Gamma(2.0, 1.0).expand([n])
    
    lmbda = numpyro.sample("lambda", lambda_prior)
    hazards = numpyro.sample("hazards", hazard_prior)
    
    # Order indices by event times
    order = jnp.argsort(y)
    ordered_y = y[order]
    ordered_x = x[order]
    ordered_hazards = hazards[order]
    
    # Calculate cumulative hazard in correct order
    cumulative_hazard = jnp.cumsum(ordered_hazards)
    exp_lp = jnp.exp(lmbda * ordered_x)
    
    log_likelihood_sum = 0.0
    for i in range(len(ordered_y)):
        numer = lmbda * ordered_x[i] + jnp.log(ordered_hazards[i])
        denom = cumulative_hazard[i] * exp_lp[i]
        log_likelihood_sum += numer - denom

    numpyro.factor("likelihood", log_likelihood_sum)

# Run MCMC with NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=n_iters, num_warmup=burn_in, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y)
mcmc_samples = mcmc.get_samples()

lambda_samples = mcmc_samples["lambda"]
hazards_samples = mcmc_samples["hazards"]

# %%

np.savetxt("output/res_cox/cox_samples_full.txt", lambda_samples)
# Posterior summaries
print(f"Posterior mean for lambda: {lambda_samples.mean()}")
print(f"Posterior SD for lambda: {lambda_samples.std()}")

# %%
