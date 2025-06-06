# %%
from tqdm.notebook import trange
import numpy as np
import scipy
from polyagamma import random_polyagamma
from src.utils import acc_rate

# load jax related only for bridged model
import jax.numpy as jnp
from jax import jit, device_put
from src.latent_gau_model import latent_gau_optim_step, latent_gau_log_prob_profile

# load torch related only for canonical model
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from src.latent_gau_canonical import latent_gau_log_prob

# %%
##########################################
### simulation for bridged posterior ###
##########################################

np.random.seed(42)  # Sets global seed
# simulating data
n = 1000 # sample size
X = (np.random.rand(n, 1) - 0.5) * 12
dist_x = scipy.spatial.distance.cdist(X, X) ** 2
w_true = jnp.cos(X)
w_true = w_true.T[0]
prob_sigmoid = 1 - 1 / (1 + jnp.exp(w_true))
y = jnp.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0

optim_jit = jit(latent_gau_optim_step)

# %%
alpha_prior = 2
beta = 5

# sampling parameters
num_samples = 10000
burn_in = 2000

lam_tau = 1
lam_b = 0.5

tau_trace, b_trace = [], []
accept_tau, accept_b = np.zeros(num_samples), np.zeros(num_samples)

# initial values
tau = jnp.array([2])
b = jnp.array([1.5])

zeta, alpha = optim_jit(dist_x, y, tau, b)
logprob = latent_gau_log_prob_profile(
    zeta, alpha, dist_x, y, tau, b, alpha_prior, beta)

for ns in trange(num_samples):
    
    # Metroplis--Hastings for tau

    tau_prop = device_put(tau + np.random.randn(1) * lam_tau)
    if (tau_prop > 0):
        zeta_prop, alpha_prop = optim_jit(dist_x, y, tau_prop, b)
        logprob_prop = latent_gau_log_prob_profile(
            zeta_prop, alpha_prop, dist_x, y, tau_prop, b, alpha_prior, beta)
        

        if (jnp.log(np.random.rand(1)) < (logprob_prop - logprob)):
            tau = tau_prop
            zeta = zeta_prop
            alpha = alpha_prop
            logprob = logprob_prop
            accept_tau[ns] = 1
            
    # Metroplis--Hastings for b
            
    b_prop = device_put(b + np.random.randn(1) * lam_b)
    if b_prop > 0:
        zeta_prop, alpha_prop = optim_jit(dist_x, y, tau, b_prop)
        logprob_prop = latent_gau_log_prob_profile(
            zeta_prop, alpha_prop, dist_x, y, tau, b_prop, alpha_prior, beta)
        
        if (jnp.log(np.random.rand(1)) < (logprob_prop - logprob)):
            b = b_prop
            zeta = zeta_prop
            alpha = alpha_prop
            logprob = logprob_prop
            accept_b[ns] = 1

    # adjust the step size of random walk
            
    if (ns % 200 == 0) and (ns < burn_in):
        if acc_rate(accept_tau, ns + 1) > 0.4:
            lam_tau *= 1.8
        if acc_rate(accept_tau, ns + 1) < 0.25:
            lam_tau /= 3
        if acc_rate(accept_b, ns + 1) > 0.4:
            lam_b *= 1.8
        if acc_rate(accept_b, ns + 1) < 0.25:
            lam_b /= 3
        
    tau_trace.append(tau)
    b_trace.append(b)
    if ns % 400 == 0:
        print('step: {:d}, accept_rate of tau: {:.3f}, of b: {:.3f}'.format(
            ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))

# %%
tau_trace, b_trace = np.stack(tau_trace), np.stack(b_trace)
np.savetxt("output/res_latent_gau/tau_trace.txt", tau_trace)
np.savetxt("output/res_latent_gau/b_trace.txt", b_trace)

# %%


#########################################
### simulation for canonical model ###
#########################################

np.random.seed(42)  # Sets global seed
torch.manual_seed(42)
# simulating data
n = 1000
X = (np.random.rand(n, 1) - 0.5) * 12
dist_x = scipy.spatial.distance.cdist(X, X) ** 2
w_true = np.cos(X)
w_true = w_true.T[0]
prob_sigmoid = 1 - 1 / (1 + np.exp(w_true))
y = np.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0
y = torch.tensor(y).float()
X = torch.tensor(X).float()

# %%
alpha_prior = 2
beta = 5

# simulating parameters
num_samples = 10000
burn_in = 2000
lam_tau = 0.5
lam_b = 0.5

tau_aug_trace, b_aug_trace = [], []
accept_tau, accept_b = np.zeros(num_samples), np.zeros(num_samples)

# initial values
tau = torch.tensor([1.5])
b = torch.tensor([1.5])
eta = torch.tensor(random_polyagamma(1, size=n))

for ns in trange(num_samples):
    
    # Gibbs step via data augmentation

    q_matrix = (torch.exp(-dist_x / b / 2)).float() + 0.01 * torch.eye(n)
    Omega = torch.diag(eta)
    w_var = torch.inverse(torch.inverse(q_matrix) / tau + Omega).float()
    w_var = (w_var + w_var.T) / 2
    w_mean = w_var @ (y - 0.5)
    w = MVN(w_mean, w_var).sample()
    eta = torch.tensor(random_polyagamma(1, w))
    logprob = latent_gau_log_prob(w, dist_x, y, tau, b, alpha_prior, beta)
    
    # Metroplis--Hastings

    tau_prop = tau + torch.randn(1) * lam_tau
    if (tau_prop > 0): 
        logprob_prop = latent_gau_log_prob(w, dist_x, y, tau_prop, b, alpha_prior, beta)
        if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
            tau = tau_prop
            logprob = logprob_prop
            accept_tau[ns] = 1
        
    b_prop = b + torch.randn(1) * lam_b
    if b_prop > 0:
        logprob_prop = latent_gau_log_prob(w, dist_x, y, tau, b_prop, alpha_prior, beta)
        if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
            b = b_prop
            logprob = logprob_prop
            accept_b[ns] = 1
            
    # adjust the step size of random walk
            
    if (ns % 200 == 0) and (ns < burn_in):
        if acc_rate(accept_tau, ns + 1) > 0.4:
            lam_tau *= 2
        if acc_rate(accept_tau, ns + 1) < 0.3:
            lam_tau /= 3
        if acc_rate(accept_b, ns + 1) > 0.4:
            lam_b *= 2
        if acc_rate(accept_b, ns + 1) < 0.3:
            lam_b /= 3
    tau_aug_trace.append(tau)
    b_aug_trace.append(b)
    if ns % 1000 == 0:
        print('step:, {:d}, accept_rate of tau: {:.3f}, of b: {:.3f}'.format(
            ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))


# %%
tau_aug_trace, b_aug_trace = np.stack(tau_aug_trace), np.stack(b_aug_trace)
np.savetxt("output/res_latent_gau/tau_aug_trace.txt", tau_aug_trace)
np.savetxt("output/res_latent_gau/b_aug_trace.txt", b_aug_trace)


# %%

####################################################
### following is the testing codes for using jax 
####################################################

# # simulation for canonical model

# # simulating data
# n = 1000
# X = (np.random.rand(n, 1) - 0.5) * 12
# dist_x = scipy.spatial.distance.cdist(X, X) ** 2
# w_true = jnp.cos(X)
# w_true = w_true.T[0]
# prob_sigmoid = 1 - 1 / (1 + jnp.exp(w_true))
# y = jnp.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0

# # %%
# alpha_prior = 2
# beta = 5

# # simulating parameters
# num_samples = 10000
# burn_in = 2000
# lam_tau = 0.5
# lam_b = 0.5

# tau_aug_trace, b_aug_trace = [], []
# accept_tau, accept_b = np.zeros(num_samples), np.zeros(num_samples)

# # initial values
# tau = jnp.array([1.5])
# b = jnp.array([1.5])
# eta = jnp.array(random_polyagamma(1, size=n))

# for ns in trange(num_samples):
    
#     # Gibbs step via data augmentation

#     q_matrix = (jnp.exp(-dist_x / b / 2)) + 0.01 * jnp.eye(n)
#     Omega = jnp.diag(eta)
#     w_var = jnp.linalg.inv(jnp.linalg.inv(q_matrix) / tau + Omega)
#     w_var = (w_var + w_var.T) / 2
#     w_mean = w_var @ (y - 0.5)
#     key = jax.random.PRNGKey(0)
#     w = jax.random.multivariate_normal(key, w_mean, w_var)
#     eta = jnp.array(random_polyagamma(1, w))
#     logprob = latent_gau_log_prob(w, dist_x, y, tau, b, alpha_prior, beta)
    
#     # Metroplis--Hastings

#     tau_prop = tau + jax.random.normal(key, 1) * lam_tau
#     if (tau_prop > 0): 
#         logprob_prop = latent_gau_log_prob(w, dist_x, y, tau_prop, b, alpha_prior, beta)
#         if (jnp.log(jax.random.uniform(key, 1)) < (logprob_prop - logprob)):
#             tau = tau_prop
#             logprob = logprob_prop
#             accept_tau[ns] = 1
        
#     b_prop = b + jax.random.normal(key, 1) * lam_b
#     if b_prop > 0:
#         logprob_prop = latent_gau_log_prob(w, dist_x, y, tau, b_prop, alpha_prior, beta)
#         if (jnp.log(jax.random.uniform(key, 1)) < (logprob_prop - logprob)):
#             b = b_prop
#             logprob = logprob_prop
#             accept_b[ns] = 1
            
#     # adjust the step size of random walk
            
#     if (ns % 200 == 0) and (ns < burn_in):
#         if acc_rate(accept_tau, ns + 1) > 0.4:
#             lam_tau *= 2
#         if acc_rate(accept_tau, ns + 1) < 0.3:
#             lam_tau /= 3
#         if acc_rate(accept_b, ns + 1) > 0.4:
#             lam_b *= 2
#         if acc_rate(accept_b, ns + 1) < 0.3:
#             lam_b /= 3
#     tau_aug_trace.append(tau)
#     b_aug_trace.append(b)
#     if ns % 1000 == 0:
#         print('step:, {:d}, accept_rate of tau: {:.3f}, of b: {:.3f}'.format(
#             ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))