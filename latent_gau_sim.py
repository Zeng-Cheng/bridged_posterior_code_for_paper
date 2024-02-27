from tqdm import trange
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np
import jax.numpy as jnp
from jax import jit, device_put
import scipy
from src.latent_gau_model import latent_gau_log_prob, latent_gau_optim_step
from src.latent_gau_model import latent_gau_log_prob_profile
from polyagamma import random_polyagamma
from src.utils import acc_rate
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('curve', type=int)

def get_w_true(curve_num, X):
    if curve_num == 1:
        # curve 1
        w_true = -np.exp(-(X - 5) ** 2 / 1) + np.exp(-(X - 1) ** 2 / 2) / 5 + np.exp(-(X + 2) ** 2 / 2)
    elif curve_num == 2:
        # curve 2
        w_true = np.exp(-(X - 4) ** 2 / 1.2) + np.exp(-(X - 0.5) ** 2 / 1) / 3 - np.exp(-(X + 4) ** 2 / 0.9)
    elif curve_num == 3:
        # curve 3
        w_true = -np.exp(-(X - 3) ** 2 / 1.4) + np.exp(-(X - 0) ** 2 / 1) / 5 + np.exp(-(X + 1) ** 2 / 2)
    elif curve_num == 4:
        # curve 4
        w_true = -np.exp(-(X - 2) ** 2 / 2.5) + np.exp(-(X + 0.5) ** 2 / 0.5) / 2 + np.exp(-(X + 5) ** 2 / 0.4)
    elif curve_num == 5:
        # curve 5
        w_true = np.exp(-(X - 1) ** 2 / 3.8) + np.exp(-(X + 1) ** 2 / 1) / 6 - np.exp(-(X + 3) ** 2 / 1)
    elif curve_num == 6:
        w_true = np.cos(X)
    elif curve_num == 7:
        w_true = (np.cos(X / 2) + np.sin(X / 1.3)) / 2
    elif curve_num == 8:
        w_true = (np.cos(X / 1.2) - np.sin(X * 1.1)) / 2
    elif curve_num == 9:
        w_true = (np.cos((X - 1) / 1.2) - np.sin(X / 1.6)) / 1.5
    else:
        w_true = None
    return w_true

def get_w_true_jnp(curve_num, X):
    if curve_num == 1:
        # curve 1
        w_true = -jnp.exp(-(X - 5) ** 2 / 1) + jnp.exp(-(X - 1) ** 2 / 2) / 5 + jnp.exp(-(X + 2) ** 2 / 2)
    elif curve_num == 2:
        # curve 2
        w_true = jnp.exp(-(X - 4) ** 2 / 1.2) + jnp.exp(-(X - 0.5) ** 2 / 1) / 3 - jnp.exp(-(X + 4) ** 2 / 0.9)
    elif curve_num == 3:
        # curve 3
        w_true = -jnp.exp(-(X - 3) ** 2 / 1.4) + jnp.exp(-(X - 0) ** 2 / 1) / 5 + jnp.exp(-(X + 1) ** 2 / 2)
    elif curve_num == 4:
        # curve 4
        w_true = -jnp.exp(-(X - 2) ** 2 / 2.5) + jnp.exp(-(X + 0.5) ** 2 / 0.5) / 2 + jnp.exp(-(X + 5) ** 2 / 0.4)
    elif curve_num == 5:
        # curve 5
        w_true = jnp.exp(-(X - 1) ** 2 / 3.8) + jnp.exp(-(X + 1) ** 2 / 1) / 6 - jnp.exp(-(X + 3) ** 2 / 1)
    elif curve_num == 6:
        w_true = jnp.cos(X)
    elif curve_num == 7:
        w_true = (jnp.cos(X / 2) + jnp.sin(X / 1.3)) / 2
    elif curve_num == 8:
        w_true = (jnp.cos(X / 1.2) - jnp.sin(X * 1.1)) / 2
    elif curve_num == 9:
        w_true = (jnp.cos((X - 1) / 1.2) - jnp.sin(X / 1.6)) / 1.5
    else:
        w_true = None
    return w_true

if __name__ == '__main__':
    # args = parser.parse_args()
    # curve_num = args.curve
    for curve_num in range(1, 10):
        for ss in [50, 200, 500, 1000]:
            
            # setup seed
            seed = 1509
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # simulating data
            n = ss
            X = (np.random.rand(n, 1) - 0.5) * 12
            dist_x = scipy.spatial.distance.cdist(X, X) ** 2

            w_true = get_w_true(curve_num, X)
            w_true = w_true.T[0]
            prob_sigmoid = 1 - 1 / (1 + np.exp(w_true))
            y = np.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0
            y = torch.tensor(y).float()
            X = torch.tensor(X).float()
        
            alpha_prior = 2
            beta = 5 - (1000 - ss) / 500
            
            # simulating parameters
            num_samples = 13000
            burn_in = 4000
            lam_tau = 0.5
            lam_b = 0.5
            
            tau_trace, b_trace = [], []
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
                logprob = latent_gau_log_prob(
                    'canonical', w, 0, dist_x, y, tau, b, alpha_prior, beta)
                
                # Metroplis--Hastings

                tau_prop = tau + torch.randn(1) * lam_tau
                if (tau_prop > 0): 
                    logprob_prop = latent_gau_log_prob(
                        'canonical', w, 0, dist_x, y, tau_prop, b, alpha_prior, beta)
                    if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
                        tau = tau_prop
                        logprob = logprob_prop
                        accept_tau[ns] = 1
                    
                b_prop = b + torch.randn(1) * lam_b
                if b_prop > 0:
                    logprob_prop = latent_gau_log_prob(
                        'canonical', w, 0, dist_x, y, tau, b_prop, alpha_prior, beta)
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
                tau_trace.append(tau)
                b_trace.append(b)
                if ns % 2000 == 0:
                    print('step:, {:d}, accept_rate of tau: {:.3f}, of b: {:.3f}'.format(
                        ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))
        
            tau_trace, b_trace = np.stack(tau_trace), np.stack(b_trace)
            np.savetxt("res_latent_gau/tau_aug_trace_curve" +
                       str(curve_num) + "ss" + str(ss) + ".txt", tau_trace[burn_in:])
            np.savetxt("res_latent_gau/b_aug_trace_curve" +
                       str(curve_num) + "ss" + str(ss) + ".txt", b_trace[burn_in:])

    optim_jit = jit(latent_gau_optim_step)

    for curve_num in range(1, 10):
        for ss in [50, 200, 500, 1000]:
            
            # setup seed
            seed = 1509
            np.random.seed(seed)
            
            # simulating data
            n = ss # sample size
            X = (np.random.rand(n, 1) - 0.5) * 12
            dist_x = scipy.spatial.distance.cdist(X, X) ** 2
            w_true = get_w_true_jnp(curve_num, X)
            
            w_true = w_true.T[0]
            prob_sigmoid = 1 - 1 / (1 + jnp.exp(w_true))
            y = jnp.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0
        
            alpha_prior = 2
            beta = 5 - (1000 - ss) / 500

            # sampling parameters
            num_samples = 5000
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
                
                # Metroplis--Hastings

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
                        
                # Metroplis--Hastings
                        
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
            
            tau_trace, b_trace = np.stack(tau_trace), np.stack(b_trace)
            
            np.savetxt("res_latent_gau/tau_trace_curve" +
                       str(curve_num) + "ss" + str(ss) + ".txt", tau_trace[burn_in:])
            np.savetxt("res_latent_gau/b_trace_curve" +
                       str(curve_num) + "ss" + str(ss) + ".txt", b_trace[burn_in:])