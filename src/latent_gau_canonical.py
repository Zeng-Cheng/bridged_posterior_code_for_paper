import torch

# We have tried to use jax, and found torch is faster than jax

def latent_gau_log_prob(zeta, dist_x, y, tau, b, alpha_prior = 2, beta = 5):
    """
    Args:
        zeta: the latent Gaussian variables; vector of 1*n
        dist_x: matrix of |xi-xj|^2
    """
    if tau <= 0:
        raise ValueError("tau must be greater than 0")
    
    if b <= 0:
        raise ValueError("b must be greater than 0")
    
    n = y.size(0) # sample size
    q_matrix = (torch.exp(-dist_x / b / 2)).float() + 0.01 * torch.eye(n)
    # q_matrix is without tau, and we add eps*I to solve singular problem

    log_tau_prior = -tau ** 2 / 2
    log_b_prior = (-alpha_prior - 1) * torch.log(b) - beta / b
    loglik = (-zeta @ torch.linalg.solve(q_matrix, zeta) / 2 / tau
                - torch.logdet(q_matrix) / 2 - n * torch.log(tau) / 2)

    if torch.isnan(loglik):
        return float('-inf')
    return loglik + log_tau_prior + log_b_prior


############################################
### below is the jax version codes ###
############################################

# def latent_gau_log_prob(zeta, dist_x, y, tau, b, alpha_prior = 2, beta = 5):
#     """
#     Args:
#         zeta: the latent Gaussian variables; vector of 1*n
#         dist_x: matrix of |xi-xj|^2
#     """
#     if tau <= 0:
#         raise ValueError("tau must be greater than 0")
    
#     if b <= 0:
#         raise ValueError("b must be greater than 0")
    
#     n = y.size # sample size
#     q_matrix = (jnp.exp(-dist_x / b / 2)) + 0.01 * jnp.eye(n)
#     # q_matrix is without tau, and we add eps*I to solve singular problem

#     log_tau_prior = -tau ** 2 / 2
#     log_b_prior = (-alpha_prior - 1) * jnp.log(b) - beta / b
#     loglik = (-zeta @ jnp.linalg.solve(q_matrix, zeta) / 2 / tau
#                 - jnp.linalg.slogdet(q_matrix)[1] / 2 - n * jnp.log(tau) / 2)

#     if jnp.isnan(loglik):
#         return float('-inf')
#     return loglik + log_tau_prior + log_b_prior


