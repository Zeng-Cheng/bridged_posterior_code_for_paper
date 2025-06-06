import jax.numpy as jnp
from jax import grad, lax

def latent_gau_optim_step(dist_x, y, tau, b):
    n = y.size
    q_matrix = tau * (jnp.exp(-dist_x / 2 / b) + 0.01 * jnp.eye(n))
    # z is alpha + y \in (0, 1)
    logit_z = jnp.zeros(n)

    def cal_loss(logit_z):
        z = 1 - 1 / (1 + jnp.exp(logit_z))
        res = z * jnp.log(z / (1 - z)) + jnp.log(1 - z)
        loss = (z - y) @ q_matrix @ (z - y) / 2 + res.sum()
        return loss

    for _ in range(20):
        loss = cal_loss(logit_z)
        
        logit_z_grad = grad(cal_loss)(logit_z)
        step = jnp.array([0.5])
        def cond_fun(step):
            return (cal_loss(logit_z - step * logit_z_grad) > loss - 
                step * jnp.linalg.norm(logit_z_grad) ** 2 / 2)[0]
        def body_fun(step):
            return step * 0.6
        step = lax.while_loop(cond_fun, body_fun, step)
        logit_z -= step * logit_z_grad
    
    z = 1 - 1 / (1 + jnp.exp(logit_z))
    return -q_matrix @ (z - y), z - y


def latent_gau_log_prob_profile(zeta, alpha, dist_x, y, tau, b, alpha_prior = 2, beta = 5):
    """
    Args:
        zeta: the latent Gaussian variables; vector of 1*n
        dist_x: matrix of |xi-xj|^2
    """
    if tau <= 0:
        raise ValueError("tau must be greater than 0")
    
    if b <= 0:
        raise ValueError("b must be greater than 0")
    
    n = y.size # sample size
    q_matrix = tau * (jnp.exp(-dist_x / 2 / b) + 0.01 * jnp.eye(n))
    log_tau_prior = -tau ** 2 / 2
    log_b_prior = (-alpha_prior - 1) * jnp.log(b) - (beta / b)
    res_latent_gau = -alpha @ q_matrix @ alpha / 2

    res_logistic_p = y * zeta - jnp.logaddexp(jnp.zeros(1), zeta)
    loglik = res_latent_gau + res_logistic_p.sum()

    if jnp.isnan(loglik):
        loglik = float('-inf')
    return loglik + log_tau_prior + log_b_prior