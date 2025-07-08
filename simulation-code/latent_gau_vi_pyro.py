# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoIAFNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

import scipy
from tqdm.notebook import trange

# %%

np.random.seed(42)  # Sets global seed
torch.manual_seed(42)
# simulating data
n = 1000
X = (np.random.rand(n, 1) - 0.5) * 12
w_true = np.cos(X)
w_true = w_true.T[0]
prob_sigmoid = 1 - 1 / (1 + np.exp(w_true))
y = np.where((np.random.rand(n) - prob_sigmoid) < 0, 1, 0) * 1.0
y = torch.tensor(y).float()
X = torch.tensor(X).float()

# %%

dist_x = torch.tensor(scipy.spatial.distance.cdist(X, X) ** 2).float()

def log_posterior(z, tau, b, y):
    Q = ((torch.exp(-dist_x / b / 2)) + 0.01 * torch.eye(n)) * tau
    term1 = torch.sum(y * z)
    term2 = -torch.sum(torch.logaddexp(torch.tensor(0.0), z))
    logL = -0.5 * z @ torch.linalg.inv(Q) @ z + term1 + term2
    return logL

def model(y):
    z = pyro.sample("z", dist.Normal(0.0, 10.0).expand([n]).to_event(1)) # type: ignore
    tau = pyro.sample("tau", dist.HalfNormal(1.0))
    b = pyro.sample("b", dist.InverseGamma(2.0, 5.0))
    logL = log_posterior(z, tau, b, y)
    pyro.factor("log_prob", logL)


# %%
#==================================
# Variational inference
#==================================

guide = AutoDiagonalNormal(model, init_scale=1.0)
# guide = AutoIAFNormal(model, hidden_dim=[32, 32], num_transforms=4)
optimizer = pyro.optim.Adam({"lr": 1e-4})
# scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 1e-3}, 'gamma': 0.6})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=5))

loss_history = []
for epoch in trange(50000):
    loss = svi.step(y)
    loss_history.append(loss)
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

# %%
plt.figure(figsize=(5, 3))
plt.plot(loss_history, alpha=0.3)
window_size = 100
smoothed = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(loss_history)), smoothed, 'r-', linewidth=2)

best_epoch = np.argmin(smoothed)
best_value = smoothed[best_epoch]
plt.scatter(best_epoch + window_size -1, best_value, c='g', s=100)

plt.xlabel("Training Step", fontsize=12)
plt.ylabel("ELBO", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
predictive = Predictive(model=model, guide=guide, num_samples=4000,
    return_sites=["tau", "b", "z"])

samples = predictive(y)

# %%
# ======================================================
# Convert samples to a format suitable for visualization
# ======================================================
tau_samples = samples['tau'].detach().cpu().numpy()
b_samples = samples['b'].detach().cpu().numpy()
z_samples = samples['z'].detach().cpu().numpy()

np.savetxt('output/res_latent_gau_vi/latent_gau_tau_samples.txt', tau_samples, delimiter=',')
np.savetxt('output/res_latent_gau_vi/latent_gau_b_samples.txt', b_samples, delimiter=',')
np.savetxt('output/res_latent_gau_vi/latent_gau_z_samples.txt', z_samples, delimiter=',')

# %%

plt.figure(figsize=(15, 6))
plt.scatter(b_samples, tau_samples, alpha=0.1, s=1)
plt.tight_layout(pad=3)
plt.show()


# %%

# plot curve of z_samples
plt.figure(figsize=(15, 6))
plt.plot(np.mean(z_samples, axis=0), label='Mean of z samples', color='blue')
plt.tight_layout(pad=3)
plt.show()

# %%
