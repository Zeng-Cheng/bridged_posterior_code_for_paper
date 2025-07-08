# %%
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import random
from statsmodels.graphics.tsaplots import plot_acf
import torch
import torch.distributions as dist
from scipy.stats import truncnorm

from tqdm.notebook import trange
from src.utils import acc_rate

# %%
# set seeds for reproducibility
torch.manual_seed(123)
random.seed(87)
np.random.seed(4812)

n_nodes = 40
net_graph = nx.gnp_random_graph(n_nodes, 0.25, directed=True)

for u, v in net_graph.edges():
    net_graph[u][v]['capacity'] = random.uniform(2.0, 10.0)

source = 0
sink = n_nodes - 1

net_graph.remove_edges_from([(u, v) for u, v in net_graph.edges() if v == source])
net_graph.remove_edges_from([(u, v) for u, v in net_graph.edges() if u == sink])

if not nx.has_path(net_graph, source, sink):
    print("Warning: No path from source to sink! You may want to regenerate the graph.")

print(f"Number of edges: {net_graph.number_of_edges()}")
G = net_graph.copy()

# %%
# plot the network graph generated

def left_to_right_layout(G, source, sink, num_middle_layers=5):
    all_nodes = list(G.nodes())
    all_nodes.remove(source)
    all_nodes.remove(sink)
    pos = {source: (0, -5), sink: (num_middle_layers + 1, -5)}

    mid_nodes = sorted(all_nodes)
    layers = np.array_split(mid_nodes, num_middle_layers)

    for layer_idx, layer_nodes in enumerate(layers, start=1):
        for i, node in enumerate(layer_nodes):
            y = -i  # stack downward
            x = layer_idx
            pos[node] = (x, y)
    
    return pos

pos = left_to_right_layout(net_graph, source, sink)
capacities = nx.get_edge_attributes(net_graph, 'capacity')
edge_widths = [capacities[(u, v)] / 20 for u, v in net_graph.edges()]

nx.draw_networkx_nodes(net_graph, pos, node_color='white', edgecolors='black',
                       node_size=200)
nx.draw_networkx_edges(net_graph, pos, width=edge_widths, edge_color='black',
                       arrows=True, node_size=60, arrowsize=20, arrowstyle='->')
nx.draw_networkx_labels(net_graph, pos)

plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Extract edges and capacities
E = np.array(net_graph.edges())
beta_truth = torch.tensor([net_graph[u][v]['capacity'] for u, v in E])

# Compute the maximum flow
flow_value, flow_dict = nx.maximum_flow(net_graph, source, sink)
print(f"capacities for edges: {beta_truth}")
print(f"Maximum flow value: {flow_value}")

# Produce the z_truth vector from flow_dict
z_truth = torch.zeros(E.shape[0])
for i, (u, v) in enumerate(net_graph.edges()):
    z_truth[i] = flow_dict[u][v]
print(f"z vector: {z_truth}, non zero z: {torch.sum(z_truth > 0)}")
torch.where(z_truth > 0)

# %%
# Generate synthetic observations y and c with Gaussian noise
sigma_y = 1.0  # Standard deviation for y
ss_y = 500

data_y = torch.randn(ss_y, E.shape[0]) * sigma_y + z_truth

# torch.where(z_truth > 0)[0][torch.where(beta_truth[torch.where(z_truth > 0)] == z_truth[torch.where(z_truth > 0)])[0]]
beta_ind = [5, 34, 146, 278, 321]
n_params = len(beta_ind)
print(f"beta truth: {beta_truth[beta_ind]}")
print(f"z truth: {z_truth[beta_ind]}")

# %%

def log_posterior(beta, sigma2_y, data_y):

    # use networkx to compute the maximum flow

    for i in beta_ind:
        u, v = E[i]
        net_graph[u][v]['capacity'] = beta[i].item()

    _, flow_dict = nx.maximum_flow(net_graph, source, sink)
    z = torch.zeros(E.shape[0])
    for i, (u, v) in enumerate(E):
        z[i] = flow_dict[u][v]

    loglik = -torch.sum((data_y - z) ** 2) / (2 * sigma2_y)
    loglik = loglik - ss_y * E.shape[0] * torch.log(sigma2_y) / 2
    
    beta_prior = dist.Exponential(0.2).expand([n_params])
    sigma2_y_prior = dist.InverseGamma(2.0, 5.0)

    logprob_beta = beta_prior.log_prob(beta[beta_ind]).sum()
    logprob_sigma2_y = sigma2_y_prior.log_prob(sigma2_y)

    logprior = logprob_beta + logprob_sigma2_y

    return loglik + logprior

# %%

n_samples = 12000
burn_in = 2000

lam_beta = [0.5] * n_params # Step size for beta
lam_sigma2_y = 0.3  # Step size for sigma2_y

# Set up initial values
beta_current = torch.mean(data_y, dim=0)
sigma2_y_current = torch.std(data_y) ** 2

beta_samples, sigma2_y_samples = [], []
accept_sigma2_y = np.zeros(n_samples)
accept_beta = np.zeros((n_samples, n_params))

logp_current = log_posterior(beta_current, sigma2_y_current, data_y)

for epoch in trange(n_samples):
    for i, beta_index in enumerate(beta_ind):
        # Propose new beta by random walk
        beta_proposal = beta_current.clone()
        beta_proposal[beta_index] = beta_proposal[beta_index] + torch.randn(1) * lam_beta[i]
        if beta_proposal[beta_index] > 0:
            logp_proposal = log_posterior(beta_proposal, sigma2_y_current, data_y)
            log_accept_ratio = logp_proposal - logp_current
            if torch.log(torch.rand(1)) < log_accept_ratio:
                beta_current = beta_proposal
                logp_current = logp_proposal
                accept_beta[epoch, i] = 1

    sigma2_y_proposal = sigma2_y_current + torch.randn(1) * lam_sigma2_y
    if sigma2_y_proposal > 0:
        logp_proposal = log_posterior(beta_current, sigma2_y_proposal, data_y)
        log_accept_ratio = logp_proposal - logp_current
        if torch.log(torch.rand(1)) < log_accept_ratio:
            sigma2_y_current = sigma2_y_proposal
            logp_current = logp_proposal
            accept_sigma2_y[epoch] = 1

    if (epoch % 200 == 0) and (epoch < burn_in):
        # Adjust the step size of random walk
        for i in range(n_params):
            if acc_rate(accept_beta[:, i], epoch + 1) > 0.4:
                lam_beta[i] *= 2
            elif acc_rate(accept_beta[:, i], epoch + 1) < 0.2:
                lam_beta[i] *= 0.3
        if acc_rate(accept_sigma2_y, epoch + 1) > 0.4:
            lam_sigma2_y *= 3
        elif acc_rate(accept_sigma2_y, epoch + 1) < 0.2:
            lam_sigma2_y *= 0.5

    beta_samples.append(beta_current[beta_ind].clone().numpy())
    sigma2_y_samples.append(sigma2_y_current.item())

    if epoch % 200 == 0:
        print(f"Step {epoch}, "
              f"Accept rate of beta0: {acc_rate(accept_beta[:, 0], epoch + 1):.3f}, "
              f"Accept rate of sigma2_y: {acc_rate(accept_sigma2_y, epoch + 1):.3f}")



# %%
beta_samples = np.array(beta_samples)
sigma2_y_samples = np.array(sigma2_y_samples)

# %%

# ===========================
# save the posterior samples
# ===========================

np.savetxt('output/res_flow_net/flow_net_beta_samples.txt', beta_samples, delimiter=',')
np.savetxt('output/res_flow_net/flow_net_sigma2_y_samples.txt', sigma2_y_samples, delimiter=',')

# %%
# Plotting for traces and autocorrelation
plt.figure(figsize=(15, 20))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    component = beta_samples[burn_in:, i]
    plt.plot(component)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'$\\beta_{i}$', fontsize=14)
    
    plt.subplot(5, 2, 2 * i + 2)
    ax = plt.gca()
    plot_acf(component, lags=40, ax=ax, title=None, auto_ylims=True)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Autocorrelation', fontsize=14)

plt.tight_layout(pad=3)
plt.show()

# %%
# Plotting for traces and autocorrelation
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(sigma2_y_samples[burn_in:])
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(f'$\\sigma^2$', fontsize=14)

plt.subplot(1, 2, 2)
ax = plt.gca()
plot_acf(sigma2_y_samples[burn_in:], lags=40, ax=ax, title=None, auto_ylims=True)
plt.xlabel('Lag', fontsize=14)
plt.ylabel('Autocorrelation', fontsize=14)

plt.tight_layout(pad=3)
plt.show()

# %%
# ======================
# histgrams of samples
# ======================

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.hist(beta_samples[burn_in:, i], bins=30, density=True, alpha=0.7)
    plt.axvline(np.asarray(z_truth)[beta_ind[i]], color='red', linestyle='--', linewidth=2)
    plt.xlabel(f"$\\beta_{i}$", fontsize=14)
    plt.ylabel("Density", fontsize=14)
plt.tight_layout(pad=3)
plt.show()

# %%
