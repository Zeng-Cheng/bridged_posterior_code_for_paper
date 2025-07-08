## Bridged Posterior: Optimization, Profile Likelihood and a New Approach to Generalized Bayes

This repository contains the reproducibility materials for the paper: Bridged Posterior: Optimization, Profile Likelihood and a New Approach to Generalized Bayes.

### Packages needed in the repository

- Python packages: torch, numpy, matplotlib, jax, scipy, tqdm, polyagamma, sklearn, hamiltorch, pyro, networkx, statsmodels, numpyro
- R packages: ggplot2, coda, scales, dplyr, tidyr, kernlab, reshape2, npmr, INLA


### Guidance of using the codes to reproduce the figures and tables in the paper

#### Organization

simulation-code contains codes for simulations.

real-data-code contains codes for Section 4.

figures-code contains codes for producing all figures and tables except for Figure 3 and S2(b).

output contains results from running the code in simulation-code and real-data-code.

#### Workflow

- Figure 3 is not produced by codes.
- Results in Section 2.2 and S5.1.
    - Run `simulation-code/latent_gau.py` to sample simulated data in `data`, run MCMC to produce results in `output/res_latent_gau/`.
    - Run `simulation-code/latent_gau_sim.py` to produce the variances comparison results in `output/res_latent_gau/`.
    - Run `simulation-code/latent_gau_vi_pyro.py` to produce results in `output/res_latent_gau_vi/`
    - Run `figures-code/plot_latent_gau.R` to produce the numbers in Section 2.2 and Figure 1, 2 and S2(a).
    - Run `simulation-code/latent_gau_inla.R` to produce Figure S2(b).
- Results in Section 4.
    - Run `data_application_runmcmc.R` to produce the results in `output/res_hfcg/`
    - Run `figures-code/plot_hfcg.R` to produce Figure 4, 5 and 6.
- Results in Section S2.
    - Run `simulation-code/cox_compare.py` to produce results in `output/res_cox/` for plotting.
    - Run `figures-code/plot_cox.R` to produce Figure S1.
- Results in Section S5.3.
    - Run `simulation-code/flow_net.py` to produce results in `output/res_flow_net/` for plotting.
    - Run `figures-code/plot_svm.R` to produce Figure S3 and S4.
- Results in Section S5.4.
    - Run `simulation-code/semi_svm.py` to produce the numbers in Table S1 and results in `output/res_svm/` for plotting.
    - Run `figures-code/plot_svm.R` to produce Figure S5 and S6.