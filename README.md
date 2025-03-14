### Bridged Posterior: Optimization, Profile Likelihood and a New Approach to Generalized Bayes

This repository contains the reproducibility materials for the paper: Bridged Posterior: Optimization, Profile Likelihood and a New Approach to Generalized Bayes.

#### Abstract

Optimization is widely used in statistics, thanks to its efficiency for delivering point estimates on useful spaces, such as those satisfying low cardinality or combinatorial structure. To quantify uncertainty, Gibbs posterior exponentiates the negative loss function to form a posterior density. Nevertheless, Gibbs posteriors are supported in high-dimensional spaces, and do not inherit the computational efficiency or constraint formulations from optimization. In this article, we explore a new generalized Bayes approach, viewing the likelihood as a function of data, parameters, and latent variables conditionally determined by an optimization sub-problem. Marginally, the latent variable given the data remains stochastic, and is characterized by its posterior distribution. This framework, coined bridged posterior, conforms to the Bayesian paradigm. Besides providing a novel generative model, we obtain a positively surprising theoretical finding that under mild conditions, the $\sqrt{n}$-adjusted posterior distribution of the parameters under our model converges to the same normal distribution as that of the canonical integrated posterior. Therefore, our result formally dispels a long-held belief that partial optimization of latent variables may lead to under-estimation of parameter uncertainty. We demonstrate the practical advantages of our approach under several settings, including maximum-margin classification, latent normal models, and harmonization of multiple networks.

#### Packages needed in the repository

- Python packages: torch, numpy, jax, scipy, tqdm, polyagamma
- R packages: ggplot2, coda

#### Guidance of using the codes to reproduce the figures and tables in the paper

- Figure 1 is not produced by codes.
- Results in Section 5.
    - First, in `latent_gau.ipynb`, we sample some simulated data, run MCMC to produce the posterior samples, and save the samples in `output/res_latent_gau/`.
    - We use `latent_gau_sim.py` to produce the variances comparison posterior samples as described in Section S.5.
    - Figure 2 and 3 are plotted by `plot_latent_gau.R`.
- Results in Section 6.
    - `data_application_runmcmc.R` 
    - `plot_hfcg.R`
- Results in Section S.6.
    - `semi_svm.ipynb`
    - `plot_svm.R`