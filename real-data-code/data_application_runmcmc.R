# The data are not included in the Github repository
# if one needs to run the data application (either runmcmc or plot_hfcg)
# first download the files in the following link to data folder
url = 'https://doi.org/10.5281/zenodo.16824059'

load("data/graph_fmri.Rda")
# 'listA' contains correlation matrices for each subject
S <- length(listA)
load("data/fmri_labels.Rda") # Labels for each subject in 'label_list'
labels = sapply(label_list, function(ll) {
    if(ll == "CN") 0
    else if (ll == "LMCI") 1
    else 2
})

# 'Zres'
load("data/hfcg_Z.Rda") # reduced rank matrices at diff lambdas

# 'geodist'
load("data/geodist_uncommon_lam_all.Rda") # dist between L at diff lambdas

# 'lambdas_all'
load("data/lam_uncommon_all.Rda") # all possible lambda values for each s

# Laplacians for each subject
laps <- lapply(1:S, function(s) {
    diag(rowSums(listA[[s]])) - listA[[s]]})

###########################################################################
## this part is for running MCMC of posterior sampling, we have saved the
## results in a .Rda file which can be read directly
###########################################################################

num_iters = 10000
run_mcmc = TRUE
if (run_mcmc) {
    source("src/hfcg_gibbs_sampler.R")
    post_samples <- gibbs_sampler(laps, Zres, lambdas_all, geodist, num_iters)
    save(post_samples, file="output/res_hfcg/hfcg_gibbs_res_test.Rda")
}

###########################################################################
## end of the part for running MCMC of posterior sampling
###########################################################################