# The data are not included in the Github repository
# if one needs to run the data application (either runmcmc or plot_hfcg)
# first download the files in the following link to data folder
url = 'https://www.dropbox.com/scl/fo/pwxznmap3cm97iycuobw8/h?rlkey=of0spg09mgh3ly9a38svnktce&e=1&dl=0'

load("data/graph_fmri_all.Rda") 
# 'list_A' contains correlation matrices for each subject
S <- length(list_A)
load("data/fmri_labels.RDa") # Labels for each subject in 'label_list'
labels = sapply(label_list, function(ll) {
    if(ll == "CN") return(0)
    else if (ll == "LMCI") return(1)
    else return(2)
    })

# 'Z_all'
load("data/Z_uncommon_lam_all.RData") # reduced rank matrices at diff lambdas

# 'geodist'
load("data/geodist_uncommon_lam_all.RData") # dist between L at diff lambdas

# 'lambdas_all'
load("data/lam_uncommon_all.RData") # all possible lambda values for each s


laps <- list() # Laplacians for each subject
for (s in 1:S) {
    laps[[s]] <- diag(rowSums(list_A[[s]])) - list_A[[s]]
}

###########################################################################
## this part is for running MCMC of posterior sampling, we have saved the
## results in a .Rda file which can be read directly
###########################################################################

run_mcmc = FALSE
if run_mcmc {
    source("src/hfcg_gibbs_sampler.R")
    
    post_samples <- gibbs_sampler(laps, Z_all, lambdas_all, geodist, num_iters)
    save(post_samples, file="hfcg_gibbs_res.Rda")
}

###########################################################################
## end of the part for running MCMC of posterior sampling
###########################################################################