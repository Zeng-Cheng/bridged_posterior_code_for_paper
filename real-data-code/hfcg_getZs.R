load("data/graph_fmri.Rda")
# 'listA' is the list with all the adjacency matrices for each subject
S <- 166
p <- dim(listA[[1]])[1]

laps <- lapply(1:S, function(s) {diag(rowSums(listA[[s]])) - listA[[s]]})

# normalized Laplacians
# norm_lap <- list()
# for (s in 1:S) {
#     d_mat <- diag(rowSums(listA[[s]]))
#     lap <- d_mat - listA[[s]]
#     D_sqrt_inv <- solve(sqrt(d_mat))
#     norm_lap[[s]] <- D_sqrt_inv %*% lap %*% D_sqrt_inv
# }

load('data/lam_uncommon_all.Rda') # 'lambdas_all' contains all lambda
# these lambdas are predetermined

require("parallel") # parallel computation
no_cores <- detectCores() - 2 # number of CPU cores
prlcl = makeCluster(no_cores) # parallel computing
clusterExport(prlcl, c('p', 'laps', 'S', 'lambdas'))
clusterEvalQ(prlcl, {
    library(Rcpp)
    library(RcppArmadillo)
    sourceCpp('src/optimization.cpp')})

Zres <- parLapply(prlcl, 1:S, function(i) {
    LA <- laps[[i]]
    lapply(lambdas[[i]], function(lam) {
        ADMM(LA, LA, lam, maxADMMIterations = 100, eta = 5,
            maxGradientDescentIterations = 30, target_rho = 1E-7,
            alpha = 0.0000005, tol = 0.00001, tol_total_loss = 0.000001,
            start_rho = 1E-4)
    })
})
stopCluster(prlcl)
# save the result
save(Zres, file="data/hfcg_Z.Rda")

##########################################
# number of communities for all subjects #
##########################################

# for(s in 1:166) {
#     Klam = length(lambdas[[s]])
#     print(sapply(1:Klam, function(j) {
#         currZ <- Zres[[s]][[j]]
#         diag(currZ) <- -1
#         currZ[currZ > -0.00002] = 0
#         diag(currZ) <- 0
#         diag(currZ) <- -rowSums(currZ)
#         sum(svd(currZ)$d < 0.000001)
#     }))
# }

#########################
# calculate distances ###
#########################

require("parallel") # parallel computation
no_cores <- detectCores() - 2 # number of CPU cores
prlcl = makeCluster(no_cores) # parallel computing
clusterExport(prlcl, c('p', 'S', 'Zres'))

geodist = parLapply(prlcl, 1:S, function(s) {
    LAs = Zres[[s]] # the Zres for each lambda
    Klams = length(LAs)
    # produce a list of length S, where each element is the matrix of
    # distances between different lambdas
    lapply(1:S, function(k) {
        LAk = Zres[[k]] # the Zres for each lambda
        Klamk = length(LAk)
        res = matrix(0, Klams, Klamk)
        for (i in 1:Klams) {
            res[i, ] <- sapply(1:Klamk, function(j) {
                a_inverse_b = solve(
                    LAs[[i]] + diag(1, p) * 1e-5, LAk[[j]] + diag(1, p) * 1e-5)
                evs = Re(eigen(a_inverse_b)$values)
                sqrt(sum((log(evs)) ^ 2))
            })
        }
        res
    })
})
stopCluster(prlcl)
save(geodist, file="data/geodist_uncommon_lam_all.Rda")
