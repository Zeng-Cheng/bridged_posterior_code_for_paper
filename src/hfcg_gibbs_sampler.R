require(npmr)

gibbs_sampler <- function(
    list_A, fmri_result, list_lambdas, dist_mat, num_iterations
) {
    S <- length(list_A)
    Klam = length(list_lambdas[[1]]) # assume same number of lams for each s
    # get all losses of ||L_A-L||
    loss_all = sapply(1:S, function(s) {
        list_dist = c()
        for(j in 1:Klam) {
            # LAs <- diag(rowSums(list_A[[s]])) - list_A[[s]]
            LAs = list_A[[s]] # list_A is alreadly Laplacian
            list_dist = c(list_dist, sum((LAs - fmri_result[[s]][[j]]) ^ 2))
        }
        return(list_dist)
    })
    # get all nuclear norms of ||L||
    star_norm_all = sapply(1:S, function(s) {
        list_dist = c()
        for(j in 1:Klam) {
            L = fmri_result[[s]][[j]]
            list_dist = c(list_dist, nuclear(L))
        }
        return(list_dist)
    })

    dist_mean <- colMeans(matrix(unlist(dist_mat), nrow=Klam ^ 2))
    alpha_gamma <- 2; beta_gamma <- var(dist_mean) # inv-gamma prior for gamma

    # inv-gamma prior for sigma2
    alpha_sigma <- 2; beta_sigma <- var(c(colMeans(sqrt(loss_all)))) 

    # inv-gamma prior for lambdas
    alpha <- 2
    beta <- 2 * mean(sapply(1:S, function(s) mean(list_lambdas[[s]]))) 

    updateLambda <- function(lambdas) {
        for(s in 1:S){
            # Update lambda_s one by one
            # record the positions of lambda's
            # loglik is all the loglik for the possible lambda values
            loglik <- (
                # (-alpha - 1) * log(list_lambdas[[s]]) -
                # beta / list_lambdas[[s]] -
                -loss_all[, s] / (2 * sigma2) -
                list_lambdas[[s]] * star_norm_all[, s] / sigma2
            )
            for(j in 1:Klam){
                # get all distances between sub s and other subs
                all_dist = sapply(1:S, function(k) {
                    if ((k != s)){ # & (labels[k] == labels[s])
                        return(dist_mat[[s]][[k]][j, lambdas[k]] ^ 2)
                    }
                    else {
                       return(c(0))
                    }
                })
                loglik[j] <- loglik[j] - sum(all_dist) / (4 * S * gamma)
            }

            gumbel = -log(-log(runif(Klam, min=0, max=1)))
            lambdas[s] <- which.max(loglik + gumbel)
        }
        return(lambdas)
    }

    updateGamma <- function(lambdas) {
        shape <- alpha_gamma + (S - 1) / 2
        sum_dist = 0
        for(s in 1:S){
            for(k in 1:S){
                if ((k != s)) { # & (labels[k] == labels[s])
                    sum_dist <- (sum_dist +
                        dist_mat[[s]][[k]][lambdas[s], lambdas[k]] ^ 2)
                }
            }
        }
        rate <- beta_gamma + sum_dist / 2 / (4 * S)
        gamma <- 1 / rgamma(1, shape, rate)
        if(is.na(gamma)) {
            gamma <- rate / shape
        }
        return(gamma)
    }

    updateSigma2 <- function(lambdas) {
        shape <- alpha_sigma + S / 2
        all_losses <- sapply(1:S, function(s) {
            return(loss_all[lambdas[s], s])
        })
        all_norms <- sapply(1:S, function(s) {
            return(list_lambdas[[s]][lambdas[s]] * 
                star_norm_all[lambdas[s], s])
        })
        rate <- beta_sigma + sum(all_losses) / 2 + sum(all_norms)
        sigma2 <- 1 / rgamma(1, shape, rate)
        if(is.na(sigma2)) {
            sigma2 <- rate / shape
        }
        return(sigma2)
    }

    # initial values
    lambdas <- sapply(1:S, function(s) {sample(1:Klam, size=1, replace=TRUE)})
    gamma <- 1 / rgamma(1, alpha_gamma, beta_gamma)
    sigma2 <- 1 / rgamma(1, alpha_sigma, beta_sigma)

    # Create vectors to store samples
    lambdas_samples <- list()
    gamma_samples <- c()
    sigma_samples <- c()

    starttime = proc.time() # record time

    for (i in 1:num_iterations){

        lambdas <- updateLambda(lambdas)
        gamma <- updateGamma(lambdas)
        sigma2 <- updateSigma2(lambdas)

        lambdas_samples[[i]] <- lambdas
        gamma_samples <- c(gamma_samples, gamma)
        sigma_samples <- c(sigma_samples, sigma2)

        if(i %% 50 == 0) {
            print(i)
            print(lambdas)
            print(gamma)
            print(sigma2)
        }

        if(i == 200) print(paste(
            "Est Time: ",
            (proc.time() - starttime)[3] / i * num_iterations / 3600 # in hours
        ))
    }
    return(list(
        lambdas = lambdas_samples, 
        gamma = gamma_samples, sigma2 = sigma_samples))
}