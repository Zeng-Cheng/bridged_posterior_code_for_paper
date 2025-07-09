library(INLA)

set.seed(123)
n <- 1000
x <- unlist(read.table("data/latent_gau_x.txt"))
y <- unlist(read.table("data/latent_gau_y.txt"))
data <- data.frame(x = x, response = y)

rgeneric_gpkernel <- function(
    cmd = c("graph","Q","mu","initial","log.norm.const","log.prior","quit"),
    theta = NULL) {
    interpret.theta <- function() {
        list(tau = exp(theta[1]), b = exp(theta[2]))
    }
    # Graph: full (all entries non-zero)
    graph <- function() {
        require(Matrix); n <- length(x)
        return(Matrix(1, nrow=n, ncol=n, sparse=TRUE))
    }
    # Precision matrix Q
    Q <- function() {
        require(Matrix)
        param <- interpret.theta()
        tau <- param$tau;  b <- param$b
        D <- as.matrix(dist(x))
        Cov <- tau * exp(- D^2 / (2*b)) + diag(1e-6, length(x))
        Qmat <- solve(Cov)
        return(as(Qmat, "dgCMatrix"))
    }
    # Mean = 0
    mu <- function() return(numeric(0))
    # Let INLA compute normalization
    log.norm.const <- function() return(numeric(0))
    # Prior for (tau,b) (with jacobians)
    log.prior <- function() {
        param <- interpret.theta()
        tau <- param$tau;  b <- param$b
        # Half-normal prior on tau (σ=1 example)
        logp_tau <- dnorm(tau, 0, 1, log=TRUE) + log(2) + log(tau)
        # Inverse-gamma on b (shape=2, scale=1 example)
        a <- 2; s <- 1
        logp_b <- (a*log(s) - lgamma(a)) - (a+1)*log(b) - s/b + log(b)
        return(logp_tau + logp_b)
    }
    initial <- function() return(c(log(2),log(2)))
    # theta starting values (-> tau=2, b=2)
    quit <- function() return(invisible())
    # Dispatch the function
    res <- switch(match.arg(cmd),
        graph = graph(),
        Q = Q(),
        mu = mu(),
        log.norm.const = log.norm.const(),
        log.prior = log.prior(),
        initial = initial(),
        quit = quit())
    return(res)
}

# Define the rgeneric model, passing data x into the environment
gpkernelModel <- inla.rgeneric.define(rgeneric_gpkernel, x = x)

data$idx <- 1:nrow(data)
formula <- response ~ 1 + f(idx, model=gpkernelModel)
result <- inla(formula, family="binomial", data=data)

# Plot fitted effect over x
plot(x, y, col = "gray", pch = 16, main = "Posterior Mean of ζ(x)")
points(x, result$summary.fitted.values$mean, col = "blue", lwd = 2)


# Sample from posterior of hyperparameters
samples <- inla.hyperpar.sample(10000, result)

# Extract log precision and log range
log_tau <- samples[, "Theta1 for idx"]
log_b <- samples[, "Theta2 for idx"]

res_tau <- exp(log_tau)
res_b <- exp(log_b)

# Create scatter plot
library(ggplot2)

ggplot(data = data.frame(b = res_b, tau = res_tau), aes(x = tau, y = b)) +
    geom_point(color = "black", size = 0.5) +
    theme_bw() + xlim(0, 5) + ylim(0, 7) +
    xlab(bquote(tau)) + ylab(bquote(b)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        legend.position = "none"
    )

# Save to file
ggsave("scatter_latent_gau_inla.pdf", width = 4, height = 2.5, units = 'in')
