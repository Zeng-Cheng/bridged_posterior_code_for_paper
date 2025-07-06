library(INLA)

set.seed(123)
n <- 1000
X <- unlist(read.table("data/latent_gau_x.txt"))
y <- unlist(read.table("data/latent_gau_y.txt"))

# Build 1D mesh over X
mesh <- inla.mesh.1d(loc = X, boundary = "free")

# Define SPDE model (1D Matérn GP)
spde <- inla.spde2.matern(mesh = mesh, alpha = 2)

# Projection matrix from mesh to observed locations
A <- inla.spde.make.A(mesh = mesh, loc = X)

# Index for spatial effect
index <- inla.spde.make.index("spatial", n.spde = spde$n.spde)

# Stack the data for INLA
stack <- inla.stack(
  data = list(y = y),
  A = list(A, 1),
  effects = list(
    spatial = index$spatial,
    intercept = rep(1, n)
  ),
  tag = "est"
)

# Fit model: y ~ ζ(X) + intercept
formula <- y ~ 0 + intercept + f(spatial, model = spde)

result <- inla(
  formula,
  family = "binomial",
  data = inla.stack.data(stack),
  control.predictor = list(A = inla.stack.A(stack), compute = TRUE),
  control.compute = list(dic = TRUE, waic = TRUE),
  verbose = FALSE
)


# Posterior of latent field ζ
zeta_mean <- result$summary.random$spatial$mean
zeta_sd <- result$summary.random$spatial$sd

# Posterior of hyperparameters (log precision and log range)
inla.hyperpar.summary(result)

# Plot fitted effect over X
plot(X, y, col = "gray", pch = 16, main = "Posterior Mean of ζ(x)")
lines(X, result$summary.fitted.values$mean, col = "blue", lwd = 2)


# Sample from posterior of hyperparameters
samples <- inla.hyperpar.sample(1000, result)

# Extract log precision and log range
log_prec <- samples[, "Theta1 for spatial"]
log_range <- samples[, "Theta2 for spatial"]

# Transform to natural scale
res_tau <- exp(-log_prec)    # τ = 1 / exp(log precision)
res_b <- exp(log_range)      # b = range

# Create scatter plot in your format
library(ggplot2)

ggplot(data = data.frame(b = res_b, tau = res_tau), aes(x = tau, y = b)) +
    geom_point(color = "black", size = 0.5) +
    theme_bw() +
    xlab(bquote(tau)) + ylab(bquote(b)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        legend.position = "none"
    )

# Save to file
ggsave("scatter_latent_gau_inla.png", width = 4, height = 2.5, units = 'in')
