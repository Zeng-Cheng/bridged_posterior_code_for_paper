library(ggplot2)
library(coda)
source('src/plot_mcmc.R')


#### input results

# load Bridged results for b
res_bri <- unlist(read.table("output/res_cox/cox_samples_bridged.txt"))
# load Canonical results for b
res_cano <- unlist(read.table("output/res_cox/cox_samples_full.txt"))

num_samples <- length(res_bri)
burn_in = 2000

#######################################################
### calculate the posterior variances (std) ###

# std from the latent quadratic exponential model
thinned_idx <- seq(burn_in + 1, num_samples, by=10)
round(sqrt(var(res_bri[thinned_idx])), 2)
round(sqrt(var(res_cano[thinned_idx])), 2)

#######################################################
### plot traces ###

# trace for full Bayesian model
plot_trace(
    res_cano, ylims=c(0.4, 1.1), yname=bquote(lambda),
    filename="trace_cox_cano.png", width=3.3, height=2)

# trace for bridged model
plot_trace(
    res_bri, ylims=c(0.4, 1.1), yname=bquote(lambda),
    filename="trace_cox_bri.png", width=3.3, height=2)


###################################
### plot acfs ###
###################################

lag_max <- 40 # number of lags
thinned_idx <- seq(1, num_samples, by = 1) # no thinning

# acf of b for canonical model

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_cano[thinned_idx],
                lag.max = lag_max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag_max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
    geom_bar(stat = "identity", width = 0.5) + theme_bw() +
    theme(
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

ggsave("acf_cox_cano.png", width=3.3, height=2, units='in')

# acf of b for bridged model

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_bri[thinned_idx],
                lag.max = lag_max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag_max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
    geom_bar(stat = "identity", width = 0.5) + theme_bw() +
    theme(
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

ggsave("acf_cox_bri.png", width=3.3, height=2, units='in')


####################################
#### plot density of lambda ###
####################################

ggplot(data = data.frame(lbd = res_bri), aes(lbd)) +
    geom_density(color = 'black') + theme_bw() + xlim(c(0.5, 0.9)) +
    theme(
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
    ylab('Density') + xlab(bquote(lambda))

ggsave("density_cox_bri.png", width=4.5, height=2.5, units='in')

ggplot(data = data.frame(lbd = res_cano), aes(lbd)) +
    geom_density(color = 'black') + theme_bw() + xlim(c(0.5, 0.9)) +
    theme(
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
    ylab('Density') + xlab(bquote(lambda))

ggsave("density_cox_cano.png", width=4.5, height=2.5, units='in')