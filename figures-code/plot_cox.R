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
plot_acf(
    res_cano[thinned_idx], filename="acf_cox_cano.png",
    lag_max=lag_max, width=3.3, height=2)

# acf of b for bridged model
plot_acf(
    res_bri[thinned_idx], filename="acf_cox_bri.png",
    lag_max=lag_max, width=3.3, height=2)


####################################
#### plot density of lambda ###
####################################

ggplot(
    data = data.frame(lbd = res_bri), aes(x = lbd, y = after_stat(density))) +
    geom_histogram(bins=50) + theme_bw() + xlim(c(0.5, 0.9)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14)) +
    ylab('Density') + xlab(bquote(lambda))

ggsave("density_cox_bri.pdf", width=4.5, height=2.5, units='in')

ggplot(
    data = data.frame(lbd = res_cano), aes(x = lbd, y = after_stat(density))) +
    geom_histogram(bins=50) + theme_bw() + xlim(c(0.5, 0.9)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14)) +
    ylab('Density') + xlab(bquote(lambda))

ggsave("density_cox_cano.pdf", width=4.5, height=2.5, units='in')
