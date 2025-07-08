library(ggplot2)
library(coda)
source('src/plot_mcmc.R')

#### input results

# load Bridged results for b
res_b_bri <- unlist(read.table("output/res_latent_gau/b_trace.txt"))
# load Canonical results for b
res_b_cano <- unlist(read.table(
    "output/res_latent_gau/b_aug_trace.txt"))

# load Bridged results for tau
res_tau_bri <- unlist(read.table("output/res_latent_gau/tau_trace.txt"))
# load Canonical results for tau
res_tau_cano <- unlist(read.table(
    "output/res_latent_gau/tau_aug_trace.txt"))


num_samples <- length(res_b_bri)
burn_in = 2000

#######################################################
### calculate ESS/time ###
#######################################################

thinned_idx <- seq(1, length(res_b_bri), by = 1)

# ESS/time for b
round(mean(effectiveSize(res_b_bri[thinned_idx]) /
            length(res_b_bri[thinned_idx])) / 8.37 * 6, 3)

round(mean(effectiveSize(res_b_cano[thinned_idx]) /
            length(res_b_cano[thinned_idx])) / 11.87 * 6, 4)

# ESS/time for tau
round(mean(effectiveSize(res_tau_bri[thinned_idx]) /
            length(res_tau_bri[thinned_idx])) / 8.37 * 6, 3)

round(mean(effectiveSize(res_tau_cano[thinned_idx]) /
            length(res_tau_cano[thinned_idx])) / 11.87 * 6, 4)


#######################################################
### calculate the posterior variances (std) ###

# std from the latent quadratic exponential model
thinned_idx <- seq(burn_in + 1, num_samples, by=10)
round(sqrt(var(res_b_bri[thinned_idx])), 2)
round(sqrt(var(res_tau_bri[thinned_idx])), 2)

# std from the latent normal model
thinned_idx <- seq(burn_in + 1, num_samples, by=30)
round(sqrt(var(res_b_cano[thinned_idx])), 2)
round(sqrt(var(res_tau_cano[thinned_idx])), 2)


#######################################################
### plot traces ###

# trace of b for latent normal model
plot_trace(
    res_b_cano, ylims=c(0, 7), yname="b",
    filename="trace_latent_gau_b_cano.png", width=3.3, height=2)

# trace of b for latent quadratic model
plot_trace(
    res_b_bri, ylims=c(0, 7), yname="b",
    filename="trace_latent_gau_b_bri.png", width=3.3, height=2)

# trace of tau for latent normal model
plot_trace(
    res_tau_cano, ylims=c(0, 5), yname=bquote(tau),
    filename="trace_latent_gau_tau_cano.png", width=3.3, height=2)

# trace of tau for latent quadratic model
plot_trace(
    res_tau_bri, ylims=c(0, 5), yname=bquote(tau),
    filename="trace_latent_gau_tau_bri.png", width=3.3, height=2)


###################################
### plot acfs ###
###################################

lag_max <- 40 # number of lags
thinned_idx <- seq(1, num_samples, by = 1) # no thinning

# acf of b for canonical model
plot_acf(
    res_b_cano[thinned_idx], filename="acf_latent_gau_b_cano.png",
    lag_max=lag_max, width=3.3, height=2)

# acf of b for bridged model
plot_acf(
    res_b_bri[thinned_idx], filename="acf_latent_gau_b_bri.png",
    lag_max=lag_max, width=3.3, height=2)

# acf of tau for canonical model
plot_acf(
    res_tau_cano[thinned_idx], filename="acf_latent_gau_tau_cano.png",
    lag_max=lag_max, width=3.3, height=2)

# acf of tau for bridged model
plot_acf(
    res_tau_bri[thinned_idx], filename="acf_latent_gau_tau_bri.png",
    lag_max=lag_max, width=3.3, height=2)

############################################
### scatter plots of posterior samples ###
############################################

# scatter plots for canonical model

ggplot(data = data.frame(
    b = res_b_cano[-(1:burn_in)],
    tau = res_tau_cano[-(1:burn_in)]
), aes(x=tau, y=b)) +
    geom_point(color = "black", size = 0.5) +
    theme_bw() + xlim(0, 5) + ylim(0, 7) + xlab(bquote(tau)) + ylab(bquote(b)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15)) +
    theme(legend.position = "none")

ggsave("scatter_latent_gau_canonical.png", width=4, height=2.5, units='in')

# scatter plot for bridged model

ggplot(data = data.frame(
    b = res_b_bri[-(1:burn_in)],
    tau = res_tau_bri[-(1:burn_in)]
), aes(x=tau, y=b)) +
    geom_point(color = "black", size = 0.5) +
    theme_bw() + xlim(0, 5) + ylim(0, 7) + xlab(bquote(tau)) + ylab(bquote(b)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15)) +
    theme(legend.position = "none")

ggsave("scatter_latent_gau_bridge.png", width=4, height=2.5, units='in')



####################################################
### repeated experiments; compare the variances ###
####################################################

ns_all <- c(50, 200, 500, 1000) # sample sizes
curve_all <- c(1:9)
res_b_bri <- c()
res_b_cano <- c()
res_tau_bri <- c()
res_tau_cano <- c()

for (curve in curve_all)
for (ns in ns_all) {
    filename <- paste("output/res_latent_gau/b_trace_curve",
        curve, "ss", ns, ".txt", sep='')
    res_b_bri <- cbind(res_b_bri, unlist(read.table(filename)))
    filename <- paste("output/res_latent_gau/b_aug_trace_curve",
        curve, "ss", ns, ".txt", sep='')
    res_b_cano <- cbind(res_b_cano, unlist(read.table(filename)))
    filename <- paste("output/res_latent_gau/tau_trace_curve",
        curve, "ss", ns, ".txt", sep='')
    res_tau_bri <- cbind(res_tau_bri, unlist(read.table(filename)))
    filename <- paste("output/res_latent_gau/tau_aug_trace_curve",
        curve, "ss",  ns, ".txt", sep='')
    res_tau_cano <- cbind(res_tau_cano, unlist(read.table(filename)))
}


thinned_idx <- seq(burn_in + 1, nrow(res_b_bri), by=10)
var_b <- diag(var(res_b_bri[thinned_idx, ]))
var_tau <- diag(var(res_tau_bri[thinned_idx, ]))
thinned_idx <- seq(burn_in + 1, nrow(res_b_cano), by=30)
var_b <- c(var_b, diag(var(res_b_cano[thinned_idx, ])))
var_tau <- c(var_tau, diag(var(res_tau_cano[thinned_idx, ])))

label <- c("Latent quadratic exponential model", "Latent normal model")

# plot for parameter tau

var_tau_res <- data.frame(
    Variance = c(var_tau),
    Model = rep(label, each = length(ns_all) * length(curve_all)),
    ss = factor(ns_all)
)

ggplot(var_tau_res, mapping=aes(x=ss, y=Variance, color=Model)) +
    geom_boxplot() + theme_bw() + xlab("Sample Size") +
    theme(
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 11),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11)) +
    theme(legend.position = "none") +
    theme(
        legend.text=element_text(),
        legend.title=element_text(size=14),
        panel.grid.minor=element_blank()) +
    scale_color_manual(values = c("#429dfb", '#7bd34b'))

ggsave("var_tau_latent.png", width=4, height=2.5, unit="in")


# plot for parameter b

var_b_res <- data.frame(
    Variance = c(var_b),
    Model = rep(label, each = length(ns_all) * length(curve_all)),
    ss = factor(ns_all)
)

ggplot(var_b_res, mapping=aes(x=ss, y=Variance, color=Model)) +
    geom_boxplot() + theme_bw() + xlab("Sample Size") +
    theme(
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 11),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11)) +
    theme(legend.position = "none") + 
    theme(
        legend.text=element_text(),
        legend.title=element_text(size=14),
        panel.grid.minor=element_blank()) +
    scale_color_manual(values = c("#429dfb", '#7bd34b'))

ggsave("var_b_latent.png", width=4, height=2.5, unit="in")


########################################
#### plot for variational inference ####
########################################

#### input results
# load results for b
res_b <- unlist(read.table(
    "output/res_latent_gau_vi/latent_gau_b_samples.txt"))
# load results for tau
res_tau <- unlist(read.table(
    "output/res_latent_gau_vi/latent_gau_tau_samples.txt"))

# scatter plots
ggplot(data = data.frame(b = res_b, tau = res_tau), aes(x=tau, y=b)) +
    geom_point(color = "black", size = 0.5) +
    theme_bw() + # xlim(0, 5) + ylim(0, 7) +
    xlab(bquote(tau)) + ylab(bquote(b)) +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15)) +
    theme(legend.position = "none")

ggsave("scatter_latent_gau_vi.png", width=4, height=2.5, units='in')