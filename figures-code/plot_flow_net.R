library(ggplot2)
library(scales)
library(dplyr)
library(tidyr)
source('src/plot_mcmc.R')

edges = c("1", "2", "3", "4", "5")

beta_samples = read.table("output/res_flow_net/flow_net_beta_samples.txt",
    sep = ",")
sigma2_y_samples = unlist(read.table(
    "output/res_flow_net/flow_net_sigma2_y_samples.txt", sep = ","))

nrow(beta_samples)

thinning = seq(2001, nrow(beta_samples), by = 10)

######################################
# Plot histograms for each dimension #
######################################

nbreaks = rep(5, 5)

for (i in c(1:5)) {
    ggplot(
        data = data.frame(x = beta_samples[thinning, i]),
        aes(x = x, y = after_stat(density))) +
        geom_histogram(bins=50) + theme_bw() +
        xlab(bquote(lambda[.(edges[i])])) + ylab('Density') +
        scale_x_continuous(n.breaks = 3) +
        theme(
            axis.title.x = element_text(size = 16),
            axis.title.y = element_text(size = 16),
            axis.text.x = element_text(size = 13),
            axis.text.y = element_text(size = 13)
        ) +
        theme(plot.margin = margin(8, 15, 3, 3))

    ggsave(
        paste("hist_flow_beta", i, ".pdf", sep=""),
        width=2.2, height=1.5, units="in")
}

##########################################
# acf plots for beta and sigma2_y #
##########################################

# plot box plots of ACFs of all beta
p <- ncol(beta_samples)
num_iters <- nrow(beta_samples)
lag_max <- 40
thinned_idx <- seq(2001, num_iters, by = 10)

acf_w <- c()
for (i in 1:p) {
    cur_acf <- as.numeric(acf(
        beta_samples[thinned_idx, i],
        lag.max = lag_max, plot = FALSE)[[1]])
    acf_w <- c(acf_w, cur_acf)
}

df_acf_w <- data.frame(
    ACF = acf_w,
    Lag = c(0:lag_max),
    dim = factor(rep(1:p, each = lag_max + 1))
)

ggplot(data = df_acf_w, aes(Lag, ACF, group = Lag)) +
    geom_boxplot() + theme_bw() +
    theme(
        axis.title.x = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14))

ggsave("acf_flow_beta.pdf", width=4.5, height=2.2, units='in')

# acf of sigma2
plot_acf(sigma2_y_samples[thinned_idx],
    filename="acf_flow_sigma2.pdf", lag_max=lag_max, width=4.5, height=2.2)