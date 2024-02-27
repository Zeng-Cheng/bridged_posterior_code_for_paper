library(ggplot2)
library(coda)

#### input results

res_b_bri <- unlist(read.table(
    "res_latent_gau/b_trace.txt")) # load Bridged results
res_b_cano <- unlist(read.table(
    "res_latent_gau/b_aug_trace.txt"))[1:10000] # load Canonical results

res_tau_bri <- unlist(read.table(
    "res_latent_gau/tau_trace.txt")) # load Bridged results
res_tau_cano <- unlist(read.table(
    "res_latent_gau/tau_aug_trace.txt"))[1:10000] # load Canonical results

num_samples <- length(res_b_bri)
burn_in = 2000

### calculate ESS

thinned_idx <- seq(1, length(res_b_bri), by = 1)

mean(effectiveSize(res_b_bri[thinned_idx]) /
        length(res_b_bri[thinned_idx])) / 8.37 * 6
mean(effectiveSize(res_b_cano[thinned_idx]) /
        length(res_b_cano[thinned_idx])) / 11.87 * 6

mean(effectiveSize(res_tau_bri[thinned_idx]) /
        length(res_tau_bri[thinned_idx])) / 8.37 * 6
mean(effectiveSize(res_tau_cano[thinned_idx]) /
        length(res_tau_cano[thinned_idx])) / 11.87 * 6

### plot traces

# trace of b

trace_latent_gau = data.frame(
    b = c(res_b_cano),
    Iteration = c(1:num_samples)
)

ggplot(data = trace_latent_gau, aes(x = Iteration, y = b)) +
geom_line() + theme_bw() + ylim(0, 7) +
theme(legend.position = "none") + ylab("b") +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 16))

ggsave("trace_latent_gau_b_cano.png", width=3.3, height=2, units='in')

trace_latent_gau = data.frame(
    b = c(res_b_bri),
    Iteration = c(1:num_samples)
)

ggplot(data = trace_latent_gau, aes(x = Iteration, y = b)) +
geom_line() + theme_bw() + ylim(0, 7) +
theme(legend.position = "none") + ylab("b") +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 16))

ggsave("trace_latent_gau_b_bri.png", width=3.3, height=2, units='in')

# trace of tau

trace_latent_gau = data.frame(
    tau = c(res_tau_cano),
    Iteration = c(1:num_samples)
)

ggplot(data = trace_latent_gau, aes(x = Iteration, y = tau)) +
geom_line() + theme_bw() + ylim(0, 3) +
theme(legend.position = "none") + ylab(bquote(tau)) +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 16))

ggsave("trace_latent_gau_tau_cano.png", width=3.3, height=2, units='in')

trace_latent_gau = data.frame(
    tau = c(res_tau_bri),
    Iteration = c(1:num_samples)
)

ggplot(data = trace_latent_gau, aes(x = Iteration, y = tau)) +
geom_line() + theme_bw() + ylim(0, 3) +
theme(legend.position = "none") + ylab(bquote(tau)) +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 16))

ggsave("trace_latent_gau_tau_bri.png", width=3.3, height=2, units='in')

### plot acfs

lag.max <- 40
thinned_idx <- seq(1, num_samples, by = 1)

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_b_cano[thinned_idx],
            lag.max = lag.max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag.max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
geom_bar(stat = "identity", width = 0.5) + theme_bw() +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16))

ggsave("acf_latent_gau_b_cano.png", width=3.3, height=2, units='in')

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_b_bri[thinned_idx],
            lag.max = lag.max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag.max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
geom_bar(stat = "identity", width = 0.5) + theme_bw() +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16))

ggsave("acf_latent_gau_b_bri.png", width=3.3, height=2, units='in')

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_tau_cano[thinned_idx],
            lag.max = lag.max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag.max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
geom_bar(stat = "identity", width = 0.5) + theme_bw() +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16))

ggsave("acf_latent_gau_tau_cano.png", width=3.3, height=2, units='in')

acf_latent_gau <- data.frame(
    ACF = c(
        as.numeric(acf(res_tau_bri[thinned_idx],
            lag.max = lag.max, plot = FALSE)[[1]])
    ),
    Lag = c(0:lag.max)
)

ggplot(data = acf_latent_gau, aes(x = Lag, y = ACF)) +
geom_bar(stat = "identity", width = 0.5) + theme_bw() +
theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16))

ggsave("acf_latent_gau_tau_bri.png", width=3.3, height=2, units='in')


### scatter plots of posterior samples

ggplot(data = data.frame(
    b = res_b_cano[-(1:burn_in)],
    tau = res_tau_cano[-(1:burn_in)]
    ), aes(x=tau, y=b)) +
geom_point(color = "black", size = 0.5) +
theme_bw() + xlim(0, 3) + ylim(0, 7) + xlab(bquote(tau)) + ylab(bquote(b)) +
theme(
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15)) +
theme(legend.position = "none")

ggsave("scatter_latent_gau_canonical.png", width=4, height=2.5, units='in')


ggplot(data = data.frame(
    b = res_b_bri[-(1:burn_in)],
    tau = res_tau_bri[-(1:burn_in)]
    ), aes(x=tau, y=b)) +
geom_point(color = "black", size = 0.5) +
theme_bw() + xlim(0, 3) + ylim(0, 7) + xlab(bquote(tau)) + ylab(bquote(b)) +
theme(
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15)) +
theme(legend.position = "none")

ggsave("scatter_latent_gau_bridge.png", width=4, height=2.5, units='in')

######################

# several number of samples; compare the variances

ns_all <- c(50, 200, 500, 1000)
curve_all <- c(1:9)[-c(2,4,9)]
res_b_bri <- c()
res_b_cano <- c()
res_tau_bri <- c()
res_tau_cano <- c()

for(curve in curve_all)
    for(ns in ns_all) {
        filename <- paste("res_latent_gau/b_trace_curve",
                curve, "ss", ns, ".txt", sep='')
        res_b_bri <- cbind(res_b_bri, unlist(read.table(filename)))
        filename <- paste("res_latent_gau/b_aug_trace_curve",
                curve, "ss", ns, ".txt", sep='')
        res_b_cano <- cbind(res_b_cano, unlist(read.table(filename)))
        filename <- paste("res_latent_gau/tau_trace_curve",
                curve, "ss", ns, ".txt", sep='')
        res_tau_bri <- cbind(res_tau_bri, unlist(read.table(filename)))
        filename <- paste("res_latent_gau/tau_aug_trace_curve",
                curve, "ss",  ns, ".txt", sep='')
        res_tau_cano <- cbind(res_tau_cano, unlist(read.table(filename)))
    }


thinned_idx <- seq(1, nrow(res_b_bri), by=10)
var_b <- diag(var(res_b_bri[thinned_idx, ]))
var_tau <- diag(var(res_tau_bri[thinned_idx, ]))
thinned_idx <- seq(1, nrow(res_b_cano), by=30)
var_b <- c(var_b, diag(var(res_b_cano[thinned_idx, ])))
var_tau <- c(var_tau, diag(var(res_tau_cano[thinned_idx, ])))


label <- c("Latent quadratic exponential model", "Latent normal model")

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
