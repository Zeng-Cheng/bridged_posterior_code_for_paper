# package and setting for plot
require("kernlab")
library(reshape2)
library(ggplot2)
jet_colors <- colorRampPalette(c("#00007F", "blue", "#007FFF",
                "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

# load data and results
# The data are not included in the Github repository
# if one needs to run the data application (either runmcmc or plot_hfcg)
# first download the files in the following link to data folder
url = 'https://www.dropbox.com/scl/fo/pwxznmap3cm97iycuobw8/h?rlkey=of0spg09mgh3ly9a38svnktce&e=1&dl=0'

# 'list_A' contains correlation matrices for each subject
load("data/graph_fmri_all.Rda") 

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

load("HFCG_gibbs_res.Rda")
num_iters <- 10000
burn_in <- 2000

#########################################
#------ analysis of data and plot ------#

# round to range to get the lambda that is closest to the posterior mean
round_to_range <- function(x, possible_values) {
    return(which.min(abs(x - possible_values)))
}
mean_post_samples <- rowMeans(sapply((burn_in + 1):num_iters, function(i) {
    sapply(1:S, function(s) lambdas_all[[s]][post_samples[[1]][[i]][s]])
    }))
post_mean_int <- sapply(1:S, function(s) {
    return(round_to_range(
        mean_post_samples[s],
        possible_values=lambdas_all[[s]]
        ))
})
print(post_mean_int) # position of posterior mean of lambda for each s

# get pairwise distances between networks with posterior mean
pair_dist <- matrix(0, length(1:S), length(1:S))
for(s in 1:S){
    for(k in 1:S){
        if(s != k){
            pair_dist[s, k] <- geodist[[s]][[k]][post_mean_int[s], post_mean_int[k]]
        }
    }
}

# spectral clustering on pairwise distances
s <- apply(pair_dist, 1, function(x) quantile(x, 0.01))
cls_res <- table(specc(
    as.kernelMatrix(exp(-t(pair_dist ^ 2 / s) / s)), centers=3
    ), labels)
cls_res
# apply(cls_res, 2, max) / colSums(cls_res)


####################################
#---- pairwise distances plots ----#

cor_matrix <- melt(pair_dist / max(pair_dist))
ggplot(cor_matrix, aes(y = Var1, x = Var2)) + geom_raster(aes(fill = value)) +
scale_fill_gradientn(
    colours = jet_colors(100), breaks = seq(0, 1, by = 0.1), limits = c(0, 1)
    ) +
guides(fill = guide_colourbar(barwidth = 0.5, barheight = 16, title = NULL)) +
labs(x = NULL, y = NULL) + scale_y_reverse(breaks = seq(0, S, length.out = 2)) +
scale_x_continuous(breaks = seq(0, S, length.out = 2)) +
theme_void() + theme(
    legend.justification = c(0, 0.6),
    legend.text = element_text(size = 11)) +
theme(
    axis.text.x = element_text(size = 11, vjust = 5),
    axis.text.y = element_text(size = 11, hjust = 1.6))
ggsave("pair_dist.png", width = 4.4, height = 3.64, unit = "in")

### boxplots

pair_dist_norm <- pair_dist
dist_healthy <- c(pair_dist_norm[1:64, 1:64])
dist_disease <- c(pair_dist_norm[65:S, 65:S])
ks.test(dist_healthy, dist_disease)
dist_bw <- c(pair_dist_norm[1:64, 65:S], pair_dist_norm[65:S, 1:64])
ks.test(dist_bw, dist_disease)

df_boxplot <- data.frame(
    Distance = c(dist_healthy, dist_disease, dist_bw),
    Source = c(
        rep("Healthy", length(dist_healthy)),
        rep("Diseased", length(dist_disease)),
        rep("Between Groups", length(dist_bw)))
)

ggplot(data = df_boxplot, aes(Distance, Source, group = Source)) +
geom_boxplot(outlier.shape = NA, varwidth = TRUE) + theme_bw() +
scale_y_discrete(labels = c("Between Groups" = "Between\nGroups")) +
xlim(10, 30)

ggsave("boxplots_dist.png", width=4, height=2.5, units='in')

### pairwise distances between original matrices

p <- dim(list_A[[1]])[1]
pair_dist_ori = matrix(0, length(1:S), length(1:S))
for(s in 1:S) {
    LAs <- laps[[s]]
    for(k in 1:S) {
        LAk <- laps[[k]]
        evs = Re(svd(solve(LAs + diag(1, p) * 1e-3, LAk + diag(1, p) * 1e-3))$d)
        pair_dist_ori[s, k] = sqrt(sum((log(evs)) ^ 2))
    }
}

### boxplots

pair_dist_norm <- pair_dist_ori
dist_healthy <- c(pair_dist_norm[1:64, 1:64])
dist_disease <- c(pair_dist_norm[65:S, 65:S])
ks.test(dist_healthy, dist_disease)
dist_bw <- c(pair_dist_norm[1:64, 65:S], pair_dist_norm[65:S, 1:64])
ks.test(dist_bw, dist_disease)

df_boxplot <- data.frame(
    Distance = c(dist_healthy, dist_disease, dist_bw),
    Source = c(
        rep("Healthy", length(dist_healthy)),
        rep("Diseased", length(dist_disease)),
        rep("Between Groups", length(dist_bw)))
)

ggplot(data = df_boxplot, aes(Distance, Source, group = Source)) +
geom_boxplot(outlier.shape = NA, varwidth = TRUE) + theme_bw() +
scale_y_discrete(labels = c("Between Groups" = "Between\nGroups")) +
xlim(1, 11)

ggsave("boxplots_dist_ori.png", width=4, height=2.5, units='in')

##################################################
##################################################

# for all subjects, find the number of communities with mean posterior lambda

num_cum_healthy <- sapply(1:64, function(s) {
    curr_fmri <- Z_all[[s]][[post_mean_int[s]]]
    return(sum(svd(curr_fmri)$d < 1E-5))
})

num_cum_diseased <- sapply(65:S, function(s) {
    curr_fmri <- Z_all[[s]][[post_mean_int[s]]]
    return(sum(svd(curr_fmri)$d < 1E-5))
})

mean(num_cum_healthy)
mean(num_cum_diseased)

ggplot(data = data.frame(x = num_cum_healthy), aes(x = x)) +
geom_bar() + theme_bw() + xlab("Number of Communities") + ylab('Count') +
geom_vline(xintercept = mean(num_cum_healthy), color = '#ef609f') +
xlim(0, 26)

ggsave("HFCG_bar_healthy.png", width = 4, height = 2.5, unit = "in")

ggplot(data = data.frame(x = num_cum_disease), aes(x = x)) +
geom_bar() + theme_bw() + xlab("Number of Communities") + ylab('Count') +
geom_vline(xintercept = mean(num_cum_diseased), color = '#ef609f') +
xlim(0, 26)

ggsave("HFCG_bar_diseased.png", width = 4, height = 2.5, unit = "in")