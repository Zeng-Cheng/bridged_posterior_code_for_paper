library(ggplot2)

###########################################
## plot ROC curves

library(pROC) # install.packages("pROC")
data_logis <- read.table('output/res_svm/pred_prob_logis.txt')
roc_logis <- roc(data_logis[, 1], data_logis[, 2])
data_bridged <- read.table('output/res_svm/pred_prob_bridged.txt')
roc_bridged <- roc(data_bridged[, 1], data_bridged[, 2])
data_gibbs <- read.table('output/res_svm/pred_prob_gibbs.txt')
roc_gibbs <- roc(data_gibbs[, 1], data_gibbs[, 2])

ggplot(
    data = data.frame(
        fpr = c(1 - roc_logis$spe, 1 - roc_gibbs$spe, 1 - roc_bridged$spe),
        tpr = c(roc_logis$sen, roc_gibbs$sen, roc_bridged$sen),
        Model = rep(
            c('Logistic regression', 'Gibbs posterior using hinge loss',
                'Bayesian maximum margin classifier'),
            times = c(length(roc_logis$spe), length(roc_gibbs$spe),
                length(roc_bridged$spe)))),
    aes(x=fpr, y=tpr, color=Model)) +
    geom_path(linewidth = 0.5) + theme_bw() +
    xlab('False positive rate') + ylab('True positive rate') +
    scale_color_manual(values = c("#ef609f", "#429dfb", '#7bd34b')) +
    theme(
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size = 11),
        legend.text = element_text(size = 9),
        legend.position = 'inside',
        legend.position.inside = c(0.7, 0.25))

ggsave("roc.pdf", width=5.3, height=3.3, units='in')


#################################################################
### plot dist_to_dec_bound vs prob_y1 for missing value ###
#################################################################

dist_to_dec_bound <- unlist(read.table(
    "output/res_svm/dist_to_dec_bound_heart.txt"))
prob_y1 <- unlist(read.table("output/res_svm/prob_y1_heart.txt"))

ggplot(
    data = data.frame(x = dist_to_dec_bound, y = prob_y1),
    aes(x = x, y = y)) +
    geom_point(color = "#1f77b4", size = 1) + theme_bw() + # xlim(0, 5) +
    xlab('Distance') + ylab(bquote(P(y[j] == 1))) +
    theme(legend.position = "none") +
    theme(
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

ggsave("missing_prob_heart.pdf", width=4.5, height=2.5, units='in')



####################################
#### plot angle distribution ###
####################################

angles_beta <- c()
res_w <- as.matrix(read.table("output/res_svm/beta_trace_heart.txt"))
num_iters <- nrow(res_w)
beta_svc <- colMeans(res_w)

for (i in 401:num_iters) {
    cos_angle <- sum(beta_svc * res_w[i, ]) / sqrt(sum(beta_svc ^ 2))
    cos_angle <- cos_angle / sqrt(sum(res_w[i, ] ^ 2))
    angles_beta <- c(angles_beta, acos(cos_angle))
}

ggplot(data = data.frame(Angle = angles_beta), aes(Angle)) +
    geom_density(color = 'black') + theme_bw() + xlim(0, 0.6) +
    theme(
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12)) +
    ylab('Density')

ggsave("density_angle_heart.pdf", width=4.5, height=2.5, units='in')
