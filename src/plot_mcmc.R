library(ggplot2)

# the function for plotting the trace of the MCMC samples
plot_trace <- function(
    res, ylims=c(-3,3), yname, filename, width=3.3, height=2) {
    num_samples <- length(res)
    df <- data.frame(b = c(res), Iteration = c(1:num_samples))
    
    ggplot(data = df, aes(x = Iteration, y = b)) +
        geom_line() + theme_bw() + ylim(ylims) +
        theme(legend.position = "none") + ylab(yname) +
        theme(
            plot.margin = margin(t = 8, r = 20, b = 8, l = 8),
            axis.title.x = element_text(size = 16),
            axis.title.y = element_text(size = 16),
            axis.text.x = element_text(size = 12),
            axis.text.y = element_text(size = 16)
        )

    ggsave(filename, width = width, height = height, units = 'in')
}


# the function for plotting the acf of the MCMC samples
plot_acf <- function(res, filename, lag_max=40, width=3.3, height=2) {

    acf_data <- data.frame(
        ACF = c(
            as.numeric(acf(res, lag.max = lag_max, plot = FALSE)[[1]])
        ),
        Lag = c(0:lag_max)
    )

    ggplot(data = acf_data, aes(x = Lag, y = ACF)) +
        geom_bar(stat = "identity", width = 0.5) + theme_bw() +
        theme(
            axis.title.x = element_text(size = 16),
            axis.title.y = element_text(size = 16),
            axis.text.x = element_text(size = 16),
            axis.text.y = element_text(size = 16))

    ggsave(filename, width=width, height=height, units='in')
}