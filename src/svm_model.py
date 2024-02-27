import torch
from sklearn import svm


def svm_optim_step(X, y, lam):
    if lam < 0:
        raise ValueError("lambda must be greater than 0")
    
    clf = svm.LinearSVC(loss='hinge', penalty='l2', dual='auto',
                    C=1/(2*lam.item()), max_iter=int(1E7))   
    clf.fit(X, y)
    beta = torch.from_numpy(clf.coef_)[0].float() 
    b = torch.from_numpy(clf.intercept_)[0]
    return beta, b


def svm_log_prob(beta, b, X, y, lam, alpha = 1, alpha_lam = 3, beta_lam = 2):
    """
    Args:
        beta: previous w, the coefficient of X\beta; vector of 1*p
        b: scalar, the intercept
    """
    alpha = torch.tensor([alpha])
    if lam < 0:
        raise ValueError("lambda must be greater than 0")
    
    if alpha < 0:
        raise ValueError("alpha must be greater than 0")

    log_lam_prior = (alpha_lam - 1) * torch.log(lam) - beta_lam * lam
    log_alpha_prior = (alpha_lam - 1) * torch.log(alpha) - beta_lam * alpha
    res_relu = (torch.relu(1 - y * (X @ beta + b))).sum()
    loglik = -res_relu - lam * (torch.linalg.norm(beta)) ** 2 # hinge loss

    return log_alpha_prior + log_lam_prior + alpha * loglik




def svm_sample_missing(lam, X, y, batch_size, num_missing, alpha_mcmc = 1,
                       alpha_lam = 3, beta_lam = 2, alpha = 1):
    '''
        X = torch.cat([X_unknown, X_known])
        y = torch.cat([y_t, y_known]) # y_t from the last iteration
    '''
    alpha = torch.tensor([alpha])
    num_batches = int((num_missing + batch_size - 1) / batch_size)
    beta, b = svm_optim_step(X, y, lam)
    logprob = svm_log_prob(beta, b, X, y, lam, alpha, alpha_lam, beta_lam)
    alpha_mcmc = torch.tensor([alpha_mcmc])

    # iterate through the num_batches batches
    for j in range(num_batches):
        idx = range(j * batch_size, min(num_missing, (j + 1) * batch_size))

        # Propose a candidate
        x_i_current = X[idx, :]
        y_pred = x_i_current @ beta + b
        p_i = 1 / (1 + torch.exp(-alpha_mcmc * torch.relu(1 + y_pred) + alpha_mcmc * torch.relu(1 - y_pred)))
        y_cand = 2 * torch.bernoulli(p_i) - 1

        # Update y with the candidate values
        y_all_cand = y.clone().detach()
        y_all_cand[idx] = y_cand

        # Optimize w and b
        beta_opt, b_opt = svm_optim_step(X, y_all_cand, lam)

        # Calculate g()
        y_pred_opt = x_i_current @ beta_opt + b_opt
        g_y_t = 1 / (1 + torch.exp(
            -alpha_mcmc * torch.relu(1 + y[idx] * y_pred_opt) + alpha_mcmc * torch.relu(1 - y[idx] * y_pred_opt)
            ))
        g_y_star = 1 / (1 + torch.exp(
            -alpha_mcmc * torch.relu(1 + y_cand * y_pred) + alpha_mcmc * torch.relu(1 - y_cand * y_pred)
            ))

        # Calculate likelihoods
        lik_star = svm_log_prob(beta_opt, b_opt, X, y_all_cand,
                                lam, alpha, alpha_lam, beta_lam)

        # Calculate the acceptance ratio
        r1 = torch.log(g_y_t).sum() - torch.log(g_y_star).sum()
        r2 = lik_star - logprob

        # Decide whether to accept the candidate state or stay in the current state
        if torch.log(torch.rand(1)) < r1 + r2:
            y[idx] = y_cand
            beta, b = beta_opt, b_opt
            logprob = lik_star

    return y, beta, b, logprob