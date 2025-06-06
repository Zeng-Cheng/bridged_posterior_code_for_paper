# %%
import torch
import numpy as np
import hamiltorch
from tqdm.notebook import trange
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from src.svm_slice_sampling import svm_log_prob, svm_optim_step, svm_sample_missing
from src.utils_slice import acc_rate

# %%
X_test = np.loadtxt('data/heart_failure_clinical_x_test.txt')
X_known = np.loadtxt('data/heart_failure_clinical_x_known.txt')
y_test = np.loadtxt('data/heart_failure_clinical_y_test.txt')
y_known = np.loadtxt('data/heart_failure_clinical_y_known.txt')

# %%
# accuracy without using missing data
clf = svm.LinearSVC(loss='hinge', penalty='l2', dual='auto', max_iter=int(1E9))   
clf.fit(X_known, y_known)
# predict labels for testing data
predicted_labels = clf.predict(X_test)
print("LinearSVC Accuracy:", accuracy_score(y_test, predicted_labels))

# %%
# Logistic Regression using only labeled data
p = X_known.shape[1]
clf = LogisticRegression(random_state=0).fit(X_known, y_known)
y_pred = clf.predict_proba(X_test)[:, 1]
# predict labels for testing data
print('Logistic regression accuracy:', accuracy_score(y_test, 2 * (y_pred > 0.5) - 1))
print('Logistic regression AUC', roc_auc_score(y_test, y_pred))
np.savetxt('res_svm/pred_prob_logis.txt', np.array([y_test, y_pred]).T)

# %%
# the bridged posterior -- Bayesian maximum margin classifier

# initial values
lam = torch.tensor([1])
alpha = torch.tensor([1])

num_samples = 1500
burn_in = 400
batch_size = 35

num_missing = X_test.shape[0]
beta_trace, b_trace, lam_trace, y_trace = [], [], [], []

# initialize the missing values of y using Bernoulli(0.5)
y_init = 2 * np.random.binomial(1, 0.5, num_missing) - 1

y_all = torch.cat([torch.tensor(y_init), torch.tensor(y_known)]).float() # the first m values are unlabeled
X_all = torch.cat([torch.tensor(X_test), torch.tensor(X_known)]).float()
beta, b = svm_optim_step(X_all, y_all, lam)
logprob = svm_log_prob(beta, b, X_all, y_all, lam)
lam_lam = 1
accept_lam = np.zeros(num_samples)
for t in trange(num_samples):
    
    y_all, beta, b, logprob = svm_sample_missing(
        lam, X_all, y_all, batch_size, num_missing, alpha_mcmc = 0.9
    )
    lam_prop = lam + torch.randn(1) * lam_lam
    if (lam_prop > 0): 
        beta_prop, b_prop = svm_optim_step(X_all, y_all, lam_prop)
        logprob_prop = svm_log_prob(beta_prop, b_prop, X_all, y_all, lam_prop)
        if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
            lam = lam_prop
            beta = beta_prop
            b = b_prop
            logprob = logprob_prop
            accept_lam[t] = 1

    if t % 400 == 0:
        print('step: {:d}, accept_rate of lam: {:.3f}, lam lam: {:.3f}'.format(
            t, acc_rate(accept_lam, t + 1), lam_lam))

    beta_trace.append(beta)
    b_trace.append(b)
    lam_trace.append(lam)
    y_trace.append(y_all.clone().detach())

beta_trace, b_trace, lam_trace, y_trace = [
    np.stack(list) for list in [beta_trace, b_trace, lam_trace, y_trace]]

y_test_pred = (y_trace[:, :X_test.shape[0]] + 1) / 2
test_pred_prob = np.mean(y_test_pred[burn_in:], axis = 0)

print("Bridged AUC:", roc_auc_score(y_test, test_pred_prob))
print("Bridged accuracy", accuracy_score(y_test, 2 * (test_pred_prob > 0.5) - 1))

# %%
np.savetxt('res_svm/pred_prob_bridged.txt', np.array([y_test, test_pred_prob]).T)
np.savetxt('res_svm/beta_trace_heart.txt', beta_trace.reshape((-1, X_known.shape[1])))

# %%
# save the proportion of y_i = 1, and the distance to the decision boundary
dist_to_dec_bound, prob_y1 = [], [] 
w_pred = np.mean(beta_trace[burn_in:, :], 0)
b_pred = np.mean(b_trace[burn_in:])

num_missing = X_test.shape[0]

dist_to_dec_bound.append(X_test @ w_pred + b_pred)
prob_y1.append((y_trace[burn_in:, :num_missing] == 1).sum(0) / (num_samples - burn_in))

dist_to_dec_bound, prob_y1 = np.stack(dist_to_dec_bound), np.stack(prob_y1)
np.savetxt('res_svm/dist_to_dec_bound_heart.txt', dist_to_dec_bound)
np.savetxt('res_svm/prob_y1_heart.txt', prob_y1)

# %%
# Gibbs posterior using HMC using only labeled data
# Setting the hyper parameters
alpha_lam, beta_lam = 3, 2
p = X_known.shape[1]
X_known, y_known = torch.tensor(X_known).float(), torch.tensor(y_known).float()
X_test = torch.tensor(X_test).float()
alpha = torch.tensor([3])

def log_prob_gibbs(params):
    beta_param = params[:p]
    b_param = params[p]
    log_lam_param = params[p+1]
    lam_param = torch.exp(log_lam_param)
    log_lik = svm_log_prob(beta_param, b_param, X_known, y_known, lam_param, alpha = alpha)
    
    y_ones = torch.ones(X_known.shape[0])
    y_minus_ones = -torch.ones(X_known.shape[0])
    res_ones = -alpha * torch.relu(1 - y_ones * (X_known @ beta_param + b_param))
    res_minus_ones = -alpha * torch.relu(1 - y_minus_ones * (X_known @ beta_param + b_param))
    res_norm_const = torch.logaddexp(res_ones, res_minus_ones).sum()       
    return log_lik - res_norm_const

beta0 = torch.randn(p) 
b0 = torch.randn(1)
log_lam = torch.log(torch.tensor([0.07]))

params_init = torch.zeros(p+2)
params_init[:p], params_init[p], params_init[p+1] = beta0, b0, log_lam
params_init.requires_grad = True

num_samples = 2000
L = 5
burn = 200
step_size = 0.01

params_hmc_gibbs = hamiltorch.sample(
    log_prob_func=log_prob_gibbs, params_init=params_init, num_samples=num_samples,
    step_size=step_size, num_steps_per_sample=L, desired_accept_rate=0.9,
    sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn)

params_gibbs = torch.stack(params_hmc_gibbs)
post_beta_gibbs, post_b_gibbs, post_lambda_gibbs = params_gibbs[:,:p], params_gibbs[:,p], params_gibbs[:,p+1]

y_test_pred = []
for t in range(num_samples - burn):
    y_pred = X_test @ post_beta_gibbs[t, ] + post_b_gibbs[t]
    p_i = 1 / (1 + torch.exp(-torch.relu(1 + y_pred) + torch.relu(1 - y_pred)))
    y_cand = torch.bernoulli(p_i)
    # y_pred = np.where(dec_bound > 0, 1, -1)
    y_test_pred.append(y_cand)

y_test_pred = np.stack(y_test_pred)
test_pred_prob = np.mean(y_test_pred, axis = 0)

print('Gibbs AUC', roc_auc_score(y_test, test_pred_prob))
print('Gibbs accuracy', accuracy_score(y_test, 2 * (test_pred_prob > 0.5) - 1))

# %%
np.savetxt('res_svm/pred_prob_gibbs.txt', np.array([y_test, test_pred_prob]).T)


