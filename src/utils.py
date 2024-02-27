import numpy as np

def acc_rate(accept, i, lags = 200):
    if i < lags:
        return np.sum(accept[:i]) / i
    else:
        return np.sum(accept[(i - lags):i]) / lags