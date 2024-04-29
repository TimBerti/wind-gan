import torch

def auto_cov(X, lag=1):
    X = X - torch.mean(X, axis=3).unsqueeze(3)
    if lag == 0:
        return (X[:, 0, 0, :] * X[:, 0, 0,:]).sum(axis=1)
    else:
        return (X[:, 0, 0, lag:] * X[:, 0, 0,:-lag]).sum(axis=1)

def auto_corr(X, lag=1):
    return auto_cov(X, lag) / auto_cov(X, 0)

def complete_auto_corr(X, max_lag=100):
    return torch.cat([auto_corr(X, lag).reshape(1, -1) for lag in range(max_lag)], axis=0)