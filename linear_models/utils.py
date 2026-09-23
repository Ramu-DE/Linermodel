"""Array helpers used by the educational implementations."""
import numpy as np
from .base import check_array


def add_intercept(X):
    X = check_array(X)
    return np.column_stack((np.ones(X.shape[0]), X))


def standardize(X, mean=None, scale=None):
    X = check_array(X)
    mean = X.mean(axis=0) if mean is None else np.asarray(mean, dtype=float)
    scale = X.std(axis=0) if scale is None else np.asarray(scale, dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    return (X - mean) / scale, mean, scale
