"""Ordinary least squares via a numerically stable least-squares solve."""
import numpy as np
from .base import BaseRegressor, check_array
from .utils import add_intercept


class LinearRegression(BaseRegressor):
    def __init__(self, fit_intercept=True): self.fit_intercept = fit_intercept
    def fit(self, X, y):
        X, y = check_array(X, y)
        design = add_intercept(X) if self.fit_intercept else X
        weights = np.linalg.lstsq(design, y, rcond=None)[0]
        self.intercept_ = float(weights[0]) if self.fit_intercept else 0.0
        self.coef_ = weights[1:] if self.fit_intercept else weights
        return self
