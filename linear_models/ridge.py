"""Ridge regression with internal centering and scale-safe L2 penalty."""
import numpy as np
from .base import BaseRegressor, check_array
from .utils import standardize


class Ridge(BaseRegressor):
    def __init__(self, alpha=1.0, fit_intercept=True):
        if alpha < 0: raise ValueError("alpha must be non-negative")
        self.alpha, self.fit_intercept = float(alpha), fit_intercept
    def fit(self, X, y):
        X, y = check_array(X, y)
        self._x_mean = X.mean(0) if self.fit_intercept else np.zeros(X.shape[1])
        self._y_mean = float(y.mean()) if self.fit_intercept else 0.0
        Xs, _, self._x_scale = standardize(X - self._x_mean)
        yc = y - self._y_mean
        system = Xs.T @ Xs + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(system, Xs.T @ yc) / self._x_scale
        self.intercept_ = self._y_mean - float(self._x_mean @ self.coef_)
        return self
