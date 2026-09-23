"""Lasso regression using coordinate descent and soft thresholding."""
import numpy as np
from .base import BaseRegressor, check_array
from .utils import standardize


class Lasso(BaseRegressor):
    def __init__(self, alpha=1.0, max_iter=5000, tol=1e-6, fit_intercept=True):
        if alpha < 0: raise ValueError("alpha must be non-negative")
        self.alpha, self.max_iter, self.tol, self.fit_intercept = float(alpha), max_iter, tol, fit_intercept
    @staticmethod
    def soft_threshold(value, threshold): return np.sign(value) * max(abs(value) - threshold, 0.0)
    def fit(self, X, y):
        X, y = check_array(X, y); n, p = X.shape
        self._x_mean = X.mean(0) if self.fit_intercept else np.zeros(p)
        self._y_mean = float(y.mean()) if self.fit_intercept else 0.0
        Xs, _, self._x_scale = standardize(X-self._x_mean); yc = y-self._y_mean
        weights = np.zeros(p); residual = yc.copy()
        for iteration in range(self.max_iter):
            old = weights.copy()
            for j in range(p):
                residual += Xs[:, j] * weights[j]
                weights[j] = self.soft_threshold(float(Xs[:, j] @ residual / n), self.alpha)
                residual -= Xs[:, j] * weights[j]
            if np.max(np.abs(weights-old)) < self.tol: break
        self.n_iter_ = iteration + 1; self.coef_ = weights / self._x_scale
        self.intercept_ = self._y_mean - float(self._x_mean @ self.coef_)
        return self
