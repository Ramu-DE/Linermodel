"""Shared validation and sklearn-style base classes."""
import numpy as np


def check_array(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 1D or 2D numeric array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if y is None:
        return X
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) != len(X) or not np.isfinite(y).all():
        raise ValueError("y must be finite and have one value per row of X")
    return X, y


class BaseRegressor:
    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict")

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has a different number of features than training data")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))


class BaseClassifier:
    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict")

    def score(self, X, y):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
