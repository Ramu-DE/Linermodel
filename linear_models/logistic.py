"""Binary logistic regression with batch gradient descent and L2 regularization."""
import numpy as np
from .base import BaseClassifier, check_array


class LogisticRegression(BaseClassifier):
    def __init__(self, learning_rate=0.1, max_iter=5000, C=1.0, tol=1e-6, fit_intercept=True):
        if C <= 0: raise ValueError("C must be positive")
        self.learning_rate, self.max_iter, self.C, self.tol, self.fit_intercept = learning_rate, max_iter, C, tol, fit_intercept
    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        X = check_array(X); raw = np.asarray(y).reshape(-1)
        if len(raw) != len(X): raise ValueError("y must have one value per row")
        self.classes_ = np.unique(raw)
        if len(self.classes_) != 2: raise ValueError("only binary classification is supported")
        target = (raw == self.classes_[1]).astype(float); self.coef_ = np.zeros(X.shape[1]); self.intercept_ = 0.0
        for iteration in range(self.max_iter):
            prob = self._sigmoid(X @ self.coef_ + self.intercept_)
            error = prob - target; gradient = X.T @ error / len(X) + self.coef_ / self.C
            intercept_gradient = error.mean() if self.fit_intercept else 0.0
            self.coef_ -= self.learning_rate * gradient; self.intercept_ -= self.learning_rate * intercept_gradient
            if np.linalg.norm(np.r_[gradient, intercept_gradient]) < self.tol: break
        self.n_iter_ = iteration + 1; return self
    def predict_proba(self, X):
        self._check_fitted(); p = self._sigmoid(check_array(X) @ self.coef_ + self.intercept_)
        return np.column_stack((1-p, p))
    def predict(self, X): return self.classes_[(self.predict_proba(X)[:,1] >= .5).astype(int)]
