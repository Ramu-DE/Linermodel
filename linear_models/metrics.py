"""Small dependency-free regression and binary-classification metrics."""
import numpy as np


def _pair(y_true, y_pred):
    a, b = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if len(a) == 0 or a.shape != b.shape:
        raise ValueError("targets must be non-empty arrays with equal shape")
    return a, b


def mean_squared_error(y_true, y_pred):
    a, b = _pair(y_true, y_pred); return float(np.mean((a - b) ** 2))
def root_mean_squared_error(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mean_absolute_error(y_true, y_pred):
    a, b = _pair(y_true, y_pred); return float(np.mean(np.abs(a - b)))
def r2_score(y_true, y_pred):
    a, b = _pair(y_true, y_pred); total = np.sum((a - a.mean()) ** 2)
    return 1.0 if total == 0 and np.allclose(a, b) else (0.0 if total == 0 else float(1 - np.sum((a-b)**2)/total))
def accuracy_score(y_true, y_pred):
    a, b = _pair(y_true, y_pred); return float(np.mean(a == b))
def confusion_matrix(y_true, y_pred):
    a, b = _pair(y_true, y_pred)
    labels = np.unique(np.r_[a, b])
    return np.array([[np.sum((a == i) & (b == j)) for j in labels] for i in labels])
def precision_score(y_true, y_pred):
    a, b = _pair(y_true, y_pred); tp=np.sum((a==1)&(b==1)); fp=np.sum((a!=1)&(b==1)); return float(tp/(tp+fp)) if tp+fp else 0.0
def recall_score(y_true, y_pred):
    a, b = _pair(y_true, y_pred); tp=np.sum((a==1)&(b==1)); fn=np.sum((a==1)&(b!=1)); return float(tp/(tp+fn)) if tp+fn else 0.0
def f1_score(y_true, y_pred):
    p, r = precision_score(y_true,y_pred), recall_score(y_true,y_pred); return 0.0 if p+r == 0 else 2*p*r/(p+r)
