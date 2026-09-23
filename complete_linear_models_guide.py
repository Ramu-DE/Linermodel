"""
COMPLETE LINEAR MODELS GUIDE
=============================
Comprehensive tutorial covering ALL linear models from scikit-learn
Based on: https://scikit-learn.org/stable/modules/linear_model.html

Topics Covered:
1. Ordinary Least Squares (OLS)
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Multi-task Lasso & Elastic Net
6. Least Angle Regression (LARS)
7. Orthogonal Matching Pursuit (OMP)
8. Bayesian Regression (Ridge & ARD)
9. Logistic Regression
10. Generalized Linear Models (GLM)
11. Stochastic Gradient Descent (SGD)
12. Perceptron
13. Passive Aggressive Algorithms
14. Robust Regression (RANSAC, Theil-Sen, Huber)
15. Quantile Regression
16. Polynomial Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("COMPLETE LINEAR MODELS TUTORIAL")
print("="*80)
print("\nThis comprehensive guide covers all linear models in scikit-learn")
print("Run each section to see visualizations and understand concepts\n")

# Generate sample data for demonstrations
np.random.seed(42)
n_samples = 100
X_train = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
y_train = 2 * X_train.ravel() + np.random.randn(n_samples) * 2

# Add some outliers for robust regression demos
X_outliers = np.random.rand(10, 1) * 10
y_outliers = 3 * X_outliers.ravel() + 50
X_with_outliers = np.vstack([X_train, X_outliers])
y_with_outliers = np.hstack([y_train, y_outliers.ravel()])

# ============================================================================
# SECTION 1: ORDINARY LEAST SQUARES (OLS)
# ============================================================================