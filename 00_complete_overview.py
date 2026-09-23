"""
COMPLETE LINEAR MODELS OVERVIEW
================================
Based on scikit-learn documentation

This tutorial covers ALL linear models from:
https://scikit-learn.org/stable/modules/linear_model.html

TOPICS COVERED:
---------------
1. Ordinary Least Squares (OLS)
2. Ridge Regression & Classification
3. Lasso Regression
4. Multi-task Lasso
5. Elastic-Net
6. Multi-task Elastic-Net
7. Least Angle Regression (LARS)
8. LARS Lasso
9. Orthogonal Matching Pursuit (OMP)
10. Bayesian Regression (Ridge & ARD)
11. Logistic Regression
12. Generalized Linear Models (GLM)
13. Stochastic Gradient Descent (SGD)
14. Perceptron
15. Passive Aggressive Algorithms
16. Robust Regression (RANSAC, Theil-Sen, Huber)
17. Quantile Regression
18. Polynomial Regression

Each topic includes:
- Simple explanation
- Mathematical intuition
- Use cases
- Code examples
- Visualizations
"""

print("="*70)
print("SCIKIT-LEARN LINEAR MODELS - COMPLETE GUIDE")
print("="*70)
print("\nThis comprehensive tutorial covers 18 different linear model types.")
print("Run the individual tutorial files to learn each model in detail.\n")

# Overview of all models
models_overview = {
    "REGRESSION MODELS (Predict Numbers)": [
        "LinearRegression - Basic OLS",
        "Ridge - L2 regularization",
        "Lasso - L1 regularization (sparse)",
        "ElasticNet - L1 + L2 combined",
        "Lars - High-dimensional data",
        "LassoLars - Lasso with LARS algorithm",
        "OrthogonalMatchingPursuit - Fixed sparsity",
        "BayesianRidge - Probabilistic Ridge",
        "ARDRegression - Automatic Relevance Determination",
        "TweedieRegressor - Poisson, Gamma, etc.",
        "PoissonRegressor - Count data",
        "GammaRegressor - Positive skewed data",
        "SGDRegressor - Large-scale learning",
        "RANSACRegressor - Robust to outliers",
        "TheilSenRegressor - Robust median-based",
        "HuberRegressor - Robust to outliers",
        "QuantileRegressor - Predict quantiles",
    ],
    "CLASSIFICATION MODELS (Predict Categories)": [
        "LogisticRegression - Binary/multiclass",
        "RidgeClassifier - Fast for many classes",
        "SGDClassifier - Large-scale learning",
        "Perceptron - Simple online learning",
        "PassiveAggressiveClassifier - Online learning",
    ],
    "MULTI-TASK MODELS (Multiple Outputs)": [
        "MultiTaskLasso - Shared feature selection",
        "MultiTaskElasticNet - Shared features with L1+L2",
    ]
}

for category, models in models_overview.items():
    print(f"\n{category}")
    print("-" * 70)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

print("\n" + "="*70)
print("TUTORIAL FILES")
print("="*70)
print("""
Run these files in order to learn each model:

01_ols_linear_regression.py      - Ordinary Least Squares
02_ridge_regression.py            - Ridge (L2 regularization)
03_lasso_regression.py            - Lasso (L1 regularization)
04_elasticnet.py                  - Elastic-Net (L1 + L2)
05_lars_algorithms.py             - LARS and LARS Lasso
06_omp.py                         - Orthogonal Matching Pursuit
07_bayesian_regression.py         - Bayesian Ridge & ARD
08_logistic_regression.py         - Classification
09_glm_models.py                  - Generalized Linear Models
10_sgd_models.py                  - SGD, Perceptron, Passive Aggressive
11_robust_regression.py           - RANSAC, Theil-Sen, Huber
12_quantile_regression.py         - Quantile Regression
13_polynomial_regression.py       - Polynomial features
14_multitask_models.py            - Multi-task Lasso & ElasticNet
15_model_selection.py             - Cross-validation & comparison
""")

print("\n" + "="*70)
print("QUICK DECISION GUIDE")
print("="*70)
print("""
WHAT DO YOU WANT TO DO?

Predict a NUMBER:
  → Start with LinearRegression
  → Too many features? Use Ridge or Lasso
  → Features correlated? Use Ridge
  → Want feature selection? Use Lasso
  → Both? Use ElasticNet

Predict a CATEGORY:
  → Use LogisticRegression
  → Many classes? Try RidgeClassifier
  → Huge dataset? Use SGDClassifier

Handle OUTLIERS:
  → Use RANSACRegressor, TheilSenRegressor, or HuberRegressor

Predict COUNT DATA:
  → Use PoissonRegressor

Predict QUANTILES (not just mean):
  → Use QuantileRegressor

LARGE DATASET (millions of samples):
  → Use SGDRegressor or SGDClassifier

HIGH-DIMENSIONAL (more features than samples):
  → Use Lars, LassoLars, or OrthogonalMatchingPursuit
""")
