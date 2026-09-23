"""
ORDINARY LEAST SQUARES (OLS) - Linear Regression
=================================================

WHAT IS IT?
-----------
The most basic linear regression. Finds coefficients that minimize
the sum of squared differences between predictions and actual values.

MATHEMATICAL FORMULA:
--------------------
Minimizes: ||Xw - y||²

Where:
- X = input features
- w = coefficients (weights)
- y = target values

INTUITION:
----------
Imagine fitting a line through points on a graph. OLS finds the line
that minimizes the total squared distance from all points to the line.

Think of it as: "Draw the line that makes the average error as small
as possible"

USE CASES:
----------
✓ Simple predictions (house prices, sales, temperature)
✓ Understanding feature importance
✓ Baseline model before trying complex methods
✓ When you have more samples than features
✓ When features are not highly correlated

WHEN NOT TO USE:
----------------
✗ Many correlated features (multicollinearity)
✗ More features than samples
✗ Outliers in data (use robust regression instead)
✗ Need sparse solutions (use Lasso instead)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

print("="*70)
print("ORDINARY LEAST SQUARES - LINEAR REGRESSION")
print("="*70)

# ============================================================================
# EXAMPLE 1: Simple Linear Regression (1 feature)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Simple Linear Regression")
print("="*70)

# Generate sample data
np.random.seed(42)
X_simple = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_simple = 2 * X_simple.flatten() + 3 + np.random.normal(0, 0.5, 10)

# Create and fit model
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

# Predictions
y_pred_simple = model_simple.predict(X_simple)

print(f"\nLearned equation: y = {model_simple.coef_[0]:.3f}x + {model_simple.intercept_:.3f}")
print(f"True equation:    y = 2.000x + 3.000")
print(f"\nR² Score: {r2_score(y_simple, y_pred_simple):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_simple, y_pred_simple)):.4f}")

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_simple, y_simple, color='blue', s=100, alpha=0.6, label='Data')
plt.plot(X_simple, y_pred_simple, color='red', linewidth=2, label='OLS Fit')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Simple Linear Regression', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# EXAMPLE 2: Multiple Linear Regression (multiple features)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Multiple Linear Regression")
print("="*70)

# Generate data with 3 features
np.random.seed(42)
n_samples = 100
X_multi = np.random.randn(n_samples, 3)
# True relationship: y = 2*x1 + 3*x2 - 1*x3 + 5
y_multi = 2*X_multi[:, 0] + 3*X_multi[:, 1] - 1*X_multi[:, 2] + 5 + np.random.normal(0, 0.5, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Fit model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Predictions
y_pred_train = model_multi.predict(X_train)
y_pred_test = model_multi.predict(X_test)

print("\nLearned coefficients:")
print(f"  w1 = {model_multi.coef_[0]:.3f} (true: 2.000)")
print(f"  w2 = {model_multi.coef_[1]:.3f} (true: 3.000)")
print(f"  w3 = {model_multi.coef_[2]:.3f} (true: -1.000)")
print(f"  intercept = {model_multi.intercept_:.3f} (true: 5.000)")

print(f"\nTraining R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# Visualize predictions vs actual
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6, s=80, color='green', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Multiple Regression: Predictions', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# EXAMPLE 3: Non-Negative Least Squares
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Non-Negative Least Squares")
print("="*70)
print("Useful when coefficients represent physical quantities (prices, counts)")

# Generate data where coefficients should be positive
np.random.seed(42)
X_pos = np.random.rand(50, 2)
y_pos = 3*X_pos[:, 0] + 2*X_pos[:, 1] + np.random.normal(0, 0.1, 50)

# Regular OLS
model_regular = LinearRegression()
model_regular.fit(X_pos, y_pos)

# Non-negative OLS
model_nonneg = LinearRegression(positive=True)
model_nonneg.fit(X_pos, y_pos)

print("\nRegular OLS coefficients:")
print(f"  w1 = {model_regular.coef_[0]:.3f}")
print(f"  w2 = {model_regular.coef_[1]:.3f}")

print("\nNon-negative OLS coefficients:")
print(f"  w1 = {model_nonneg.coef_[0]:.3f} (≥ 0)")
print(f"  w2 = {model_nonneg.coef_[1]:.3f} (≥ 0)")

# Visualize coefficient comparison
plt.subplot(1, 3, 3)
x_labels = ['Feature 1', 'Feature 2']
regular_coefs = model_regular.coef_
nonneg_coefs = model_nonneg.coef_

x_pos_plot = np.arange(len(x_labels))
width = 0.35

plt.bar(x_pos_plot - width/2, regular_coefs, width, label='Regular OLS', alpha=0.7, color='blue')
plt.bar(x_pos_plot + width/2, nonneg_coefs, width, label='Non-negative OLS', alpha=0.7, color='green')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Regular vs Non-negative OLS', fontsize=13, fontweight='bold')
plt.xticks(x_pos_plot, x_labels)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('01_ols_linear_regression.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: 01_ols_linear_regression.png")
plt.show()

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. OLS is the simplest and fastest linear regression method
2. Works best when:
   - More samples than features
   - Features are not highly correlated
   - No significant outliers
3. Provides interpretable coefficients
4. Can constrain coefficients to be non-negative with positive=True
5. Complexity: O(n_samples × n_features²)

NEXT STEPS:
-----------
- If features are correlated → Try Ridge (02_ridge_regression.py)
- If you want sparse solutions → Try Lasso (03_lasso_regression.py)
- If you have outliers → Try Robust Regression (11_robust_regression.py)
""")
