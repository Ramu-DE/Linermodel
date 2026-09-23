"""
LASSO REGRESSION - L1 Regularization
=====================================

WHAT IS IT?
-----------
Lasso (Least Absolute Shrinkage and Selection Operator) uses L1 penalty.
Unlike Ridge, Lasso can set coefficients to EXACTLY ZERO = automatic feature selection!

MATHEMATICAL FORMULA:
--------------------
Minimizes: (1/2n)||Xw - y||² + α||w||₁

Where:
- α (alpha) = regularization strength
- ||w||₁ = sum of absolute values of coefficients (L1 norm)

INTUITION:
----------
Lasso says: "Fit the data well, BUT use as few features as possible"
It automatically identifies and removes unimportant features by setting their coefficients to 0.

Think of it as: "Keep only what matters, discard the rest"

LASSO vs RIDGE:
---------------
Lasso (L1): Can set coefficients to ZERO → Feature selection
Ridge (L2): Shrinks coefficients but keeps all features

USE CASES:
----------
✓ Feature selection (automatic)
✓ High-dimensional data with many irrelevant features
✓ Want interpretable sparse models
✓ Compressed sensing
✓ When only few features are truly important

KEY PARAMETER:
--------------
alpha (α): Controls sparsity
- α = 0: Same as OLS
- α → ∞: All coefficients → 0
- Larger α = more coefficients become zero
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*70)
print("LASSO REGRESSION - L1 REGULARIZATION & FEATURE SELECTION")
print("="*70)

# ============================================================================
# EXAMPLE 1: Lasso for Feature Selection
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Automatic Feature Selection with Lasso")
print("="*70)

# Generate data: only 3 out of 10 features are important
np.random.seed(42)
n_samples, n_features = 100, 10
X_sparse = np.random.randn(n_samples, n_features)
# Only features 0, 3, 7 are important
true_coef = np.zeros(n_features)
true_coef[0] = 3.0
true_coef[3] = -2.0
true_coef[7] = 1.5
y_sparse = X_sparse @ true_coef + np.random.normal(0, 0.5, n_samples)

# Standardize
scaler = StandardScaler()
X_sparse_scaled = scaler.fit_transform(X_sparse)

# Fit different models
ols = LinearRegression()
ols.fit(X_sparse_scaled, y_sparse)

ridge = Ridge(alpha=1.0)
ridge.fit(X_sparse_scaled, y_sparse)

lasso = Lasso(alpha=0.1)
lasso.fit(X_sparse_scaled, y_sparse)

print("\nTrue coefficients (only 3 non-zero):")
print(f"{true_coef}")
print(f"\nOLS coefficients (all non-zero):")
print(f"{ols.coef_}")
print(f"\nRidge coefficients (all non-zero, but small):")
print(f"{ridge.coef_}")
print(f"\nLasso coefficients (sparse - many zeros!):")
print(f"{lasso.coef_}")
print(f"\nNumber of non-zero coefficients:")
print(f"  True: {np.sum(true_coef != 0)}")
print(f"  OLS: {np.sum(np.abs(ols.coef_) > 0.01)}")
print(f"  Ridge: {np.sum(np.abs(ridge.coef_) > 0.01)}")
print(f"  Lasso: {np.sum(np.abs(lasso.coef_) > 0.01)}")

# Visualize
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
x_pos = np.arange(n_features)
width = 0.2

plt.bar(x_pos - 1.5*width, true_coef, width, label='True', alpha=0.8, color='black')
plt.bar(x_pos - 0.5*width, ols.coef_, width, label='OLS', alpha=0.8, color='blue')
plt.bar(x_pos + 0.5*width, ridge.coef_, width, label='Ridge', alpha=0.8, color='green')
plt.bar(x_pos + 1.5*width, lasso.coef_, width, label='Lasso', alpha=0.8, color='red')

plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Feature Selection: Lasso Sets Coefficients to Zero', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# ============================================================================
# EXAMPLE 2: Lasso Path - Effect of Alpha
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Lasso Path - How Alpha Controls Sparsity")
print("="*70)

# Compute Lasso path
alphas_range = np.logspace(-2, 1, 100)
coefs_lasso = []
n_nonzero = []

for alpha in alphas_range:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(X_sparse_scaled, y_sparse)
    coefs_lasso.append(lasso_temp.coef_)
    n_nonzero.append(np.sum(np.abs(lasso_temp.coef_) > 0.01))

coefs_lasso = np.array(coefs_lasso)

plt.subplot(2, 3, 2)
for i in range(n_features):
    plt.plot(alphas_range, coefs_lasso[:, i], linewidth=2)

plt.xscale('log')
plt.xlabel('Alpha (regularization strength)', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Lasso Path: Coefficients Become Zero', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# Plot number of non-zero coefficients
plt.subplot(2, 3, 3)
plt.plot(alphas_range, n_nonzero, linewidth=3, color='purple')
plt.xscale('log')
plt.xlabel('Alpha', fontsize=11)
plt.ylabel('Number of Non-Zero Coefficients', fontsize=11)
plt.title('Sparsity vs Alpha', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# ============================================================================
# EXAMPLE 3: LassoCV - Automatic Alpha Selection
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: LassoCV - Cross-Validation for Alpha")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse_scaled, y_sparse, test_size=0.3, random_state=42)

# Use LassoCV
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

print(f"\nBest alpha found: {lasso_cv.alpha_:.4f}")
print(f"Number of selected features: {np.sum(np.abs(lasso_cv.coef_) > 0.01)}/{n_features}")

# Compare performance
y_pred_lasso = lasso_cv.predict(X_test)
print(f"\nTest R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")

# Visualize selected features
plt.subplot(2, 3, 4)
selected_features = np.abs(lasso_cv.coef_) > 0.01
colors = ['red' if s else 'lightgray' for s in selected_features]
plt.bar(range(n_features), np.abs(lasso_cv.coef_), color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('|Coefficient|', fontsize=11)
plt.title(f'Selected Features (Red) by LassoCV', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# ============================================================================
# EXAMPLE 4: Lasso vs Ridge Comparison
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Lasso vs Ridge - Key Differences")
print("="*70)

# Generate data with correlated features
X_corr = np.random.randn(100, 2)
X_corr[:, 1] = X_corr[:, 0] + np.random.normal(0, 0.3, 100)
y_corr = X_corr[:, 0] + X_corr[:, 1] + np.random.normal(0, 0.5, 100)

scaler_corr = StandardScaler()
X_corr_scaled = scaler_corr.fit_transform(X_corr)

# Fit models with same alpha
alpha_compare = 0.5
lasso_comp = Lasso(alpha=alpha_compare)
ridge_comp = Ridge(alpha=alpha_compare)

lasso_comp.fit(X_corr_scaled, y_corr)
ridge_comp.fit(X_corr_scaled, y_corr)

print(f"\nWith correlated features (α={alpha_compare}):")
print(f"Lasso coefficients: {lasso_comp.coef_}")
print(f"Ridge coefficients: {ridge_comp.coef_}")
print("\nLasso picks one feature, Ridge uses both!")

plt.subplot(2, 3, 5)
x_pos = np.arange(2)
width = 0.35

plt.bar(x_pos - width/2, lasso_comp.coef_, width, label='Lasso', alpha=0.7, color='red')
plt.bar(x_pos + width/2, ridge_comp.coef_, width, label='Ridge', alpha=0.7, color='green')
plt.xlabel('Feature (correlated)', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Lasso vs Ridge with Correlation', fontsize=12, fontweight='bold')
plt.xticks(x_pos, ['Feature 1', 'Feature 2'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Summary
plt.subplot(2, 3, 6)
summary_text = """
LASSO SUMMARY

✓ L1 penalty: α||w||₁
✓ Sets coefficients to ZERO
✓ Automatic feature selection
✓ Sparse solutions
✓ Good for interpretability

KEY DIFFERENCES:
Lasso (L1): Sparse, feature selection
Ridge (L2): Dense, all features

WHEN TO USE LASSO:
→ Many irrelevant features
→ Want automatic selection
→ Need interpretable model
→ Compressed sensing

WHEN TO USE RIDGE:
→ All features important
→ Correlated features
→ More stable predictions

Use LassoCV for automatic α!
"""
plt.text(0.1, 0.5, summary_text, fontsize=9.5, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.axis('off')

plt.tight_layout()
plt.savefig('03_lasso_regression.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: 03_lasso_regression.png")
plt.show()

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Lasso uses L1 penalty → sets coefficients to EXACTLY ZERO
2. Automatic feature selection - removes unimportant features
3. Creates sparse, interpretable models
4. Use LassoCV for automatic alpha selection
5. When features are correlated, Lasso picks one arbitrarily
6. Great for high-dimensional data with few important features

NEXT: Try ElasticNet (04_elasticnet.py) for best of both worlds!
""")
