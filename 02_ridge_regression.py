"""
RIDGE REGRESSION - L2 Regularization
=====================================

WHAT IS IT?
-----------
Ridge adds a penalty for large coefficients to prevent overfitting.
It's OLS with an L2 (squared) penalty on the coefficient magnitudes.

MATHEMATICAL FORMULA:
--------------------
Minimizes: ||Xw - y||² + α||w||²

Where:
- α (alpha) = regularization strength
- ||w||² = sum of squared coefficients (L2 norm)

INTUITION:
----------
Regular OLS can create very large coefficients when features are correlated.
Ridge says: "Fit the data well, BUT keep coefficients small and stable"

Think of it as: "Don't put all your eggs in one basket"

KEY PARAMETER:
--------------
alpha (α): Controls regularization strength
- α = 0: Same as OLS (no regularization)
- α → ∞: All coefficients → 0
- Typical values: 0.01, 0.1, 1.0, 10, 100

USE CASES:
----------
✓ Many correlated features (multicollinearity)
✓ More stable predictions than OLS
✓ Preventing overfitting
✓ When all features are somewhat important
✓ Ridge Classifier for fast multiclass classification

RIDGE vs OLS:
-------------
Ridge: Shrinks coefficients, more stable, handles correlation
OLS: Can have large coefficients, unstable with correlation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*70)
print("RIDGE REGRESSION - L2 REGULARIZATION")
print("="*70)

# ============================================================================
# EXAMPLE 1: Ridge vs OLS with Correlated Features
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Ridge vs OLS with Correlated Features")
print("="*70)

# Generate correlated features
np.random.seed(42)
n_samples = 100
X_base = np.random.randn(n_samples, 1)
# Create correlated features
X_corr = np.hstack([X_base, X_base + np.random.normal(0, 0.1, (n_samples, 1)),
                    X_base + np.random.normal(0, 0.1, (n_samples, 1))])
y_corr = X_base.flatten() + np.random.normal(0, 0.5, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_corr, y_corr, test_size=0.3, random_state=42)

# Standardize features (important for Ridge!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# Fit Ridge with different alphas
alphas = [0.01, 0.1, 1.0, 10.0]
ridge_models = {}
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_models[alpha] = ridge

print("\nCoefficients comparison (features are highly correlated):")
print(f"OLS:        {ols.coef_}")
for alpha, model in ridge_models.items():
    print(f"Ridge(α={alpha:5.2f}): {model.coef_}")

# Visualize coefficients
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
models_list = ['OLS'] + [f'Ridge\nα={a}' for a in alphas]
coef_matrix = np.vstack([ols.coef_] + [m.coef_ for m in ridge_models.values()])

x_pos = np.arange(len(models_list))
for i in range(3):
    plt.plot(x_pos, coef_matrix[:, i], marker='o', linewidth=2, markersize=8, label=f'Feature {i+1}')

plt.xlabel('Model', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('How Ridge Shrinks Coefficients', fontsize=12, fontweight='bold')
plt.xticks(x_pos, models_list)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# ============================================================================
# EXAMPLE 2: Ridge Path - Effect of Alpha
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Ridge Path - Visualizing Alpha Effect")
print("="*70)

# Generate data
np.random.seed(42)
X_path = np.random.randn(50, 5)
y_path = X_path @ np.array([1, 2, -1, 0.5, -0.5]) + np.random.normal(0, 0.5, 50)

# Standardize
scaler_path = StandardScaler()
X_path_scaled = scaler_path.fit_transform(X_path)

# Compute coefficients for different alphas
alphas_range = np.logspace(-2, 3, 100)
coefs = []
for alpha in alphas_range:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_path_scaled, y_path)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

plt.subplot(2, 3, 2)
for i in range(5):
    plt.plot(alphas_range, coefs[:, i], linewidth=2, label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Alpha (regularization strength)', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Ridge Path: Coefficients vs Alpha', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# ============================================================================
# EXAMPLE 3: RidgeCV - Automatic Alpha Selection
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: RidgeCV - Cross-Validation for Alpha")
print("="*70)

# Use RidgeCV to find best alpha
alphas_cv = np.logspace(-2, 3, 50)
ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha found by cross-validation: {ridge_cv.alpha_:.4f}")

# Compare performance
y_pred_ols = ols.predict(X_test_scaled)
y_pred_ridge_cv = ridge_cv.predict(X_test_scaled)

print(f"\nTest Set Performance:")
print(f"OLS R²:       {r2_score(y_test, y_pred_ols):.4f}")
print(f"Ridge CV R²:  {r2_score(y_test, y_pred_ridge_cv):.4f}")
print(f"OLS RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f}")
print(f"Ridge CV RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge_cv)):.4f}")

# Visualize predictions
plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_ols, alpha=0.6, s=80, label='OLS', color='blue', edgecolors='black')
plt.scatter(y_test, y_pred_ridge_cv, alpha=0.6, s=80, label='Ridge CV', color='red', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', linewidth=2, label='Perfect')
plt.xlabel('Actual Values', fontsize=11)
plt.ylabel('Predicted Values', fontsize=11)
plt.title('OLS vs Ridge: Predictions', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# EXAMPLE 4: Ridge Classifier
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Ridge Classifier for Classification")
print("="*70)

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate classification data
X_clf, y_clf = make_classification(n_samples=200, n_features=10, n_informative=5,
                                    n_redundant=5, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42)

# Fit models
ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train_clf, y_train_clf)

logistic_clf = LogisticRegression(max_iter=1000)
logistic_clf.fit(X_train_clf, y_train_clf)

# Compare
y_pred_ridge_clf = ridge_clf.predict(X_test_clf)
y_pred_logistic = logistic_clf.predict(X_test_clf)

print(f"\nClassification Accuracy:")
print(f"Ridge Classifier:      {accuracy_score(y_test_clf, y_pred_ridge_clf):.4f}")
print(f"Logistic Regression:   {accuracy_score(y_test_clf, y_pred_logistic):.4f}")
print("\nRidge Classifier is often faster for multiclass problems!")

# Visualize coefficient magnitudes
plt.subplot(2, 3, 4)
plt.bar(range(len(ols.coef_)), np.abs(ols.coef_), alpha=0.7, label='OLS', color='blue')
plt.bar(range(len(ridge_cv.coef_)), np.abs(ridge_cv.coef_), alpha=0.7, label='Ridge', color='red')
plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('|Coefficient|', fontsize=11)
plt.title('Coefficient Magnitudes', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# ============================================================================
# EXAMPLE 5: Effect of Standardization
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Importance of Feature Standardization")
print("="*70)

# Create data with different scales
X_scale = np.random.randn(100, 3)
X_scale[:, 0] *= 1      # Feature 1: scale 1
X_scale[:, 1] *= 10     # Feature 2: scale 10
X_scale[:, 2] *= 100    # Feature 3: scale 100
y_scale = X_scale[:, 0] + X_scale[:, 1] + X_scale[:, 2] + np.random.normal(0, 1, 100)

# Ridge without standardization
ridge_no_scale = Ridge(alpha=1.0)
ridge_no_scale.fit(X_scale, y_scale)

# Ridge with standardization
scaler_demo = StandardScaler()
X_scale_std = scaler_demo.fit_transform(X_scale)
ridge_with_scale = Ridge(alpha=1.0)
ridge_with_scale.fit(X_scale_std, y_scale)

print("\nWithout standardization (features have different scales):")
print(f"Coefficients: {ridge_no_scale.coef_}")
print("\nWith standardization:")
print(f"Coefficients: {ridge_with_scale.coef_}")
print("\n⚠️  Always standardize features before using Ridge!")

plt.subplot(2, 3, 5)
features = ['Feature 1\n(scale=1)', 'Feature 2\n(scale=10)', 'Feature 3\n(scale=100)']
x_pos = np.arange(len(features))
width = 0.35

plt.bar(x_pos - width/2, ridge_no_scale.coef_, width, label='No Scaling', alpha=0.7, color='orange')
plt.bar(x_pos + width/2, ridge_with_scale.coef_, width, label='With Scaling', alpha=0.7, color='green')
plt.xlabel('Features', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Impact of Standardization', fontsize=12, fontweight='bold')
plt.xticks(x_pos, features, fontsize=9)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Summary plot
plt.subplot(2, 3, 6)
summary_text = """
RIDGE REGRESSION SUMMARY

✓ Adds L2 penalty: α||w||²
✓ Shrinks coefficients
✓ Handles correlated features
✓ More stable than OLS
✓ Never sets coefficients to 0

KEY POINTS:
• Always standardize features
• Use RidgeCV to find best α
• Larger α = more shrinkage
• Good for multicollinearity

WHEN TO USE:
→ Correlated features
→ Overfitting with OLS
→ Want all features included
→ Fast multiclass classification
"""
plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')

plt.tight_layout()
plt.savefig('02_ridge_regression.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: 02_ridge_regression.png")
plt.show()

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Ridge adds L2 penalty to prevent overfitting
2. Shrinks coefficients but never sets them to exactly zero
3. Excellent for handling correlated features (multicollinearity)
4. ALWAYS standardize features before using Ridge
5. Use RidgeCV for automatic alpha selection via cross-validation
6. RidgeClassifier is fast for multiclass classification

NEXT: Try Lasso (03_lasso_regression.py) for sparse solutions!
""")
