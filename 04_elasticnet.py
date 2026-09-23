"""
ELASTIC-NET - Best of Both Worlds (L1 + L2)
============================================

WHAT IS IT?
-----------
ElasticNet combines Ridge (L2) and Lasso (L1) penalties.
It gets the benefits of both: feature selection AND stability!

MATHEMATICAL FORMULA:
--------------------
Minimizes: (1/2n)||Xw - y||² + α·ρ||w||₁ + α·(1-ρ)/2·||w||²

Where:
- α (alpha) = overall regularization strength
- ρ (l1_ratio) = balance between L1 and L2
  - ρ = 0: Pure Ridge (L2 only)
  - ρ = 1: Pure Lasso (L1 only)
  - 0 < ρ < 1: Mix of both

INTUITION:
----------
Lasso can be unstable when features are correlated (picks one randomly).
Ridge keeps all features but doesn't do selection.
ElasticNet: "Select features like Lasso, but be stable like Ridge"

USE CASES:
----------
✓ Many correlated features (better than Lasso alone)
✓ Want feature selection with stability
✓ High-dimensional data
✓ When Lasso is too aggressive
✓ Genomics, text analysis, image processing

ELASTICNET vs LASSO vs RIDGE:
------------------------------
ElasticNet: Feature selection + handles correlation
Lasso: Feature selection, unstable with correlation
Ridge: No selection, handles correlation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*70)
print("ELASTIC-NET - L1 + L2 REGULARIZATION")
print("="*70)

# ============================================================================
# EXAMPLE 1: ElasticNet with Correlated Features
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: ElasticNet vs Lasso with Correlated Features")
print("="*70)

# Generate correlated features
np.random.seed(42)
n_samples = 100
X_base = np.random.randn(n_samples, 1)
# Create 3 groups of correlated features
X_corr = np.hstack([
    X_base + np.random.normal(0, 0.1, (n_samples, 3)),  # Group 1
    -X_base + np.random.normal(0, 0.1, (n_samples, 3)), # Group 2
    np.random.randn(n_samples, 4)  # Noise features
])
# True model uses features from both groups
y_corr = 2*X_corr[:, 0] + 2*X_corr[:, 1] - 1.5*X_corr[:, 3] - 1.5*X_corr[:, 4] + np.random.normal(0, 0.5, n_samples)

# Standardize
scaler = StandardScaler()
X_corr_scaled = scaler.fit_transform(X_corr)

# Fit models
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_corr_scaled, y_corr)

elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
elasticnet.fit(X_corr_scaled, y_corr)

ridge = Ridge(alpha=0.1)
ridge.fit(X_corr_scaled, y_corr)

print("\nCoefficients with correlated features:")
print(f"Lasso:      {lasso.coef_}")
print(f"ElasticNet: {elasticnet.coef_}")
print(f"Ridge:      {ridge.coef_}")

print(f"\nNon-zero coefficients:")
print(f"Lasso:      {np.sum(np.abs(lasso.coef_) > 0.01)}")
print(f"ElasticNet: {np.sum(np.abs(elasticnet.coef_) > 0.01)}")
print(f"Ridge:      {np.sum(np.abs(ridge.coef_) > 0.01)}")

# Visualize
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
x_pos = np.arange(10)
width = 0.25

plt.bar(x_pos - width, lasso.coef_, width, label='Lasso', alpha=0.7, color='red')
plt.bar(x_pos, elasticnet.coef_, width, label='ElasticNet', alpha=0.7, color='purple')
plt.bar(x_pos + width, ridge.coef_, width, label='Ridge', alpha=0.7, color='green')

plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Lasso vs ElasticNet vs Ridge', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add annotations for feature groups
plt.text(1, plt.ylim()[1]*0.9, 'Group 1\n(correlated)', ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
plt.text(4, plt.ylim()[1]*0.9, 'Group 2\n(correlated)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))
plt.text(7.5, plt.ylim()[1]*0.9, 'Noise', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# ============================================================================
# EXAMPLE 2: Effect of l1_ratio
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Effect of l1_ratio Parameter")
print("="*70)

# Try different l1_ratio values
l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
models_ratio = {}

for ratio in l1_ratios:
    if ratio == 0.0:
        model = Ridge(alpha=0.1)
    elif ratio == 1.0:
        model = Lasso(alpha=0.1, max_iter=10000)
    else:
        model = ElasticNet(alpha=0.1, l1_ratio=ratio, max_iter=10000)
    model.fit(X_corr_scaled, y_corr)
    models_ratio[ratio] = model

print("\nEffect of l1_ratio on sparsity:")
for ratio, model in models_ratio.items():
    n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
    if ratio == 0.0:
        name = "Ridge (L2 only)"
    elif ratio == 1.0:
        name = "Lasso (L1 only)"
    else:
        name = f"ElasticNet (L1={ratio:.2f})"
    print(f"  {name:25s}: {n_nonzero} non-zero coefficients")

# Visualize l1_ratio effect
plt.subplot(2, 3, 2)
for ratio, model in models_ratio.items():
    if ratio == 0.0:
        label = 'Ridge (ρ=0)'
        color = 'green'
    elif ratio == 1.0:
        label = 'Lasso (ρ=1)'
        color = 'red'
    else:
        label = f'ρ={ratio}'
        color = 'purple'
    plt.plot(range(10), model.coef_, marker='o', label=label, linewidth=2, 
             markersize=6, alpha=0.7, color=color)

plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('Coefficient Value', fontsize=11)
plt.title('Effect of l1_ratio on Coefficients', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# ============================================================================
# EXAMPLE 3: ElasticNetCV - Automatic Parameter Selection
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: ElasticNetCV - Cross-Validation")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_corr_scaled, y_corr, test_size=0.3, random_state=42)

# Use ElasticNetCV to find best alpha and l1_ratio
elasticnet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                              cv=5, random_state=42, max_iter=10000)
elasticnet_cv.fit(X_train, y_train)

print(f"\nBest parameters found:")
print(f"  alpha: {elasticnet_cv.alpha_:.4f}")
print(f"  l1_ratio: {elasticnet_cv.l1_ratio_:.4f}")
print(f"  Selected features: {np.sum(np.abs(elasticnet_cv.coef_) > 0.01)}/10")

# Compare performance
y_pred = elasticnet_cv.predict(X_test)
print(f"\nTest Performance:")
print(f"  R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Visualize predictions
plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.6, s=80, color='purple', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Values', fontsize=11)
plt.ylabel('Predicted Values', fontsize=11)
plt.title('ElasticNetCV Predictions', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# EXAMPLE 4: Comparison on High-Dimensional Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: High-Dimensional Data (n_features > n_samples)")
print("="*70)

# Generate high-dimensional data
np.random.seed(42)
n_samples_hd = 50
n_features_hd = 100
X_hd = np.random.randn(n_samples_hd, n_features_hd)
# Only 5 features are important
true_coef_hd = np.zeros(n_features_hd)
true_coef_hd[:5] = [3, -2, 1.5, -1, 0.5]
y_hd = X_hd @ true_coef_hd + np.random.normal(0, 0.5, n_samples_hd)

scaler_hd = StandardScaler()
X_hd_scaled = scaler_hd.fit_transform(X_hd)

# Fit models
lasso_hd = Lasso(alpha=0.1, max_iter=10000)
lasso_hd.fit(X_hd_scaled, y_hd)

elasticnet_hd = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
elasticnet_hd.fit(X_hd_scaled, y_hd)

print(f"\nHigh-dimensional results (100 features, 50 samples):")
print(f"  True non-zero: 5")
print(f"  Lasso selected: {np.sum(np.abs(lasso_hd.coef_) > 0.01)}")
print(f"  ElasticNet selected: {np.sum(np.abs(elasticnet_hd.coef_) > 0.01)}")

# Visualize sparsity pattern
plt.subplot(2, 3, 4)
plt.plot(np.abs(true_coef_hd), 'k-', linewidth=2, label='True', alpha=0.7)
plt.plot(np.abs(lasso_hd.coef_), 'r-', linewidth=1.5, label='Lasso', alpha=0.7)
plt.plot(np.abs(elasticnet_hd.coef_), 'purple', linewidth=1.5, label='ElasticNet', alpha=0.7)
plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('|Coefficient|', fontsize=11)
plt.title('High-Dimensional Sparsity Pattern', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)  # Focus on first 20 features

# ============================================================================
# EXAMPLE 5: Stability Analysis
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Stability with Different Data Splits")
print("="*70)

# Test stability across multiple random splits
n_splits = 10
lasso_coefs = []
elasticnet_coefs = []

for i in range(n_splits):
    X_temp, _, y_temp, _ = train_test_split(X_corr_scaled, y_corr, 
                                             test_size=0.3, random_state=i)
    
    lasso_temp = Lasso(alpha=0.1, max_iter=10000)
    lasso_temp.fit(X_temp, y_temp)
    lasso_coefs.append(lasso_temp.coef_)
    
    en_temp = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    en_temp.fit(X_temp, y_temp)
    elasticnet_coefs.append(en_temp.coef_)

lasso_coefs = np.array(lasso_coefs)
elasticnet_coefs = np.array(elasticnet_coefs)

# Calculate stability (standard deviation across splits)
lasso_std = np.std(lasso_coefs, axis=0)
elasticnet_std = np.std(elasticnet_coefs, axis=0)

print(f"\nCoefficient stability (lower = more stable):")
print(f"  Lasso average std: {np.mean(lasso_std):.4f}")
print(f"  ElasticNet average std: {np.mean(elasticnet_std):.4f}")
print("\nElasticNet is more stable!")

plt.subplot(2, 3, 5)
x_pos = np.arange(10)
plt.bar(x_pos - 0.2, lasso_std, 0.4, label='Lasso', alpha=0.7, color='red')
plt.bar(x_pos + 0.2, elasticnet_std, 0.4, label='ElasticNet', alpha=0.7, color='purple')
plt.xlabel('Feature Index', fontsize=11)
plt.ylabel('Coefficient Std Dev', fontsize=11)
plt.title('Stability Across Data Splits', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Summary
plt.subplot(2, 3, 6)
summary_text = """
ELASTIC-NET SUMMARY

✓ Combines L1 + L2 penalties
✓ Feature selection + stability
✓ Better than Lasso for correlated features
✓ Controlled by α and l1_ratio

PARAMETERS:
• alpha: Overall regularization
• l1_ratio (ρ): L1 vs L2 balance
  - ρ=0: Pure Ridge
  - ρ=1: Pure Lasso
  - 0<ρ<1: Mix

WHEN TO USE:
→ Correlated features + need selection
→ Lasso is too unstable
→ High-dimensional data
→ Genomics, text analysis

Use ElasticNetCV for automatic
parameter selection!
"""
plt.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
plt.axis('off')

plt.tight_layout()
plt.savefig('04_elasticnet.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: 04_elasticnet.png")
plt.show()

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. ElasticNet = Lasso + Ridge (best of both worlds)
2. l1_ratio controls the mix: 0=Ridge, 1=Lasso, 0.5=balanced
3. More stable than Lasso with correlated features
4. Still does feature selection (sets coefficients to zero)
5. Use ElasticNetCV to find optimal alpha and l1_ratio
6. Great for high-dimensional data with correlation

DECISION GUIDE:
- Need feature selection + have correlation? → ElasticNet
- Only need selection, no correlation? → Lasso
- No selection needed? → Ridge
""")
