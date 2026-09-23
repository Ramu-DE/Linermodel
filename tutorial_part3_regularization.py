"""Ridge and Lasso - Preventing Overfitting"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Create data with noise
np.random.seed(42)
X = np.random.randn(30, 1)
y = 2 * X.flatten() + np.random.randn(30) * 1.5

# Train three models
linear = LinearRegression().fit(X, y)
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.5).fit(X, y)

# Predictions
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)

plt.figure(figsize=(12, 5))

# Plot comparison
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.6, s=80, label='Data')
plt.plot(X_test, linear.predict(X_test), 'r-', linewidth=2, label='Linear')
plt.plot(X_test, ridge.predict(X_test), 'g-', linewidth=2, label='Ridge')
plt.plot(X_test, lasso.predict(X_test), 'm-', linewidth=2, label='Lasso')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Regularization Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Coefficients comparison
plt.subplot(1, 2, 2)
models = ['Linear', 'Ridge', 'Lasso']
coefs = [linear.coef_[0], ridge.coef_[0], lasso.coef_[0]]
colors = ['red', 'green', 'magenta']
plt.bar(models, coefs, color=colors, alpha=0.7)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('How Regularization Shrinks Coefficients', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300)
plt.show()

print("\nCoefficients:")
print(f"Linear: {linear.coef_[0]:.3f}")
print(f"Ridge:  {ridge.coef_[0]:.3f} (smaller - more stable)")
print(f"Lasso:  {lasso.coef_[0]:.3f} (can be zero - feature selection)")
