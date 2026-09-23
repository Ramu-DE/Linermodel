"""Logistic Regression - Classification"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create binary classification data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, 
                           random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Create decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, edgecolors='black', 
            label='Class 0', alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, edgecolors='black', 
            label='Class 1', alpha=0.7)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Logistic Regression - Classification', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('logistic_regression.png', dpi=300)
plt.show()

accuracy = model.score(X, y)
print(f"\nAccuracy: {accuracy*100:.1f}%")
print("\nLogistic Regression predicts CATEGORIES (yes/no, spam/not spam)")
print("Unlike Linear Regression which predicts NUMBERS")
