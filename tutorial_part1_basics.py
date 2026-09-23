"""Linear Regression - The Basics"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple example
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', s=100, label='Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Best Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression_basic.png')
plt.show()

print(f"Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
