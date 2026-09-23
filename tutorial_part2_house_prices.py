"""Real Example: Predicting House Prices"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# House data: size (sq ft) vs price ($1000s)
sizes = np.array([[600], [800], [1000], [1200], [1500], [1800], [2000], [2200]])
prices = np.array([150, 180, 220, 250, 300, 350, 380, 420])

model = LinearRegression()
model.fit(sizes, prices)

# Predict for new house
new_house = np.array([[1300]])
predicted_price = model.predict(new_house)

plt.figure(figsize=(10, 6))
plt.scatter(sizes, prices, color='green', s=150, label='Actual Prices', zorder=3)
plt.plot(sizes, model.predict(sizes), color='red', linewidth=2, label='Prediction Line')
plt.scatter(new_house, predicted_price, color='orange', s=200, marker='*', 
            label=f'New Prediction: ${predicted_price[0]:.0f}k', zorder=4)
plt.xlabel('House Size (sq ft)', fontsize=12)
plt.ylabel('Price ($1000s)', fontsize=12)
plt.title('House Price Prediction with Linear Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('house_price_prediction.png', dpi=300)
plt.show()

print(f"\nModel: Price = {model.coef_[0]:.3f} × Size + {model.intercept_:.2f}")
print(f"A 1300 sq ft house costs approximately: ${predicted_price[0]:.2f}k")
