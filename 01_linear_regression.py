"""
LINEAR REGRESSION - The Foundation of Linear Models
====================================================

WHAT IS IT?
-----------
Linear Regression is like drawing the "best fit" straight line through your data.
It helps you predict a NUMBER (like price, temperature, sales) based on input features.

REAL-WORLD ANALOGY:
-------------------
Imagine plotting house sizes vs prices on graph paper. You see a pattern: 
bigger houses cost more. Linear regression finds the line that best represents 
this relationship, so you can predict prices for new houses.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("="*70)
print("PART 1: LINEAR REGRESSION - Understanding the Basics")
print("="*70)

# ============================================================================
# EXAMPLE 1: Simple Linear Regression (One Feature)
# ============================================================================

print("\n📊 EXAMPLE 1: Predicting House Prices from Size")
print("-" * 70)

# Create sample data
np.random.seed(42)
house_sizes = np.array([500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300])
# Price formula: base_price + (price_per_sqft * size) + some randomness
house_prices = 50000 + 100 * house_sizes + np.random.normal(0, 10000