"""
🎉 WELCOME TO LINEAR MODELS TUTORIAL!
======================================

This script will help you get started with your learning journey.
Run this first to check your setup and see what's available.
"""

import sys

print("="*70)
print("🎓 LINEAR MODELS TUTORIAL - SETUP CHECK")
print("="*70)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check required packages
print("\n📦 Checking required packages...")
required_packages = {
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'sklearn': 'Scikit-learn'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name} installed")
    except ImportError:
        print(f"  ✗ {name} NOT installed")
        missing_packages.append(name)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("\nTo install, run:")
    print("  pip install numpy matplotlib scikit-learn")
    sys.exit(1)
else:
    print("\n✅ All required packages are installed!")

# Show available tutorials
print("\n" + "="*70)
print("📚 AVAILABLE TUTORIALS")
print("="*70)

tutorials = {
    "🌱 BEGINNER (Start Here!)": [
        ("START_HERE.txt", "Quick start guide"),
        ("tutorial_part1_basics.py", "Linear regression basics"),
        ("tutorial_part2_house_prices.py", "Real-world example"),
        ("tutorial_part3_regularization.py", "Ridge vs Lasso"),
        ("tutorial_part4_logistic.py", "Classification"),
    ],
    "🌿 INTERMEDIATE": [
        ("00_complete_overview.py", "All models overview"),
        ("01_ols_linear_regression.py", "OLS in depth"),
        ("02_ridge_regression.py", "Ridge regression"),
        ("03_lasso_regression.py", "Lasso regression"),
        ("04_elasticnet.py", "ElasticNet"),
    ],
    "📖 GUIDES": [
        ("README.md", "Complete documentation"),
        ("LEARNING_ROADMAP.md", "Learning path"),
        ("quick_reference.txt", "Quick reference"),
    ]
}

for category, files in tutorials.items():
    print(f"\n{category}")
    print("-" * 70)
    for filename, description in files:
        print(f"  • {filename:35s} - {description}")

# Quick start guide
print("\n" + "="*70)
print("🚀 QUICK START")
print("="*70)
print("""
OPTION 1: Complete Beginner
----------------------------
1. Read START_HERE.txt
2. Run: python tutorial_part1_basics.py
3. Run: python tutorial_part2_house_prices.py

OPTION 2: Some ML Experience
-----------------------------
1. Run: python 00_complete_overview.py
2. Run: python 01_ols_linear_regression.py
3. Continue with 02, 03, 04...

OPTION 3: Quick Reference
--------------------------
1. Open quick_reference.txt
2. Find your use case
3. Run the recommended tutorial
""")

# Decision helper
print("\n" + "="*70)
print("🤔 WHAT DO YOU WANT TO LEARN?")
print("="*70)
print("""
I want to predict NUMBERS (prices, sales, temperature):
  → Start with: tutorial_part2_house_prices.py

I want to predict CATEGORIES (yes/no, spam/not spam):
  → Start with: tutorial_part4_logistic.py

I want to understand REGULARIZATION (Ridge, Lasso):
  → Start with: tutorial_part3_regularization.py

I want to see ALL available models:
  → Start with: 00_complete_overview.py

I want a COMPLETE learning path:
  → Read: LEARNING_ROADMAP.md
""")

# Test visualization
print("\n" + "="*70)
print("🎨 TESTING VISUALIZATION")
print("="*70)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a simple test plot
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x, 2*x + 1, 'r-', linewidth=2, label='True Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Test Plot: Linear Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization test successful!")
    print("  Created: test_plot.png")
    print("\n  If you can see this file, visualizations will work in tutorials!")
    
except Exception as e:
    print(f"⚠️  Visualization test failed: {e}")
    print("  Tutorials will still work, but plots may not display.")

# Final message
print("\n" + "="*70)
print("✨ YOU'RE ALL SET!")
print("="*70)
print("""
Everything is ready to go! Here's what to do next:

1. Choose your starting point from the options above
2. Run the tutorial file: python <filename>
3. Read the code and comments carefully
4. Experiment by changing parameters
5. Have fun learning! 🎉

Tips for Success:
-----------------
✓ Run code, don't just read it
✓ Modify examples and see what happens
✓ Visualize everything
✓ Take breaks when needed
✓ Practice on your own data

Need help? Check README.md for detailed documentation.

Happy Learning! 🚀
""")

print("="*70)
