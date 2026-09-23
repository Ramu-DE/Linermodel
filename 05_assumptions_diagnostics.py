"""
ASSUMPTIONS OF LINEAR REGRESSION - Diagnostics & Visualization
================================================================

This tutorial covers the 6 KEY ASSUMPTIONS of linear regression:

1. LINEARITY - Linear relationship between X and Y
2. HOMOSCEDASTICITY - Equal variance of errors
3. MULTIVARIATE NORMALITY - Errors are normally distributed
4. INDEPENDENCE - Observations are independent (no autocorrelation)
5. NO MULTICOLLINEARITY - Predictors are not correlated with each other
6. NO ENDOGENEITY - No correlation between predictors and errors

For each assumption we show:
- ✅ What it looks like when SATISFIED
- ❌ What it looks like when VIOLATED
- 🔧 How to detect the violation
- 💡 What to do about it
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

np.random.seed(42)

print("="*70)
print("ASSUMPTIONS OF LINEAR REGRESSION - COMPLETE DIAGNOSTICS")
print("="*70)

# Create figure with 6 rows x 2 columns (good vs bad for each assumption)
fig, axes = plt.subplots(6, 2, figsize=(14, 28))
fig.suptitle('6 Assumptions of Linear Regression\n✅ Satisfied (Left) vs ❌ Violated (Right)',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# ASSUMPTION 1: LINEARITY
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 1: LINEARITY")
print("="*70)
print("""
WHAT IT MEANS:
  The relationship between X and Y must be a straight line.
  
✅ GOOD: Points scatter around a straight line
❌ BAD:  Points follow a curve (quadratic, exponential, etc.)

HOW TO CHECK:
  - Plot Y vs X (should look linear)
  - Plot residuals vs fitted values (should be random, no pattern)

WHAT TO DO IF VIOLATED:
  - Transform features (log, sqrt, polynomial)
  - Use polynomial regression
  - Use non-linear models
""")

# ✅ Linear relationship (GOOD)
X_linear = np.linspace(0, 10, 100)
y_linear = 2 * X_linear + 3 + np.random.normal(0, 2, 100)

ax = axes[0, 0]
ax.scatter(X_linear, y_linear, alpha=0.5, s=30, color='blue')
model_lin = LinearRegression()
model_lin.fit(X_linear.reshape(-1, 1), y_linear)
ax.plot(X_linear, model_lin.predict(X_linear.reshape(-1, 1)), 'r-', linewidth=2)
ax.set_title('✅ Assumption 1: LINEARITY Satisfied', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.text(0.05, 0.95, 'Points scatter around\na straight line', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Non-linear relationship (BAD)
X_nonlinear = np.linspace(0, 10, 100)
y_nonlinear = 0.5 * X_nonlinear**2 + np.random.normal(0, 2, 100)

ax = axes[0, 1]
ax.scatter(X_nonlinear, y_nonlinear, alpha=0.5, s=30, color='red')
model_nonlin = LinearRegression()
model_nonlin.fit(X_nonlinear.reshape(-1, 1), y_nonlinear)
ax.plot(X_nonlinear, model_nonlin.predict(X_nonlinear.reshape(-1, 1)), 'k-', linewidth=2)
ax.set_title('❌ Assumption 1: LINEARITY Violated', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.text(0.05, 0.95, 'Points follow a CURVE\nnot a straight line!', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

# ============================================================================
# ASSUMPTION 2: HOMOSCEDASTICITY (Equal Variance)
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 2: HOMOSCEDASTICITY (Equal Variance)")
print("="*70)
print("""
WHAT IT MEANS:
  The spread (variance) of errors should be CONSTANT across all X values.
  
✅ GOOD: Residuals have same spread everywhere (like a band)
❌ BAD:  Residuals fan out (spread increases with X) = "heteroscedasticity"

HOW TO CHECK:
  - Plot residuals vs fitted values
  - Should look like a random cloud with constant width
  - Breusch-Pagan test or White's test

WHAT TO DO IF VIOLATED:
  - Log-transform the target variable
  - Use Weighted Least Squares (WLS)
  - Use robust standard errors
""")

# ✅ Homoscedastic (GOOD) - constant variance
X_homo = np.linspace(1, 10, 100)
y_homo = 3 * X_homo + np.random.normal(0, 2, 100)  # constant noise

ax = axes[1, 0]
model_homo = LinearRegression()
model_homo.fit(X_homo.reshape(-1, 1), y_homo)
residuals_homo = y_homo - model_homo.predict(X_homo.reshape(-1, 1))
ax.scatter(model_homo.predict(X_homo.reshape(-1, 1)), residuals_homo, alpha=0.5, s=30, color='blue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('✅ Assumption 2: HOMOSCEDASTICITY Satisfied', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.text(0.05, 0.95, 'Residuals have EQUAL\nspread everywhere', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Heteroscedastic (BAD) - variance increases with X
X_hetero = np.linspace(1, 10, 100)
y_hetero = 3 * X_hetero + np.random.normal(0, 1, 100) * X_hetero  # noise grows with X!

ax = axes[1, 1]
model_hetero = LinearRegression()
model_hetero.fit(X_hetero.reshape(-1, 1), y_hetero)
residuals_hetero = y_hetero - model_hetero.predict(X_hetero.reshape(-1, 1))
ax.scatter(model_hetero.predict(X_hetero.reshape(-1, 1)), residuals_hetero, alpha=0.5, s=30, color='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_title('❌ Assumption 2: HOMOSCEDASTICITY Violated', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.text(0.05, 0.95, 'Residuals FAN OUT!\nSpread increases →', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

# ============================================================================
# ASSUMPTION 3: NORMALITY OF ERRORS
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 3: MULTIVARIATE NORMALITY (Normal Errors)")
print("="*70)
print("""
WHAT IT MEANS:
  The residuals (errors) should follow a normal (bell-curve) distribution.
  
✅ GOOD: Residuals form a bell curve, Q-Q plot follows diagonal
❌ BAD:  Residuals are skewed, heavy-tailed, or bimodal

HOW TO CHECK:
  - Histogram of residuals (should be bell-shaped)
  - Q-Q plot (points should follow the diagonal line)
  - Shapiro-Wilk test or Kolmogorov-Smirnov test

WHAT TO DO IF VIOLATED:
  - Transform target variable (log, Box-Cox)
  - Remove outliers
  - Use robust regression
  - With large samples, this matters less (Central Limit Theorem)
""")

# ✅ Normal residuals (GOOD)
residuals_normal = np.random.normal(0, 1, 200)

ax = axes[2, 0]
ax.hist(residuals_normal, bins=20, density=True, alpha=0.6, color='blue', edgecolor='black')
# Overlay normal curve
x_norm = np.linspace(-4, 4, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, 0, 1), 'r-', linewidth=2, label='Normal curve')
ax.set_title('✅ Assumption 3: NORMALITY Satisfied', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Density')
ax.legend()
ax.text(0.05, 0.95, 'Bell-shaped distribution\n(symmetric, no skew)', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Non-normal residuals (BAD) - skewed
residuals_skewed = np.random.exponential(1, 200) - 1  # right-skewed

ax = axes[2, 1]
ax.hist(residuals_skewed, bins=20, density=True, alpha=0.6, color='red', edgecolor='black')
x_skew = np.linspace(-2, 5, 100)
ax.plot(x_skew, stats.norm.pdf(x_skew, np.mean(residuals_skewed), np.std(residuals_skewed)),
        'k--', linewidth=2, label='Expected normal')
ax.set_title('❌ Assumption 3: NORMALITY Violated', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Density')
ax.legend()
ax.text(0.05, 0.95, 'SKEWED distribution!\nNot bell-shaped', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

# ============================================================================
# ASSUMPTION 4: INDEPENDENCE (No Autocorrelation)
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 4: INDEPENDENCE (No Autocorrelation)")
print("="*70)
print("""
WHAT IT MEANS:
  Each observation should be independent of others.
  Residuals should NOT show patterns over time/order.
  
✅ GOOD: Residuals are randomly scattered (no pattern in sequence)
❌ BAD:  Residuals show waves or trends (autocorrelation)

HOW TO CHECK:
  - Plot residuals in order (should be random)
  - Durbin-Watson test (value near 2 = good)
  - Autocorrelation plot

WHAT TO DO IF VIOLATED:
  - Use time-series models (ARIMA)
  - Add lagged variables
  - Use Generalized Least Squares (GLS)
  - Common in time-series data!
""")

# ✅ Independent residuals (GOOD)
residuals_indep = np.random.normal(0, 1, 50)

ax = axes[3, 0]
ax.plot(range(50), residuals_indep, 'bo-', markersize=4, alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('✅ Assumption 4: INDEPENDENCE Satisfied', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('Observation Order')
ax.set_ylabel('Residual')
ax.text(0.05, 0.95, 'Random scatter\nNo pattern in sequence', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Autocorrelated residuals (BAD) - wave pattern
t = np.linspace(0, 4*np.pi, 50)
residuals_autocorr = np.sin(t) + np.random.normal(0, 0.2, 50)

ax = axes[3, 1]
ax.plot(range(50), residuals_autocorr, 'ro-', markersize=4, alpha=0.6)
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_title('❌ Assumption 4: INDEPENDENCE Violated', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('Observation Order')
ax.set_ylabel('Residual')
ax.text(0.05, 0.95, 'WAVE pattern!\nResiduals are correlated', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

# ============================================================================
# ASSUMPTION 5: NO MULTICOLLINEARITY
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 5: NO MULTICOLLINEARITY")
print("="*70)
print("""
WHAT IT MEANS:
  Predictor variables (features) should NOT be highly correlated with each other.
  
✅ GOOD: Features are independent (X1 and X2 are unrelated)
❌ BAD:  Features are highly correlated (X1 ≈ X2)

HOW TO CHECK:
  - Correlation matrix / heatmap
  - Variance Inflation Factor (VIF) > 5 or 10 = problem
  - Condition number of X'X matrix

WHAT TO DO IF VIOLATED:
  - Remove one of the correlated features
  - Use Ridge regression (designed for this!)
  - Use PCA to create uncorrelated features
  - Combine correlated features into one
""")

# ✅ No multicollinearity (GOOD) - independent features
X1_indep = np.random.normal(0, 1, 100)
X2_indep = np.random.normal(0, 1, 100)

ax = axes[4, 0]
ax.scatter(X1_indep, X2_indep, alpha=0.5, s=30, color='blue')
ax.set_title('✅ Assumption 5: NO MULTICOLLINEARITY', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
corr_good = np.corrcoef(X1_indep, X2_indep)[0, 1]
ax.text(0.05, 0.95, f'X₁ and X₂ are INDEPENDENT\nCorrelation = {corr_good:.3f}', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Multicollinearity (BAD) - highly correlated features
X1_corr = np.random.normal(0, 1, 100)
X2_corr = X1_corr + np.random.normal(0, 0.1, 100)  # X2 ≈ X1

ax = axes[4, 1]
ax.scatter(X1_corr, X2_corr, alpha=0.5, s=30, color='red')
ax.set_title('❌ Assumption 5: MULTICOLLINEARITY Present', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
corr_bad = np.corrcoef(X1_corr, X2_corr)[0, 1]
ax.text(0.05, 0.95, f'X₁ and X₂ are CORRELATED!\nCorrelation = {corr_bad:.3f}', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

# ============================================================================
# ASSUMPTION 6: NO ENDOGENEITY
# ============================================================================
print("\n" + "="*70)
print("ASSUMPTION 6: NO ENDOGENEITY (Exogeneity)")
print("="*70)
print("""
WHAT IT MEANS:
  Predictors should NOT be correlated with the error term.
  E(error | X) = 0 (errors are random given X)
  
✅ GOOD: Residuals vs X show no pattern (random cloud)
❌ BAD:  Residuals vs X show a trend or pattern

HOW TO CHECK:
  - Plot residuals vs each predictor
  - Should be a random cloud centered at 0
  - Hausman test (advanced)

WHAT TO DO IF VIOLATED:
  - Add missing variables (omitted variable bias)
  - Use instrumental variables
  - Use two-stage least squares (2SLS)
  - This often means your model is missing something!
""")

# ✅ No endogeneity (GOOD) - residuals uncorrelated with X
X_exog = np.linspace(0, 10, 100)
y_exog = 2 * X_exog + 5 + np.random.normal(0, 1.5, 100)
model_exog = LinearRegression()
model_exog.fit(X_exog.reshape(-1, 1), y_exog)
resid_exog = y_exog - model_exog.predict(X_exog.reshape(-1, 1))

ax = axes[5, 0]
ax.scatter(X_exog, resid_exog, alpha=0.5, s=30, color='blue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('✅ Assumption 6: NO ENDOGENEITY', fontsize=11, fontweight='bold', color='green')
ax.set_xlabel('X (Predictor)')
ax.set_ylabel('Residuals')
corr_resid_good = np.corrcoef(X_exog, resid_exog)[0, 1]
ax.text(0.05, 0.95, f'Residuals are RANDOM\nCorr(X, residuals) = {corr_resid_good:.3f}', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ❌ Endogeneity (BAD) - residuals correlated with X (omitted variable)
X_endog = np.linspace(0, 10, 100)
# True model: y = 2*X + 3*Z + noise, but we omit Z which is correlated with X
Z_hidden = 0.5 * X_endog + np.random.normal(0, 0.5, 100)
y_endog = 2 * X_endog + 3 * Z_hidden + np.random.normal(0, 1, 100)
# Fit model WITHOUT Z (omitted variable)
model_endog = LinearRegression()
model_endog.fit(X_endog.reshape(-1, 1), y_endog)
resid_endog = y_endog - model_endog.predict(X_endog.reshape(-1, 1))

ax = axes[5, 1]
ax.scatter(X_endog, resid_endog, alpha=0.5, s=30, color='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_title('❌ Assumption 6: ENDOGENEITY Present', fontsize=11, fontweight='bold', color='red')
ax.set_xlabel('X (Predictor)')
ax.set_ylabel('Residuals')
corr_resid_bad = np.corrcoef(X_endog, resid_endog)[0, 1]
ax.text(0.05, 0.95, f'Residuals show PATTERN!\nCorr(X, residuals) = {corr_resid_bad:.3f}', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('05_assumptions_diagnostics.png', dpi=200, bbox_inches='tight')
print("\n💾 Saved: 05_assumptions_diagnostics.png")
plt.show()

# ============================================================================
# QUANTITATIVE TESTS
# ============================================================================

print("\n" + "="*70)
print("QUANTITATIVE TESTS FOR ASSUMPTIONS")
print("="*70)

# Test 1: Linearity - using residuals vs fitted
print("\n📊 Test 1: LINEARITY")
print("   Visual check: Plot residuals vs fitted values")
print("   If you see a curve → linearity is violated")

# Test 2: Homoscedasticity - Breusch-Pagan (simplified)
print("\n📊 Test 2: HOMOSCEDASTICITY")
print("   Breusch-Pagan test (simplified):")
resid_sq = residuals_hetero**2
bp_model = LinearRegression()
bp_model.fit(model_hetero.predict(X_hetero.reshape(-1, 1)).reshape(-1, 1), resid_sq)
bp_r2 = bp_model.score(model_hetero.predict(X_hetero.reshape(-1, 1)).reshape(-1, 1), resid_sq)
print(f"   R² of residuals² vs fitted: {bp_r2:.4f}")
print(f"   {'❌ Heteroscedasticity detected!' if bp_r2 > 0.05 else '✅ Homoscedastic'}")

# Test 3: Normality - Shapiro-Wilk
print("\n📊 Test 3: NORMALITY")
stat_good, p_good = stats.shapiro(residuals_normal[:50])
stat_bad, p_bad = stats.shapiro(residuals_skewed[:50])
print(f"   Shapiro-Wilk test (normal residuals):  p-value = {p_good:.4f} {'✅ Normal' if p_good > 0.05 else '❌ Not normal'}")
print(f"   Shapiro-Wilk test (skewed residuals):  p-value = {p_bad:.4f} {'✅ Normal' if p_bad > 0.05 else '❌ Not normal'}")
print("   Rule: p-value > 0.05 → residuals are normal")

# Test 4: Independence - Durbin-Watson (simplified)
print("\n📊 Test 4: INDEPENDENCE (Durbin-Watson)")
def durbin_watson(residuals):
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

dw_good = durbin_watson(residuals_indep)
dw_bad = durbin_watson(residuals_autocorr)
print(f"   DW statistic (independent):     {dw_good:.4f} {'✅ Independent' if 1.5 < dw_good < 2.5 else '❌ Autocorrelated'}")
print(f"   DW statistic (autocorrelated):   {dw_bad:.4f} {'✅ Independent' if 1.5 < dw_bad < 2.5 else '❌ Autocorrelated'}")
print("   Rule: DW ≈ 2 → independent, DW < 1.5 or > 2.5 → autocorrelation")

# Test 5: Multicollinearity - VIF
print("\n📊 Test 5: MULTICOLLINEARITY (VIF)")
print("   Variance Inflation Factor (VIF):")
# Simple VIF calculation for 2 features
corr_matrix_good = np.corrcoef(X1_indep, X2_indep)
vif_good = 1 / (1 - corr_matrix_good[0, 1]**2)
corr_matrix_bad = np.corrcoef(X1_corr, X2_corr)
vif_bad = 1 / (1 - corr_matrix_bad[0, 1]**2)
print(f"   VIF (independent features):  {vif_good:.2f} {'✅ No multicollinearity' if vif_good < 5 else '❌ Multicollinearity!'}")
print(f"   VIF (correlated features):   {vif_bad:.2f} {'✅ No multicollinearity' if vif_bad < 5 else '❌ Multicollinearity!'}")
print("   Rule: VIF < 5 → OK, VIF > 10 → serious problem")

# Test 6: Endogeneity
print("\n📊 Test 6: ENDOGENEITY")
print(f"   Corr(X, residuals) - good model: {corr_resid_good:.4f} {'✅ No endogeneity' if abs(corr_resid_good) < 0.1 else '❌ Endogeneity!'}")
print(f"   Corr(X, residuals) - bad model:  {corr_resid_bad:.4f} {'✅ No endogeneity' if abs(corr_resid_bad) < 0.1 else '❌ Endogeneity!'}")
print("   Rule: Correlation should be ≈ 0")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: WHAT TO DO WHEN ASSUMPTIONS ARE VIOLATED")
print("="*70)
print("""
┌─────────────────────────┬──────────────────────────────────────────────┐
│ Assumption Violated     │ Solution                                     │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 1. Non-linearity        │ Polynomial features, log transform, or       │
│                         │ non-linear models (trees, neural nets)        │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 2. Heteroscedasticity   │ Log-transform Y, Weighted Least Squares,     │
│                         │ or robust standard errors                     │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 3. Non-normal errors    │ Transform Y (log, Box-Cox), remove outliers, │
│                         │ or use robust regression                      │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 4. Autocorrelation      │ Time-series models (ARIMA), add lagged       │
│                         │ variables, or Generalized Least Squares       │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 5. Multicollinearity    │ Ridge regression, remove features, PCA,      │
│                         │ or combine correlated features                │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 6. Endogeneity          │ Add omitted variables, instrumental          │
│                         │ variables, or two-stage least squares         │
└─────────────────────────┴──────────────────────────────────────────────┘
""")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. ALWAYS check assumptions before trusting your linear model
2. Residual plots are your best friend for diagnostics
3. Most violations have practical solutions
4. Ridge/Lasso handle multicollinearity automatically
5. Large samples make normality less critical (CLT)
6. If linearity is violated, no amount of regularization helps
   → you need a different model type

DIAGNOSTIC WORKFLOW:
1. Fit your model
2. Calculate residuals = actual - predicted
3. Plot residuals vs fitted values (checks 1, 2, 6)
4. Plot histogram/Q-Q of residuals (checks 3)
5. Plot residuals in order (checks 4)
6. Check correlation matrix / VIF (checks 5)
""")
