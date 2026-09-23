# Linear Models Implementation - Design Document

## Architecture Overview

### System Design
```
linear_models/
├── base.py              # Base classes and interfaces
├── linear.py            # LinearRegression
├── ridge.py             # Ridge Regression
├── lasso.py             # Lasso Regression
├── logistic.py          # Logistic Regression
├── metrics.py           # Evaluation metrics
├── utils.py             # Helper functions
└── visualizations.py    # Plotting utilities

tests/
├── test_linear.py
├── test_ridge.py
├── test_lasso.py
├── test_logistic.py
└── test_metrics.py

examples/
├── 01_linear_regression_demo.py
├── 02_ridge_demo.py
├── 03_lasso_demo.py
└── 04_logistic_demo.py
```

## Component Design

### 1. Base Classes

#### BaseRegressor
```python
class BaseRegressor:
    """Base class for all regression models"""
    
    def fit(self, X, y):
        """Fit model to training data"""
        pass
    
    def predict(self, X):
        """Make predictions on new data"""
        pass
    
    def score(self, X, y):
        """Calculate R² score"""
        pass
    
    def _validate_input(self, X, y=None):
        """Validate and preprocess input data"""
        pass
```

#### BaseClassifier
```python
class BaseClassifier:
    """Base class for all classification models"""
    
    def fit(self, X, y):
        """Fit model to training data"""
        pass
    
    def predict(self, X):
        """Predict class labels"""
        pass
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        pass
    
    def score(self, X, y):
        """Calculate accuracy score"""
        pass
```

### 2. Linear Regression

#### Algorithm: Normal Equation
```
Mathematical Formula:
w = (X^T X)^(-1) X^T y

Steps:
1. Add intercept column to X (column of ones)
2. Compute X^T X
3. Compute inverse (or use pseudo-inverse)
4. Compute X^T y
5. Multiply to get coefficients
```

#### Implementation Details
```python
class LinearRegression(BaseRegressor):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        
        # Normal equation: w = (X^T X)^(-1) X^T y
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Use pseudo-inverse for numerical stability
        w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0
            self.coef_ = w
        
        return self
```

### 3. Ridge Regression

#### Algorithm: Regularized Normal Equation
```
Mathematical Formula:
w = (X^T X + αI)^(-1) X^T y

Where:
- α = regularization parameter
- I = identity matrix

Steps:
1. Standardize features (important!)
2. Add αI to X^T X
3. Solve regularized system
4. Transform coefficients back to original scale
```

#### Implementation Details
```python
class Ridge(BaseRegressor):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._X_mean = None
        self._X_std = None
    
    def fit(self, X, y):
        # Center data
        if self.fit_intercept:
            self._X_mean = np.mean(X, axis=0)
            self._y_mean = np.mean(y)
            X = X - self._X_mean
            y = y - self._y_mean
        
        # Standardize for regularization
        self._X_std = np.std(X, axis=0)
        X = X / self._X_std
        
        # Ridge solution: (X^T X + αI)^(-1) X^T y
        n_features = X.shape[1]
        XtX = X.T @ X
        XtX += self.alpha * np.eye(n_features)
        Xty = X.T @ y
        
        self.coef_ = np.linalg.solve(XtX, Xty)
        
        # Transform back to original scale
        self.coef_ = self.coef_ / self._X_std
        
        if self.fit_intercept:
            self.intercept_ = self._y_mean - np.dot(self._X_mean, self.coef_)
        else:
            self.intercept_ = 0
        
        return self
```

### 4. Lasso Regression

#### Algorithm: Coordinate Descent
```
Coordinate Descent Algorithm:
For each iteration:
    For each feature j:
        1. Compute residual without feature j
        2. Compute correlation with residual
        3. Apply soft-thresholding:
           w_j = soft_threshold(correlation, α)
        4. Update residual
    Check convergence

Soft-Thresholding Function:
soft_threshold(z, α) = sign(z) * max(|z| - α, 0)
```

#### Implementation Details
```python
class Lasso(BaseRegressor):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, z, alpha):
        """Soft-thresholding operator"""
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Center data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X = X - X_mean
        y = y - y_mean
        
        # Normalize features
        X_std = np.std(X, axis=0)
        X = X / X_std
        
        # Initialize coefficients
        w = np.zeros(n_features)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            w_old = w.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y - X @ w + X[:, j] * w[j]
                
                # Compute correlation
                rho = X[:, j] @ residual / n_samples
                
                # Soft-thresholding
                w[j] = self._soft_threshold(rho, self.alpha)
            
            # Check convergence
            if np.max(np.abs(w - w_old)) < self.tol:
                break
        
        # Transform back
        self.coef_ = w / X_std
        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        
        return self
```

### 5. Logistic Regression

#### Algorithm: Gradient Descent
```
Sigmoid Function:
σ(z) = 1 / (1 + e^(-z))

Log-Loss (Binary Cross-Entropy):
L = -1/n Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

Gradient:
∇L = 1/n X^T (ŷ - y)

Update Rule:
w = w - learning_rate * ∇L
```

#### Implementation Details
```python
class LogisticRegression(BaseClassifier):
    def __init__(self, learning_rate=0.01, max_iter=1000, 
                 C=1.0, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C  # Inverse of regularization strength
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _log_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)
            
            # Compute gradients
            error = y_pred - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.mean(error)
            
            # Add L2 regularization
            if self.C > 0:
                grad_w += (1 / self.C) * self.coef_
            
            # Update weights
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
            
            # Check convergence
            if np.linalg.norm(grad_w) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)
    
    def predict(self, X):
        """Predict class labels"""
        return (self.predict_proba(X) >= 0.5).astype(int)
```

### 6. Metrics Module

```python
def r2_score(y_true, y_pred):
    """R² (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def accuracy_score(y_true, y_pred):
    """Classification accuracy"""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """2x2 confusion matrix for binary classification"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
```

### 7. Visualization Module

```python
def plot_regression_line(X, y, model, title="Regression"):
    """Plot data points and regression line"""
    plt.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Model')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_coefficient_path(alphas, coefs, title="Regularization Path"):
    """Plot how coefficients change with regularization"""
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f'Feature {i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """Plot classification decision boundary"""
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                edgecolors='black', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
```

## Data Flow

### Training Flow
```
1. Input: X (features), y (targets)
2. Validation: Check shapes, types, missing values
3. Preprocessing: Standardization, centering
4. Optimization: Solve for coefficients
5. Store: coef_, intercept_
6. Return: self (for method chaining)
```

### Prediction Flow
```
1. Input: X (features)
2. Validation: Check shape matches training
3. Preprocessing: Apply same transformations
4. Compute: y = X @ coef_ + intercept_
5. Return: predictions
```

## Testing Strategy

### Unit Tests
- Test each method independently
- Test with known solutions
- Test edge cases (empty, single sample, etc.)

### Integration Tests
- Compare with scikit-learn outputs
- Test on real datasets
- Verify numerical accuracy

### Property-Based Tests
- Test mathematical properties (e.g., R² ≤ 1)
- Test invariants (e.g., predictions shape)
- Test with random data

## Performance Considerations

### Optimization Techniques
1. Use NumPy vectorization (no Python loops)
2. Use `@` operator for matrix multiplication
3. Use `np.linalg.lstsq` for numerical stability
4. Clip values to prevent overflow

### Memory Management
- Avoid creating unnecessary copies
- Use in-place operations where possible
- Clear large intermediate arrays

## Error Handling

### Input Validation
- Check X and y shapes match
- Check for NaN/Inf values
- Check for sufficient samples
- Provide clear error messages

### Numerical Issues
- Handle singular matrices
- Prevent overflow in exponentials
- Use appropriate tolerances

## Documentation Standards

### Docstring Format
```python
def method(self, X, y):
    """
    Brief description.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    
    Returns
    -------
    self : object
        Returns self for method chaining
    
    Notes
    -----
    Mathematical formula and algorithm details
    
    Examples
    --------
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
```

## Correctness Properties

### Property 1: Perfect Fit on Training Data (OLS)
For OLS without regularization, predictions on training data should minimize MSE.

### Property 2: Coefficient Shrinkage (Ridge)
Ridge coefficients should have smaller magnitude than OLS coefficients.

### Property 3: Sparsity (Lasso)
Lasso should set some coefficients to exactly zero for sufficiently large alpha.

### Property 4: Probability Bounds (Logistic)
Predicted probabilities should always be between 0 and 1.

### Property 5: API Consistency
All models should follow the same fit/predict/score interface.
