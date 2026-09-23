# Visual Foundations for Linear Models

Run the companion lesson from this folder:

```bash
python 06_visual_foundations.py
```

It writes seven deterministic PNG lessons to `visual_foundations/`. Read them in filename order.

| Figure | Learn visually | Core takeaway |
|---|---|---|
| 01 | OLS residuals, squared loss, negative R² | OLS minimizes squared residuals. A test-set R² can be below zero when predictions are worse than a mean-only baseline. |
| 02 | Holdout, k-fold CV, scaling leakage | Split first. Put preprocessing and the estimator in a `Pipeline` so every fold fits its scaler only on training rows. |
| 03 | Underfit / useful complexity / overfit | Select complexity with validation data, not training error alone. |
| 04 | L1 versus L2 constraint geometry | L1's corners make zero coefficients more likely; L2 typically shrinks rather than zeroes coefficients. |
| 05 | Ridge alpha, CV error, coefficient norm | Larger alpha increases shrinkage. Choose it with CV, with scaling inside the pipeline. |
| 06 | Sigmoid, probability surface, threshold, confusion matrix, ROC | Logistic regression models a probability; a class label follows only after selecting a threshold. |
| 07 | Robust regression and Poisson GLM | Outliers motivate robust estimators; count targets often call for a positive-mean count model rather than ordinary least squares. |

## Important corrections to the existing tutorials

- **Scale penalized features for comparable penalties.** Unscaled Ridge or Lasso is not intrinsically invalid, but one shared penalty treats coefficients in different units unequally. Fit the scaler on training data only, ideally through a `Pipeline`.
- **Lasso sparsity is not a feature-importance proof.** With correlated predictors, Lasso can retain one and zero another; selection can vary across samples. Treat it as a predictive-model property, then validate domain conclusions separately.
- **R² is not bounded below by zero.** `1.0` is perfect; a mean-only predictor scores `0.0`; a model can have negative test R².
- **Normality concerns residual inference, not OLS unbiasedness.** The core causal/identification condition is exogeneity (`E[error | X] = 0`). Normal residuals support exact small-sample inference; they are not needed for OLS to be unbiased or for the Gauss–Markov result.
- **`sklearn.linear_model.LogisticRegression` is L2-regularized by default.** It models `P(y=1|X)` through a sigmoid. The default 0.5 decision threshold is a convention, not an inevitable choice.
- **PCA can remove collinearity but changes the questions coefficients answer.** Components are mixtures of features and are unsupervised; use it deliberately when predictive performance outweighs original-feature interpretability.

## Coverage status and next steps

The original README claims 18+ models, but the extracted project contains substantive tutorials primarily for OLS, Ridge, Lasso, ElasticNet, diagnostics, and a small logistic example. It also contains empty stubs and references to non-existent tutorial files. This visual guide closes the most important conceptual gaps; it does not make those claimed advanced tutorials exist.

For a full linear-model curriculum, add focused, tested tutorials for: robust regression (RANSAC, Theil–Sen, Huber), generalized linear models (Poisson, Gamma, Tweedie), quantile regression, Bayesian Ridge/ARD, SGD models, LARS/OMP, multi-task models, and model-selection workflows. Add each only with a clear use case, a reproducible data example, an evaluation metric, and a caveat.

## Primary references

- [scikit-learn Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [`r2_score` API: negative scores and mean baseline](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- [scikit-learn underfitting vs. overfitting example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [An Introduction to Statistical Learning (free edition)](https://www.statlearning.com/)
