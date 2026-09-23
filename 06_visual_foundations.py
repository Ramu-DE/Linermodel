"""Generate visual foundations for the linear-models tutorial.

Run: python 06_visual_foundations.py
Outputs: visual_foundations/*.png

The examples are deterministic synthetic demonstrations. They teach concepts;
they are not model-selection advice for a particular real-world data set.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import auc, confusion_matrix, mean_squared_error, r2_score, roc_curve
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUTDIR = Path(__file__).resolve().parent / "visual_foundations"
RNG = np.random.default_rng(42)


def save(fig, name):
    """Save a figure consistently and release its resources."""
    OUTDIR.mkdir(exist_ok=True)
    path = OUTDIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"created {path.name}")


def plot_ols_and_r2():
    """Show residuals, squared loss, and why test R² can be negative."""
    x = np.linspace(0, 10, 32)
    y = 2 + 1.35 * x + RNG.normal(0, 1.8, x.size)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    fitted = model.predict(x.reshape(-1, 1))
    slopes = np.linspace(0.2, 2.5, 200)
    losses = [np.mean((y - (y.mean() + slope * (x - x.mean()))) ** 2) for slope in slopes]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(x, y, color="#377eb8", label="observations")
    axes[0].plot(x, fitted, color="#e41a1c", lw=2.5, label="OLS fit")
    for xi, yi, pred in zip(x[::4], y[::4], fitted[::4]):
        axes[0].vlines(xi, min(yi, pred), max(yi, pred), color="gray", alpha=.7)
    axes[0].set(title="OLS minimizes squared residuals", xlabel="x", ylabel="y")
    axes[0].legend()

    axes[1].plot(slopes, losses, color="#984ea3", lw=2.5)
    axes[1].axvline(model.coef_[0], color="#e41a1c", ls="--", label=f"best slope = {model.coef_[0]:.2f}")
    axes[1].set(title="Mean squared error loss", xlabel="candidate slope", ylabel="MSE")
    axes[1].legend()

    y_test = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    mean_pred = np.full_like(y_test, y_test.mean())
    bad_pred = np.array([8.0, 7.0, 9.0, 7.0, 8.0])
    positions = np.arange(y_test.size)
    axes[2].plot(positions, y_test, "o-", label="actual y", color="#377eb8")
    axes[2].plot(positions, mean_pred, "--", label="mean baseline: R² = 0", color="gray")
    axes[2].plot(positions, bad_pred, "s--", label=f"bad predictions: R² = {r2_score(y_test, bad_pred):.2f}", color="#e41a1c")
    axes[2].set(title="R² is allowed to be negative", xlabel="test observation", ylabel="target")
    axes[2].legend(fontsize=8)
    save(fig, "01_ols_residuals_loss_and_negative_r2.png")


def plot_split_cv_and_pipeline():
    """Visualize train/test holdout, k-fold validation, and leakage-safe scaling."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    n = 20
    colors = {"train": "#4daf4a", "test": "#e41a1c", "validation": "#377eb8"}
    axes[0].barh(0, 15, color=colors["train"], label="train: fit scaler + model")
    axes[0].barh(0, 5, left=15, color=colors["test"], label="test: use once at the end")
    axes[0].set(title="Holdout split: test data must stay untouched", yticks=[])
    axes[0].legend(ncol=2, loc="upper center")
    for fold in range(5):
        test_start = fold * 4
        axes[1].barh(fold, n, color=colors["train"])
        axes[1].barh(fold, 4, left=test_start, color=colors["validation"])
    axes[1].set(title="5-fold cross-validation: each block validates once", ylabel="fold")
    axes[1].set_yticks(range(5), [f"fold {i + 1}" for i in range(5)])
    axes[2].axis("off")
    axes[2].text(
        .02, .7,
        "Leakage-safe workflow\n\n"
        "Pipeline(StandardScaler(), Ridge())\n"
        "• inside each CV fold: scaler.fit(train fold) → scaler.transform(train/validation)\n"
        "• after selection: fit the whole pipeline on training data only\n"
        "• then evaluate once on held-out test data\n\n"
        "Do not fit StandardScaler on all rows before splitting.",
        fontsize=13, va="center", family="monospace",
        bbox={"boxstyle": "round,pad=.6", "facecolor": "#fff7d6", "edgecolor": "#b8860b"},
    )
    axes[2].set_xlim(0, n)
    save(fig, "02_train_test_cross_validation_and_no_leakage.png")


def plot_underfit_overfit():
    """Show polynomial model complexity versus train/test error."""
    x = np.linspace(-3, 3, 42)
    y = np.sin(x) + RNG.normal(0, .20, x.size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.35, random_state=7)
    grid = np.linspace(-3.2, 3.2, 500)
    degrees = [1, 5, 13]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, degree in zip(axes.flat[:3], degrees):
        model = Pipeline([("poly", PolynomialFeatures(degree)), ("linear", LinearRegression())])
        model.fit(x_train.reshape(-1, 1), y_train)
        ax.scatter(x_train, y_train, color="#377eb8", label="train")
        ax.scatter(x_test, y_test, color="#e41a1c", marker="x", label="test")
        ax.plot(grid, model.predict(grid.reshape(-1, 1)), color="#4daf4a", lw=2.5)
        label = {1: "underfit", 5: "useful complexity", 13: "overfit risk"}[degree]
        ax.set(title=f"degree {degree}: {label}", xlabel="x", ylabel="y")
        ax.legend(fontsize=8)

    all_degrees = range(1, 16)
    train_mse, test_mse = [], []
    for degree in all_degrees:
        model = Pipeline([("poly", PolynomialFeatures(degree)), ("linear", LinearRegression())])
        model.fit(x_train.reshape(-1, 1), y_train)
        train_mse.append(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1))))
        test_mse.append(mean_squared_error(y_test, model.predict(x_test.reshape(-1, 1))))
    axes[1, 1].plot(all_degrees, train_mse, "o-", label="training MSE", color="#377eb8")
    axes[1, 1].plot(all_degrees, test_mse, "o-", label="test MSE", color="#e41a1c")
    axes[1, 1].set(title="Choose complexity using unseen data", xlabel="polynomial degree", ylabel="MSE")
    axes[1, 1].legend()
    save(fig, "03_underfitting_overfitting_and_model_complexity.png")


def plot_l1_l2_geometry():
    """Show why L1 corners make exact zero coefficients more likely."""
    w1, w2 = np.meshgrid(np.linspace(-2.2, 2.2, 400), np.linspace(-2.2, 2.2, 400))
    loss = ((w1 - 1.25) / 1.25) ** 2 + ((w2 - .9) / .65) ** 2
    levels = [.2, .6, 1.2, 2.0, 3.2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    theta = np.linspace(0, 2 * np.pi, 500)
    diamond_x = np.array([0, 1, 0, -1, 0])
    diamond_y = np.array([1, 0, -1, 0, 1])
    for ax, penalty in zip(axes, ["L2 / Ridge", "L1 / Lasso"]):
        ax.contour(w1, w2, loss, levels=levels, colors="gray")
        if penalty.startswith("L2"):
            ax.plot(np.cos(theta), np.sin(theta), color="#377eb8", lw=3, label="L2 constraint: circle")
            annotation = "Smooth boundary: touch is usually not on an axis"
        else:
            ax.plot(diamond_x, diamond_y, color="#e41a1c", lw=3, label="L1 constraint: diamond")
            ax.scatter([1, 0, -1, 0], [0, 1, 0, -1], color="#e41a1c", zorder=3)
            annotation = "Corners lie on axes: exact zeros become more likely"
        ax.scatter([1.25], [.9], marker="*", s=180, color="#4daf4a", label="unconstrained OLS")
        ax.set(title=penalty, xlabel="coefficient w₁", ylabel="coefficient w₂", aspect="equal")
        ax.text(-2.05, -1.9, annotation, fontsize=9, bbox={"facecolor": "white", "alpha": .9})
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Same loss contours; different regularization geometry", fontsize=15)
    save(fig, "04_l1_vs_l2_regularization_geometry.png")


def plot_ridge_cv():
    """Show the bias-variance motivation for selecting alpha by CV in a pipeline."""
    x = RNG.normal(size=(110, 30))
    coefficients = np.zeros(30)
    coefficients[:5] = [2.3, -1.8, 1.2, .8, -.6]
    y = x @ coefficients + RNG.normal(0, 2.0, x.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=4)
    alphas = np.logspace(-3, 3, 20)
    cv = KFold(n_splits=5, shuffle=True, random_state=4)
    cv_mse, test_mse, norms = [], [], []
    for alpha in alphas:
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        cv_mse.append(-cross_val_score(pipe, x_train, y_train, cv=cv, scoring="neg_mean_squared_error").mean())
        pipe.fit(x_train, y_train)
        test_mse.append(mean_squared_error(y_test, pipe.predict(x_test)))
        norms.append(np.linalg.norm(pipe.named_steps["ridge"].coef_))
    best = int(np.argmin(cv_mse))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].semilogx(alphas, cv_mse, "o-", label="5-fold CV MSE", color="#377eb8")
    axes[0].semilogx(alphas, test_mse, "o--", label="held-out test MSE (illustration)", color="#e41a1c")
    axes[0].axvline(alphas[best], color="black", ls=":", label=f"CV choice α={alphas[best]:.3g}")
    axes[0].set(title="Regularization trades variance for bias", xlabel="Ridge alpha (log scale)", ylabel="MSE")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(alphas, norms, "o-", color="#984ea3")
    axes[1].set(title="Larger alpha shrinks coefficient norm", xlabel="Ridge alpha (log scale)", ylabel="||coefficient vector||₂")
    axes[1].text(.03, .12, "Scale inside the Pipeline\nso every CV fold fits its own scaler.", transform=axes[1].transAxes,
                 bbox={"facecolor": "#fff7d6", "edgecolor": "#b8860b"})
    save(fig, "05_ridge_bias_variance_cross_validation_pipeline.png")


def plot_logistic_and_metrics():
    """Teach sigmoid probabilities, thresholding, and two classification metrics."""
    x, y = make_classification(n_samples=260, n_features=2, n_redundant=0, n_informative=2,
                               class_sep=1.15, flip_y=.08, random_state=9)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.35, stratify=y, random_state=9)
    model = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(random_state=9))])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= .5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probability)
    matrix = confusion_matrix(y_test, prediction)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    z = np.linspace(-7, 7, 300)
    axes[0, 0].plot(z, 1 / (1 + np.exp(-z)), color="#377eb8", lw=3)
    axes[0, 0].axhline(.5, color="gray", ls="--")
    axes[0, 0].axvline(0, color="gray", ls="--")
    axes[0, 0].set(title="Sigmoid: linear score → P(class 1)", xlabel="linear score z", ylabel="probability")

    grid_x, grid_y = np.meshgrid(np.linspace(x[:, 0].min() - .5, x[:, 0].max() + .5, 250),
                                 np.linspace(x[:, 1].min() - .5, x[:, 1].max() + .5, 250))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_prob = model.predict_proba(grid)[:, 1].reshape(grid_x.shape)
    axes[0, 1].contourf(grid_x, grid_y, grid_prob, levels=20, cmap="RdBu", alpha=.6)
    axes[0, 1].contour(grid_x, grid_y, grid_prob, levels=[.5], colors="black", linewidths=2)
    axes[0, 1].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="bwr", edgecolor="black", s=28)
    axes[0, 1].set(title="Probability field; black line = 0.5 threshold", xlabel="feature 1", ylabel="feature 2")

    image = axes[1, 0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, matrix[i, j], ha="center", va="center", fontsize=14)
    axes[1, 0].set(title="Confusion matrix at threshold 0.5", xlabel="predicted class", ylabel="actual class",
                   xticks=[0, 1], yticks=[0, 1])
    fig.colorbar(image, ax=axes[1, 0], fraction=.046)

    axes[1, 1].plot(fpr, tpr, lw=2.5, label=f"ROC AUC = {auc(fpr, tpr):.2f}", color="#4daf4a")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="random")
    axes[1, 1].set(title="ROC curve: ranking across all thresholds", xlabel="false positive rate", ylabel="true positive rate")
    axes[1, 1].legend()
    fig.suptitle("LogisticRegression models probabilities; labels come from a chosen threshold", fontsize=14)
    save(fig, "06_logistic_probabilities_threshold_confusion_matrix_roc.png")


def plot_robust_and_poisson():
    """Compare OLS/Huber for outliers and linear/Poisson models for counts."""
    x = np.linspace(0, 10, 70)
    y = 2 + .9 * x + RNG.normal(0, .8, x.size)
    y[[8, 27, 51]] += [10, -11, 12]
    ols = LinearRegression().fit(x.reshape(-1, 1), y)
    huber = HuberRegressor().fit(x.reshape(-1, 1), y)

    count_x = np.linspace(0, 4, 100)
    count_y = RNG.poisson(np.exp(.35 + .42 * count_x))
    linear = LinearRegression().fit(count_x.reshape(-1, 1), count_y)
    poisson = PoissonRegressor(alpha=.01, max_iter=1000).fit(count_x.reshape(-1, 1), count_y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(x, y, color="#377eb8", label="data (three outliers marked)")
    axes[0].scatter(x[[8, 27, 51]], y[[8, 27, 51]], color="#e41a1c", s=65, label="outliers")
    axes[0].plot(x, ols.predict(x.reshape(-1, 1)), color="#e41a1c", lw=2.5, label="OLS")
    axes[0].plot(x, huber.predict(x.reshape(-1, 1)), color="#4daf4a", lw=2.5, label="Huber")
    axes[0].set(title="Robust regression reduces outlier influence", xlabel="x", ylabel="y")
    axes[0].legend(fontsize=8)

    axes[1].scatter(count_x, count_y, alpha=.55, color="#377eb8", label="count observations")
    axes[1].plot(count_x, linear.predict(count_x.reshape(-1, 1)), color="#e41a1c", lw=2.5, label="LinearRegression")
    axes[1].plot(count_x, poisson.predict(count_x.reshape(-1, 1)), color="#4daf4a", lw=2.5, label="PoissonRegressor (log link)")
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set(title="Counts: a Poisson GLM keeps mean predictions positive", xlabel="predictor", ylabel="event count")
    axes[1].legend(fontsize=8)
    save(fig, "07_robust_regression_and_poisson_glm.png")


def main():
    print(f"Writing visual foundations to: {OUTDIR}")
    plot_ols_and_r2()
    plot_split_cv_and_pipeline()
    plot_underfit_overfit()
    plot_l1_l2_geometry()
    plot_ridge_cv()
    plot_logistic_and_metrics()
    plot_robust_and_poisson()
    print("Done. Open the PNG files in visual_foundations/ in numeric order.")


if __name__ == "__main__":
    main()
