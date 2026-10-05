"""Step 4: split conformal prediction on synthetic heteroskedastic data.

Sanity check that the coverage guarantee holds before applying it to VaR.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import PLOTS_DIR

np.random.seed(42)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
N_TOTAL = 50000
ALPHA = 0.10          # target miscoverage rate -> we want 90% coverage
TRAIN_FRAC = 0.4
CALIB_FRAC = 0.3
# remaining fraction is the TEST set


def generate_synthetic_data(n: int):
    """
    y = x + noise, where noise variance grows with x (heteroskedastic).
    This is deliberately not simple constant-variance noise, to show
    conformal prediction doesn't need the noise to be "nice" or Gaussian.
    """
    x = np.random.uniform(-3, 3, size=n)
    noise_scale = 0.3 + 0.5 * np.abs(x)   # variance grows with |x|
    noise = np.random.normal(0, noise_scale)
    y = x + noise
    return x, y


def fit_simple_model(x_train, y_train):
    """
    Simple linear fit: y = a*x + b.
    The model doesn't need to be sophisticated -- conformal prediction
    wraps ANY model, even a deliberately simple/imperfect one, and still
    gives a valid coverage guarantee on the INTERVAL, even though the
    point prediction itself may be biased or miss the heteroskedasticity.
    """
    a, b = np.polyfit(x_train, y_train, deg=1)
    return lambda x: a * x + b


def split_conformal_calibrate(predict_fn, x_calib, y_calib, alpha: float) -> float:
    """
    Step 2 + 3: compute residuals on calibration set, take the
    ceil((1-alpha)(n+1))/n quantile (the standard finite-sample-correct
    version, slightly more conservative than a naive quantile).
    """
    residuals = np.abs(y_calib - predict_fn(x_calib))
    n = len(residuals)
    # finite-sample corrected quantile level
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)  # safety clamp
    q_hat = np.quantile(residuals, q_level)
    return q_hat


def build_intervals(predict_fn, x_test, q_hat: float):
    """Step 4: build [pred - q_hat, pred + q_hat] for each test point."""
    preds = predict_fn(x_test)
    lower = preds - q_hat
    upper = preds + q_hat
    return preds, lower, upper


def empirical_coverage(y_test, lower, upper) -> float:
    """Fraction of test points whose true y falls inside [lower, upper]."""
    covered = (y_test >= lower) & (y_test <= upper)
    return covered.mean()


def plot_results(x_test, y_test, preds, lower, upper, coverage: float, alpha: float):
    order = np.argsort(x_test)
    x_sorted = x_test[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_test, y_test, s=8, color="steelblue", alpha=0.5, label="Test points (true y)")
    ax.plot(x_sorted, preds[order], color="black", linewidth=1.5, label="Point prediction")
    ax.fill_between(x_sorted, lower[order], upper[order], color="orange", alpha=0.25,
                     label=f"Conformal interval (target {(1-alpha)*100:.0f}%)")

    ax.set_title(f"Toy Split Conformal Prediction\nEmpirical coverage: {coverage:.2%}  "
                 f"(target: {(1-alpha)*100:.0f}%)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "toy_conformal_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    x, y = generate_synthetic_data(N_TOTAL)

    n_train = int(N_TOTAL * TRAIN_FRAC)
    n_calib = int(N_TOTAL * CALIB_FRAC)

    x_train, y_train = x[:n_train], y[:n_train]
    x_calib, y_calib = x[n_train:n_train + n_calib], y[n_train:n_train + n_calib]
    x_test, y_test = x[n_train + n_calib:], y[n_train + n_calib:]

    print(f"Train: {len(x_train)}, Calibration: {len(x_calib)}, Test: {len(x_test)}")

    predict_fn = fit_simple_model(x_train, y_train)

    q_hat = split_conformal_calibrate(predict_fn, x_calib, y_calib, ALPHA)
    print(f"\nCalibrated q_hat (half-width of interval): {q_hat:.4f}")

    preds, lower, upper = build_intervals(predict_fn, x_test, q_hat)
    coverage = empirical_coverage(y_test, lower, upper)

    print(f"\nTarget coverage   : {(1-ALPHA)*100:.1f}%")
    print(f"Empirical coverage: {coverage*100:.2f}%  (on {len(x_test)} held-out test points)")

    if coverage >= (1 - ALPHA) - 0.02:
        print("=> Coverage guarantee HOLDS (within sampling noise of the target).")
    else:
        print("=> Coverage is noticeably below target -- check exchangeability assumptions.")

    plot_results(x_test, y_test, preds, lower, upper, coverage, ALPHA)


if __name__ == "__main__":
    main()