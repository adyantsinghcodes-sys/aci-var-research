"""Step 1: full-sample historical VaR/CVaR and rolling VaR breaches."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CRISIS_WINDOWS, ROLLING_WINDOW, base_parser, fetch_returns,
                    find_breaches, plot_path, rolling_var)

CONFIDENCE_LEVELS = [0.95, 0.99]


def historical_var_cvar(returns: pd.Series, confidence: float) -> tuple[float, float]:
    losses = -returns
    var = np.quantile(losses, confidence)
    return var, losses[losses >= var].mean()


def plot_breaches(ticker, returns, var, breaches, confidence):
    losses = -returns
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(losses.index, losses.values, color="steelblue", linewidth=0.6, label="Daily loss")
    ax.plot(var.index, var.values, color="black", linewidth=1.2,
            label=f"Rolling VaR ({int(confidence * 100)}%)")
    dates = breaches[breaches].index
    ax.scatter(dates, losses.loc[dates], color="red", s=12, zorder=5, label="Breach")
    for start, end, _ in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="orange", alpha=0.15)
    ax.set_title(f"Rolling {int(confidence * 100)}% VaR vs Daily Losses — {ticker}")
    ax.set_ylabel("Loss (fraction of price, negative = gain)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = plot_path(ticker, f"rolling_var_{int(confidence * 100)}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = base_parser(__doc__).parse_args()
    returns = fetch_returns(args.ticker, args.start, args.end)
    print(f"{args.ticker}: {len(returns)} daily returns\n")
    print(returns.describe())

    for confidence in CONFIDENCE_LEVELS:
        var, cvar = historical_var_cvar(returns, confidence)
        print(f"\n--- {int(confidence * 100)}% ---")
        print(f"Full-sample VaR : {var:.4%}")
        print(f"Full-sample CVaR: {cvar:.4%}")

        rvar = rolling_var(returns, ROLLING_WINDOW, confidence)
        breaches = find_breaches(returns, rvar)
        n, x = len(breaches), int(breaches.sum())
        print(f"Rolling VaR breaches: {x}/{n} = {x / n:.2%} (expected {1 - confidence:.2%})")
        plot_breaches(args.ticker, returns, rvar, breaches, confidence)


if __name__ == "__main__":
    main()
