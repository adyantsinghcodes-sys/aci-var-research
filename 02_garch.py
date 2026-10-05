"""Step 2: stationarity, volatility clustering, and GARCH(1,1) diagnostics.

GARCH is used here to SHOW volatility clustering and why rolling VaR lags.
It is not used as a VaR model anywhere in this repo.
"""

import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

from common import (CRISIS_WINDOWS, ROLLING_WINDOW, base_parser, fetch_returns,
                    find_breaches, plot_path, rolling_var)


def run_adf_test(returns: pd.Series) -> None:
    stat, p_value, *_ , crit, _ = adfuller(returns.dropna(), result_object=False)
    print("--- ADF test on returns (H0: unit root) ---")
    print(f"ADF statistic={stat:.4f}, p-value={p_value:.6f}, critical values={crit}")
    print("=> stationary in mean" if p_value < 0.05 else "=> NOT stationary")


def plot_acf_comparison(ticker: str, returns: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(returns, lags=40, ax=axes[0])
    axes[0].set_title("ACF: Raw Returns")
    plot_acf(returns ** 2, lags=40, ax=axes[1])
    axes[1].set_title("ACF: Squared Returns (volatility clustering)")
    fig.suptitle(f"Autocorrelation Comparison — {ticker}")
    fig.tight_layout()
    path = plot_path(ticker, "acf_comparison")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fit_garch(returns: pd.Series) -> pd.Series:
    # arch works better numerically with returns in percent.
    fitted = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
    print("\n--- GARCH(1,1) ---")
    print(fitted.summary())
    return fitted.conditional_volatility / 100


def plot_garch_vs_breaches(ticker, returns, cond_vol, confidence=0.95) -> None:
    breaches = find_breaches(returns, rolling_var(returns, ROLLING_WINDOW, confidence))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cond_vol.index, cond_vol.values, color="darkorange", linewidth=1,
            label="GARCH conditional volatility")
    for date in breaches[breaches].index:
        ax.axvline(date, color="red", alpha=0.15, linewidth=0.8)
    for start, end, _ in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="steelblue", alpha=0.1)
    ax.set_ylabel("GARCH volatility (daily)")
    ax.set_title(f"GARCH Conditional Volatility vs {int(confidence * 100)}% VaR Breaches — {ticker}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = plot_path(ticker, "garch_vs_breaches")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = base_parser(__doc__).parse_args()
    returns = fetch_returns(args.ticker, args.start, args.end)
    print(f"{args.ticker}: {len(returns)} daily returns\n")
    run_adf_test(returns)
    plot_acf_comparison(args.ticker, returns)
    cond_vol = fit_garch(returns)
    plot_garch_vs_breaches(args.ticker, returns, cond_vol)


if __name__ == "__main__":
    main()
