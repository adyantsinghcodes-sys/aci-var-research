"""Shared helpers: data loading, rolling VaR, breach detection, backtests.

Every script imports from here so the VaR definition and the tests are
defined exactly once.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.special import xlogy
from scipy.stats import chi2

DEFAULT_START = "2005-01-01"
DEFAULT_END = "2026-01-01"
ROLLING_WINDOW = 252
CRISIS_WINDOWS = [
    ("2008-01-01", "2009-06-30", "2008 GFC"),
    ("2020-02-01", "2020-06-30", "2020 COVID"),
]
PLOTS_DIR = Path(__file__).parent / "plots"


# ---------------------------------------------------------------- CLI / IO

def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ticker", default="^NSEI", help="e.g. ^NSEI (Nifty 50) or ^GSPC (S&P 500)")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    return parser


def plot_path(ticker: str, name: str) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    return PLOTS_DIR / f"{ticker.replace('^', '')}_{name}.png"


def fetch_returns(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")
    # yfinance can return multi-level columns for a single ticker; squeeze to 1-D.
    prices = data["Close"].squeeze()
    returns = prices.pct_change().dropna()
    returns.name = "return"
    return returns


def crisis_mask(index: pd.DatetimeIndex) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for start, end, _ in CRISIS_WINDOWS:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask


# ---------------------------------------------------------------- VaR

def rolling_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    """Historical VaR for day t, using ONLY days t-window .. t-1.

    The .shift(1) is essential: without it the window includes day t's own
    loss, which is then tested against itself (look-ahead bias that
    undercounts breaches).
    """
    losses = -returns
    var = losses.rolling(window=window).quantile(confidence).shift(1)
    var.name = f"VaR_{int(round(confidence * 100))}"
    return var


def find_breaches(returns: pd.Series, var: pd.Series) -> pd.Series:
    """Boolean breach series, restricted to days where VaR is defined.

    Warm-up days (VaR = NaN) are dropped, not counted as non-breaches;
    counting them would dilute the breach rate.
    """
    valid = var.notna()
    breaches = (-returns[valid]) > var[valid]
    breaches.name = "breach"
    return breaches


# ---------------------------------------------------------------- tests

def kupiec_pof_test(breaches: pd.Series, confidence: float) -> dict:
    """Kupiec proportion-of-failures test. H0: breach rate == 1 - confidence."""
    b = pd.Series(breaches).dropna().astype(int)
    n, x = len(b), int(b.sum())
    p = 1 - confidence
    if n == 0:
        return {"n": 0, "x": 0, "reject_H0": None, "note": "no observations"}
    pi_hat = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - pi_hat) + xlogy(x, pi_hat)
    lr = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(lr, df=1)
    return {"n": n, "x": x, "expected_rate": p, "observed_rate": pi_hat,
            "LR_stat": lr, "p_value": p_value, "reject_H0": p_value < 0.05}


def christoffersen_independence_test(breaches: pd.Series) -> dict:
    """Christoffersen independence test. H0: breaches are not clustered.

    Uses xlogy so zero transition counts (common at 99%, where n11 is often 0)
    are handled instead of skipped.
    """
    b = pd.Series(breaches).dropna().astype(int).to_numpy()
    if len(b) < 2:
        return {"reject_H0": None, "note": "fewer than 2 observations"}
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return {"n01": n01, "n11": n11, "reject_H0": None,
                "note": "no breaches (or only breaches) -- test undefined"}

    pi01, pi11 = n01 / n0, n11 / n1
    pi = (n01 + n11) / (n0 + n1)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(lr, df=1)
    return {"n01": n01, "n11": n11, "pi01": pi01, "pi11": pi11,
            "LR_stat": lr, "p_value": p_value, "reject_H0": p_value < 0.05}


def print_kupiec(label: str, r: dict) -> None:
    print(f"-- Kupiec POF: {label} --")
    if r.get("reject_H0") is None:
        print(f"  {r.get('note')}")
        return
    print(f"  n={r['n']}, breaches={r['x']}, expected={r['expected_rate']:.2%}, "
          f"observed={r['observed_rate']:.2%}")
    print(f"  LR={r['LR_stat']:.3f}, p={r['p_value']:.4f} => "
          f"{'REJECT (rate is wrong)' if r['reject_H0'] else 'fail to reject (rate OK)'}")


def print_christoffersen(label: str, r: dict) -> None:
    print(f"-- Christoffersen independence: {label} --")
    if r.get("reject_H0") is None:
        print(f"  {r.get('note')}")
        return
    print(f"  P(breach | breach yesterday)={r['pi11']:.2%} vs "
          f"P(breach | no breach yesterday)={r['pi01']:.2%}")
    print(f"  LR={r['LR_stat']:.3f}, p={r['p_value']:.4f} => "
          f"{'REJECT (breaches cluster)' if r['reject_H0'] else 'fail to reject (independent)'}")
