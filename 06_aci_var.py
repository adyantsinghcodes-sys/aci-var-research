"""Step 6: Adaptive Conformal Inference (Gibbs & Candes, 2021) for VaR.

Each day, VaR is the empirical (1 - alpha_t) quantile of the trailing window
of losses (today excluded). After seeing whether today breached it:

    alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), bounds)

A breach (err_t = 1) lowers alpha_t, so tomorrow's VaR is more conservative.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ROLLING_WINDOW, base_parser, christoffersen_independence_test,
                    crisis_mask, fetch_returns, kupiec_pof_test,
                    print_christoffersen, print_kupiec, plot_path)

CONFIDENCE = 0.99
ALPHA = 1 - CONFIDENCE


def run_aci(returns, window, target_alpha, gamma, alpha_bounds, calib_end) -> pd.DataFrame:
    losses = -returns
    start = max(returns.index.searchsorted(pd.Timestamp(calib_end)), window)
    alpha_t = target_alpha
    records = []
    for i in range(start, len(returns)):
        window_losses = losses.iloc[i - window:i]          # excludes today
        base_var = np.quantile(window_losses, 1 - target_alpha)
        aci_var = np.quantile(window_losses, 1 - alpha_t)
        loss = losses.iloc[i]
        aci_breach = loss > aci_var
        records.append({"date": returns.index[i], "loss": loss, "base_var": base_var,
                        "aci_var": aci_var, "alpha_t": alpha_t,
                        "base_breach": loss > base_var, "aci_breach": aci_breach})
        alpha_t = float(np.clip(alpha_t + gamma * (target_alpha - int(aci_breach)), *alpha_bounds))
    return pd.DataFrame(records).set_index("date")


def plot_aci(ticker: str, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(df.index, df["loss"], color="steelblue", linewidth=0.6, label="Daily loss")
    axes[0].plot(df.index, df["base_var"], color="black", linewidth=1, label="Base VaR (fixed alpha)")
    axes[0].plot(df.index, df["aci_var"], color="red", linewidth=1.2, label="ACI VaR (adaptive alpha)")
    axes[0].set_title(f"Base VaR vs ACI VaR — {ticker} ({int(CONFIDENCE * 100)}%)")
    axes[0].legend(loc="upper left")
    axes[1].plot(df.index, df["alpha_t"], color="purple", linewidth=1)
    axes[1].axhline(ALPHA, color="gray", linestyle="--", linewidth=1, label="target alpha")
    axes[1].set_title("ACI alpha_t over time (lower = more conservative VaR)")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30"), color="orange", alpha=0.15)
    fig.tight_layout()
    path = plot_path(ticker, "aci_vs_base")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--calib-end", default="2018-01-01", help="test period starts here")
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--bounds", type=float, nargs=2, default=[0.001, 0.03],
                        metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    returns = fetch_returns(args.ticker, args.start, args.end)
    df = run_aci(returns, ROLLING_WINDOW, ALPHA, args.gamma, tuple(args.bounds), args.calib_end)
    print(f"{args.ticker}: test {df.index[0].date()}..{df.index[-1].date()} ({len(df)} obs), "
          f"gamma={args.gamma}, bounds={tuple(args.bounds)}\n")

    for label, col in [("BASE (fixed alpha)", "base_breach"), ("ACI (adaptive alpha)", "aci_breach")]:
        print_kupiec(label, kupiec_pof_test(df[col], CONFIDENCE))
        print_christoffersen(label, christoffersen_independence_test(df[col]))
        print()

    crisis = crisis_mask(df.index)
    print(f"{'':22s}{'Base':>10s}{'ACI':>10s}")
    for label, mask in [("Breach rate, crisis", crisis), ("Breach rate, calm", ~crisis)]:
        print(f"{label:22s}{df.loc[mask, 'base_breach'].mean():>10.2%}"
              f"{df.loc[mask, 'aci_breach'].mean():>10.2%}")
    for label, mask in [("Mean VaR, full", np.ones(len(df), bool)),
                        ("Mean VaR, calm", ~crisis), ("Mean VaR, crisis", crisis)]:
        print(f"{label:22s}{df.loc[mask, 'base_var'].mean():>10.2%}"
              f"{df.loc[mask, 'aci_var'].mean():>10.2%}")
    print(f"\nDays with alpha_t at lower bound: {(df['alpha_t'] <= args.bounds[0] + 1e-12).mean():.1%} "
          f"(at the floor, VaR is ~ the window maximum)")

    plot_aci(args.ticker, df)


if __name__ == "__main__":
    main()
