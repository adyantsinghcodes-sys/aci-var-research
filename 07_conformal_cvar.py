"""
Step 7: conformalized CVaR (Expected Shortfall).

CVaR (Expected Shortfall) is the average loss on the worst alpha% of
days -- an EXPECTATION, not a quantile like VaR. This means it doesn't
have a clean yes/no "breach" event, so Kupiec/Christoffersen-style
tests don't directly apply. Instead we use the standard descriptive
approach for CVaR backtesting: on days where VaR is breached, compare
the ACTUAL realized loss to the PREDICTED CVaR for that day.

  exceedance ratio = mean(actual loss on breach days) / mean(predicted CVaR on breach days)

  ratio ~= 1   -> CVaR estimate is well calibrated
  ratio > 1    -> CVaR underestimates how bad breach-day losses really are
  ratio < 1    -> CVaR overestimates (overly conservative)

We compute this ratio for:
  1. BASE CVaR (rolling historical, fixed alpha) -- same idea as base VaR
  2. CONFORMAL CVaR -- calibrated with a flat shift learned on the
     calibration period, same split-conformal logic as step 5, but
     applied to the CVaR target instead of the VaR target

...on both the full test period and the 2020 crisis window specifically.

"""

import numpy as np
import pandas as pd

from common import ROLLING_WINDOW, base_parser, fetch_returns

CONFIDENCE = 0.99
ALPHA = 1 - CONFIDENCE


def rolling_var_cvar(losses: pd.Series, window: int, confidence: float):
    """
    For each day, using the trailing `window` days:
      VaR  = the (confidence) quantile of losses
      CVaR = the mean of losses that are >= that quantile
    Returns two aligned Series.
    """
    var_list, cvar_list = [], []
    for i in range(len(losses)):
        if i < window:
            var_list.append(np.nan)
            cvar_list.append(np.nan)
            continue
        window_losses = losses.iloc[i - window:i]
        var_t = np.quantile(window_losses, confidence)
        tail = window_losses[window_losses >= var_t]
        cvar_t = tail.mean() if len(tail) > 0 else var_t
        var_list.append(var_t)
        cvar_list.append(cvar_t)
    return (pd.Series(var_list, index=losses.index, name="var"),
            pd.Series(cvar_list, index=losses.index, name="cvar"))


def exceedance_ratio(actual_losses_on_breach_days: pd.Series,
                     predicted_cvar_on_breach_days: pd.Series) -> float:
    """
    Core CVaR evaluation metric (descriptive, not a formal p-value test --
    see module docstring for why CVaR doesn't have a standard hypothesis
    test the way VaR does).
    """
    if len(actual_losses_on_breach_days) == 0:
        return np.nan
    return actual_losses_on_breach_days.mean() / predicted_cvar_on_breach_days.mean()


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--calib-end", default="2018-01-01")
    args = parser.parse_args()
    returns = fetch_returns(args.ticker, args.start, args.end)
    losses = -returns
    print(f"Got {len(returns)} daily return observations.")

    print("\nComputing rolling VaR/CVaR (this loops day by day, may take a few seconds)...")
    base_var, base_cvar = rolling_var_cvar(losses, ROLLING_WINDOW, CONFIDENCE)

    calib_mask = returns.index < pd.Timestamp(args.calib_end)
    test_mask = ~calib_mask

    # ---- Conformal calibration for CVaR, same logic as step 5 but on the CVaR target ----
    # nonconformity score = actual loss - predicted CVaR, computed ONLY on
    # calibration-period breach days (days where the loss actually exceeded VaR --
    # CVaR is only meant to describe THOSE days)
    calib_breach_mask = calib_mask & (losses > base_var)
    calib_scores = (losses - base_cvar)[calib_breach_mask].dropna()
    # NOTE: CVaR targets a MEAN, so the correct bias correction is the MEAN
    # of the calibration errors -- NOT a high quantile (e.g. 0.99) of them.
    # Taking a tail quantile of an already-small, already-extreme sample
    # (breach days only, here just ~30 points) is unstable and grabs close
    # to the single largest error in the sample, producing wild overshoot.
    q_hat = calib_scores.mean() if len(calib_scores) > 0 else 0.0
    print(f"\nCalibration breach days used: {len(calib_scores)}")
    print(f"Conformal CVaR shift q_hat = {q_hat:.4%}")

    conformal_cvar = base_cvar.copy()
    conformal_cvar[test_mask] = base_cvar[test_mask] + q_hat

    # ---- Evaluate on TEST period breach days ----
    test_breach_mask = test_mask & (losses > base_var)
    actual_on_breach = losses[test_breach_mask]
    base_cvar_on_breach = base_cvar[test_breach_mask]
    conformal_cvar_on_breach = conformal_cvar[test_breach_mask]

    print(f"\nTest period: {returns[test_mask].index[0].date()} to {returns[test_mask].index[-1].date()}")
    print(f"Breach days in test period: {test_breach_mask.sum()}")

    ratio_base = exceedance_ratio(actual_on_breach, base_cvar_on_breach)
    ratio_conformal = exceedance_ratio(actual_on_breach, conformal_cvar_on_breach)

    print("\n" + "=" * 60)
    print("EXCEEDANCE RATIO -- FULL TEST PERIOD BREACH DAYS")
    print("=" * 60)
    print(f"Mean actual loss on breach days : {actual_on_breach.mean():.4%}")
    print(f"Mean BASE CVaR on breach days    : {base_cvar_on_breach.mean():.4%}")
    print(f"Mean CONFORMAL CVaR on breach days: {conformal_cvar_on_breach.mean():.4%}")
    print(f"\nBase exceedance ratio     : {ratio_base:.3f}  (1.0 = perfectly calibrated)")
    print(f"Conformal exceedance ratio: {ratio_conformal:.3f}")
    if abs(ratio_conformal - 1) < abs(ratio_base - 1):
        print("=> Conformal calibration brought CVaR closer to well-calibrated (ratio nearer 1.0)")
    else:
        print("=> Conformal calibration did NOT improve CVaR calibration this time")

    # ---- Same evaluation restricted to the 2020 crisis window ----
    crisis_mask = (losses.index >= "2020-02-01") & (losses.index <= "2020-06-30")
    crisis_breach_mask = crisis_mask & test_breach_mask
    actual_crisis = losses[crisis_breach_mask]
    base_cvar_crisis = base_cvar[crisis_breach_mask]
    conformal_cvar_crisis = conformal_cvar[crisis_breach_mask]

    print("\n" + "=" * 60)
    print("EXCEEDANCE RATIO -- 2020 CRISIS WINDOW BREACH DAYS ONLY")
    print("=" * 60)
    print(f"Breach days in crisis window: {crisis_breach_mask.sum()}")
    if crisis_breach_mask.sum() > 0:
        ratio_base_crisis = exceedance_ratio(actual_crisis, base_cvar_crisis)
        ratio_conformal_crisis = exceedance_ratio(actual_crisis, conformal_cvar_crisis)
        print(f"Mean actual loss on crisis breach days : {actual_crisis.mean():.4%}")
        print(f"Mean BASE CVaR on crisis breach days    : {base_cvar_crisis.mean():.4%}")
        print(f"Mean CONFORMAL CVaR on crisis breach days: {conformal_cvar_crisis.mean():.4%}")
        print(f"\nBase exceedance ratio (crisis)     : {ratio_base_crisis:.3f}")
        print(f"Conformal exceedance ratio (crisis): {ratio_conformal_crisis:.3f}")
    else:
        print("No breach days in crisis window for this test split -- try a different --calib-end.")


if __name__ == "__main__":
    main()