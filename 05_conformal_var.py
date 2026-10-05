"""Step 5: static split-conformal VaR.

Learn one flat shift on the calibration period, apply it to the test period,
and compare against plain rolling VaR. Motivates ACI: one fixed shift cannot
react to a regime change.
"""

import numpy as np
import pandas as pd

from common import (ROLLING_WINDOW, base_parser, christoffersen_independence_test,
                    fetch_returns, find_breaches, kupiec_pof_test,
                    print_christoffersen, print_kupiec, rolling_var)

CONFIDENCE = 0.99
ALPHA = 1 - CONFIDENCE


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--calib-end", default="2018-01-01")
    args = parser.parse_args()

    returns = fetch_returns(args.ticker, args.start, args.end)
    losses = -returns
    base_var = rolling_var(returns, ROLLING_WINDOW, CONFIDENCE)
    calib = returns.index < pd.Timestamp(args.calib_end)

    scores = (losses - base_var)[calib].dropna()
    n = len(scores)
    q_level = min(np.ceil((1 - ALPHA) * (n + 1)) / n, 1.0)
    q_hat = np.quantile(scores, q_level)
    print(f"{args.ticker}: calibration {scores.index[0].date()}..{scores.index[-1].date()} ({n} obs)")
    print(f"Conformal shift q_hat = {q_hat:.4%}")

    test_returns = returns[~calib]
    base_breaches = find_breaches(test_returns, base_var[~calib])
    conformal_breaches = find_breaches(test_returns, base_var[~calib] + q_hat)
    print(f"Test period {test_returns.index[0].date()}..{test_returns.index[-1].date()} "
          f"({len(test_returns)} obs)\n")

    for label, b in [("BASE rolling VaR", base_breaches),
                     ("STATIC CONFORMAL VaR", conformal_breaches)]:
        print_kupiec(label, kupiec_pof_test(b, CONFIDENCE))
        print_christoffersen(label, christoffersen_independence_test(b))
        print()


if __name__ == "__main__":
    main()
