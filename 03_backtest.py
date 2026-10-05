"""Step 3: backtest rolling historical VaR with Kupiec and Christoffersen,
on the full sample, crisis windows only, and calm periods only."""

from common import (ROLLING_WINDOW, base_parser, christoffersen_independence_test,
                    crisis_mask, fetch_returns, find_breaches, kupiec_pof_test,
                    print_christoffersen, print_kupiec, rolling_var)

CONFIDENCE_LEVELS = [0.95, 0.99]


def main():
    args = base_parser(__doc__).parse_args()
    returns = fetch_returns(args.ticker, args.start, args.end)
    print(f"{args.ticker}: {len(returns)} daily returns")

    for confidence in CONFIDENCE_LEVELS:
        breaches = find_breaches(returns, rolling_var(returns, ROLLING_WINDOW, confidence))
        in_crisis = crisis_mask(breaches.index)
        # Note: the crisis subsample joins 2008-09 and 2020 back to back, which adds
        # one artificial transition to the Christoffersen counts. Negligible here.
        for label, subset in [("FULL SAMPLE", breaches),
                              ("CRISIS WINDOWS ONLY", breaches[in_crisis]),
                              ("CALM PERIODS ONLY", breaches[~in_crisis])]:
            print(f"\n=== {int(confidence * 100)}% VaR — {label} ===")
            print_kupiec(label, kupiec_pof_test(subset, confidence))
            print_christoffersen(label, christoffersen_independence_test(subset))


if __name__ == "__main__":
    main()
