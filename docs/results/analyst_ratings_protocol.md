# Pre-registration: analyst rating changes and price-target revisions

Written 2026-10-31, after the rules below were approved and before any return
after an action (beyond day +1) was computed. The only results seen
beforehand are in [analyst_ratings_validation.md](analyst_ratings_validation.md):
the excess return vs SPY on days −2 to +1 around each action (used to check
the timestamps). The analysis is `scripts/analyst_ratings_strategy.py`;
results go to [analyst_ratings_strategy.md](analyst_ratings_strategy.md).
Nothing below may change after the first run; deviations, if any, must be
listed in the results.

## Approved rules

1. **Events:** Yahoo rating actions (`yfinance` `upgrades_downgrades`) on
   current S&P 500 members, from 2013-01-01.
2. **Timing:** day 0 is the first session whose close comes after the action
   (`first_reacting_session`: before the close → that day, after the close →
   the next session). Entry at the close of day +1.
3. **Hold:** 20 trading days (primary); 5 and 60 as sensitivity only.
4. **Signal 1 (rating changes):** long upgrades, short downgrades. Equal weight
   within each leg, 10 bps per unit traded.
5. **Signal 2 (price targets):** revision = new target / old target − 1, for
   actions with both targets positive. Long a raise in the top 20% of earlier
   raises by size; short a cut in the top 20% of earlier cuts by size.
6. **Earnings filter (secondary):** both signals rerun without actions whose
   day 0 is within 3 trading days (either side) of an earnings report.
7. **Holdout:** day 0 from 2025-11-01 on, untouched until the development
   results are final. Development: day 0 from 2013-01-01 to 2025-10-31.

## Details fixed at writing (not in the approved summary)

* **Engine:** `auto_researcher.backtest.event_strategy`, as in the
  earnings-drift study: daily calendar-time portfolios, `entry_lag = 1`,
  costs on the daily change in weights, dividend-adjusted closes (yfinance).
* **Membership:** an event counts only on or after the stock's S&P 500 add date
  (current constituents, Wikipedia). Survivorship bias remains: companies
  that left the index are missing.
* **Duplicates:** several brokers acting on the same stock with the same day 0
  and direction make one position. If the same stock gets both an upgrade and
  a downgrade (or a top raise and a top cut) on the same day 0, both are
  dropped.
* **"Top 20% of earlier":** the engine's trailing percentile: each raise is
  ranked among raises with day 0 in the previous 365 calendar days (strictly
  earlier, at least 200 of them), and likewise cuts by absolute size. No
  later event sets a threshold.
* **Earnings dates:** Alpha Vantage `reportedDate`, mapped to its first
  reacting session the same way (`reportTime` "post-market" → next session).
  Stocks with no Alpha Vantage history are kept unfiltered and counted.
* **Measures:** the long-short net daily return (primary), and each leg's net
  return minus SPY. Annualized mean, Sharpe, t-statistic of the daily mean,
  max drawdown; deflated P(SR > 0) over the 12 variants run
  (2 signals × 3 holds × 2 filters). Also by year.

## Decision rule

A signal is **supported** if, with the 20-day hold and no earnings filter,
its development-period long-short net return has t ≥ 2.0 **and** its
holdout mean is positive. The earnings filter, the other holds and the legs
are reported to explain the result, not to rescue it: a signal that fails
the primary test is not supported because a variant passes.
