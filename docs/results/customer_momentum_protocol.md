# Pre-registration: customer momentum (Cohen & Frazzini) in S&P 500 suppliers

Written 2026-11-17, after the rules below were approved and before any return
was computed for the suppliers or their customers. The only things seen
beforehand are the link counts and hand checks in
[customer_links_count.md](customer_links_count.md) and the per-year reports
(`data/customers/report_<year>.md`), and which tickers have price files. The
analysis will be `scripts/customer_momentum.py`; results go to
[customer_momentum.md](customer_momentum.md). Nothing below may change after
the first run; deviations, if any, must be listed in the results.

## Hypothesis

Cohen & Frazzini (2008, "Economic Links and Predictable Returns") found that a
stock's major customers' return in one month predicts the stock's return in the
next: investors are slow to price news that reaches a supplier through its
customers. They reported about 1.5% a month long-short in 1980-2004, strongest
in small, little-followed suppliers. This tests the effect out of sample, in
S&P 500 suppliers in 2018-2026.

## Approved rules

1. **Links:** supplier → customer links from the extraction rules frozen at
   commit 52834aa (listed customer, 10% or more of total revenue, for the year
   the 10-K reports), from 10-Ks filed from 2018 to October 2026, plus the
   inline-XBRL concentration tags in the same filings that name a listed
   company. The 2026 10-Ks are run with the same rules before any return is
   computed. Links to the filer itself and segment-only links are left out.
2. **Timing:** a 10-K's links count from the first month-end after its filing
   date. They are replaced when the supplier's next 10-K is filed (even if that
   one has no links), or expire 15 months after filing if none is.
3. **Signal:** at each month-end t, a supplier's customer momentum is its
   customers' total return over month t, weighted by each customer's stated
   share of revenue (a share given only as "10% or more" counts as 10). A
   customer without a return for month t is left out; a supplier with none
   has no signal.
4. **Portfolio:** each month-end, long the third of suppliers with the highest
   customer momentum and short the third with the lowest, equal weight, held
   for month t+1 at the suppliers' total returns. Costs 10 bps per unit
   traded, on the monthly change in weights. A month with fewer than 30
   suppliers with a signal is skipped.
5. **Prices:** daily closes and dividends from sentiq's price files (read
   only), and yfinance adjusted closes for tickers sentiq lacks. Before any
   signal is built, the two sources are compared on the tickers both cover. A
   stock that delists earns its return up to its last close, then cash.
6. **Primary measure:** the long-short monthly return net of costs, its mean,
   and its Newey-West t-statistic (3 lags), over the full sample. No holdout:
   the hypothesis and its parameters come from the paper, nothing is tuned,
   and a holdout would cut power that is already marginal.
7. **Secondary (to explain, not to rescue):** the gross return; quintiles in
   place of terciles; the portfolio without links to Walmart; a monthly
   cross-sectional regression of supplier return in t+1 on customer momentum
   and the supplier's own return in month t (Fama-MacBeth, Newey-West t of the
   customer-momentum coefficient), because suppliers and customers often share
   an industry; the monthly rank correlation of customer momentum with the
   next month's return; each leg's return minus SPY; 2018-2023 and 2024-2026
   reported separately.

## Details fixed at writing (not in the approved summary)

* **Month-ends:** the last NYSE session of each calendar month (SPY's trading
  days). Month t's return is the close-to-close return from the last session
  of month t-1 to the last session of month t, with each cash dividend added
  back on its ex-date: day return = (close + dividend) / previous close − 1.
* **Sample:** signal months from January 2018 (the first month-end after the
  first 2018 10-K) to September 2026, so the holding months run from February
  2018 to October 2026, the last full month with prices.
* **Price sources, in order:** sentiq `data/ref/prices` (Tradier), then
  `data/ref/prices_tiingo` (Tiingo, which keeps delisted stocks), then yfinance
  with `auto_adjust=True`. Renamed tickers are read under the name the source
  uses (AmerisourceBergen ABC → COR). The source check reports, for the tickers
  in both sentiq and yfinance, the share of monthly returns that differ by more
  than 0.5 percentage points; a ticker whose sources differ by more than that
  in over 10% of its months is listed, and sentiq's figures are still used.
* **A supplier's links:** one weight per customer. A pair found in both the text
  and the XBRL tags takes the text's share; an XBRL-only pair takes the tagged
  percentage. A customer named in two of the supplier's 10-Ks follows rule 2 (the
  newer filing replaces the older one). The supplier's own ticker never enters
  its signal.
* **Terciles:** suppliers sorted by customer momentum; with n suppliers the long
  leg is the top floor(n/3) and the short leg the bottom floor(n/3). Ties are
  broken by ticker. Quintiles in the same way with floor(n/5).
* **Suppliers without a return in month t+1** (delisted before the holding month
  starts) are left out of that month's sort.
* **Deflated Sharpe:** P(SR > 0) deflated over the 4 portfolio variants run
  (net, gross, quintiles, without Walmart).
* **Counts reported:** suppliers with a signal per month (min, median, max),
  distinct customers per month, the share of supplier-months whose signal
  depends on Walmart, and the price coverage of suppliers and customers.

## Power, stated before the run

About 63 suppliers and 46 distinct customers have a link in a typical month.
With terciles of about 21 stocks and roughly 105 holding months, the smallest
long-short mean the test detects reliably (80% power, two-sided 5%) is about
1% a month, under an assumed residual volatility of 3.5% a month. Customers are
concentrated (Walmart is about 18% of supplier-years, the three drug
wholesalers another 16%), so the effective number of independent signals is
lower than 46. A null result rules out an effect of the paper's size in these
suppliers, not a smaller one.

## Decision rule

Customer momentum is **supported** if the primary long-short net return has a
Newey-West t-statistic of 2.0 or more. The secondary measures are reported to
explain the result: an effect that fails the primary test is not supported
because a variant passes.
