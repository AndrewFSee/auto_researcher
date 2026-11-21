# Pre-registration: customer momentum regression on 2010-2017

Written 2026-11-17, after the 2018-2026 test
([customer_momentum.md](customer_momentum.md)) and before any 10-K filed
before 2018 was downloaded or any return before 2017 was computed. The user
approved the decision rule below ("regression decides"). The analysis will be
`scripts/customer_momentum.py` run on the early sample; results go to
[customer_momentum_2010.md](customer_momentum_2010.md). Nothing below may change
after the first run; deviations, if any, must be listed in the results.

## Why this test

The pre-registered 2018-2026 test was not supported: the tercile long-short
made +0.13% a month net (Newey-West t 0.33). One secondary measure was
positive: the monthly cross-sectional regression of a supplier's next-month
return on its customers' return and its own month-t return (Fama-MacBeth) gave a
slope of +0.075, t 2.40 over 104 months (0.057, t 1.64, without the own-return
control). That was the best of seven secondary measures, chosen after seeing
them, so it is a hypothesis, not a finding. A diagnostic on 2018-2026, run after
the results and before this protocol, found the slope broad rather than driven
by a few months (positive in 61% of months, t 2.38 after dropping the five
largest months, positive in each year 2019-2025), suppliers' returns correlated
with their customers' (median +0.21) and a small own-return reversal (slope
-0.029, t -1.21), which the tercile sort on the raw signal does not separate.
Testing the regression again on 2018-2026 would only repeat it, so it is tested
on earlier years no return has been computed for.

## Hypothesis

In S&P 500 suppliers in 2010-2017, the customer-momentum slope in the monthly
Fama-MacBeth regression, controlling for the supplier's own month-t return, is
positive.

## Data

1. **Suppliers:** the companies (by CIK) in sentiq's
   `ref/universe/sp500_members.parquet`, which lists every S&P 500 member from
   2018 on (694 companies). Point-in-time membership before 2018 is not
   available, so the sample is companies that were in, or later joined, the
   index: survivorship is stated, not corrected.
2. **Filings:** each company's original 10-Ks filed from January 2009 to
   December 2017, downloaded from EDGAR (the SEC user agent from the project
   `.env`), primary document only, as sentiq's lake stores the 2018+ filings.
   Text is made the way sentiq makes it (its `html_to_text`, no truncation) and
   then the way `customer_links.load_10ks` reads it. One 10-K per company per
   calendar year, the first filed, as for 2018+. Exhibits are not read: older
   10-Ks often put the financial statements in Exhibit 13, so some customer
   notes will be missed; the count of filings with an Exhibit 13 is reported.
3. **Links:** the extraction and rules frozen at 52834aa, unchanged (qwen3:8b,
   prompt 97826f76; aliases, the SEPARATE table and the group-share rule as
   committed), text only. Inline-XBRL customer tags do not exist before 2019,
   so there are no XBRL links (2018-2019 filings were barely tagged either).
   Customers are matched with the current SEC ticker list, so customers that
   have since been acquired or delisted (Express Scripts, Medco, Sprint and
   the like) are not matched and drop out; the most common unmatched names are
   reported. Links to the filer itself and segment-only links are left out.
4. **Prices:** yfinance daily closes with `auto_adjust=True` (dividends and
   splits in the adjustment) under each company's current SEC ticker, since
   sentiq's price files start in August 2017. Month-ends are SPY's last session
   of each month from the same source. A stock whose prices end earns its return
   to the last close, then cash; delisted stocks yfinance lacks are absent.

## Rules carried over unchanged from the 2018-2026 protocol

Timing (links count from the first month-end after filing, replaced by the next
10-K or expire after 15 months), the signal (share-weighted customer return over
month t, "10% or more" weighs 10, the supplier's own ticker excluded), months
with fewer than 30 suppliers skipped, suppliers without a month t+1 return left
out, costs 10 bps per unit traded, Newey-West with 3 lags. The code is
`signals`, `links_at`, `portfolio` and `fama_macbeth` as committed in 206726b.

## Sample

Signal months January 2010 to December 2017; holding months February 2010 to
January 2018 (96). None overlaps the 2018-2026 test, whose first holding month
was February 2018.

## Primary measure and decision rule

Each holding month with at least 30 suppliers, an OLS across suppliers of the
month t+1 return on a constant, customer momentum and the supplier's own month-t
return. The customer-momentum coefficients are averaged over the months.

Customer momentum is **supported** if the mean coefficient is positive and its
Newey-West t-statistic (3 lags) is 2.0 or more. Nothing else decides.

## Secondary (to explain, not to rescue)

* The slope without the own-return control; the own-return slope.
* The tercile long-short on customer momentum (net and gross), as in 2018-2026.
* A residual sort: each month, customer momentum regressed across suppliers on
  a constant and the supplier's own month-t return; terciles on the residual,
  long-short net and gross, same costs. This is the tradeable form of the
  regression.
* The monthly rank correlation of customer momentum with the next month's return.
* The primary slope in 2010-2013 and 2014-2017 holding months separately.
* For information only, the primary slope pooled over 2010-2026 (both samples).
* Counts: filings and links per year, suppliers per month (min, median, max),
  distinct customers, the share of supplier-months depending on Walmart, price
  coverage, unmatched customer names.

A blind hand check of random links from 2009-2017 may be done at any time, to
measure extraction precision on older filings; it does not change the rules or
the decision.

## Power, stated before the run

If the true slope equals the 2018-2026 estimate (0.075) with the same monthly
volatility, the expected t over 96 months is about 2.3 and the test passes about
60% of the time. Because the 2018-2026 estimate was the best of seven, the true
slope is probably smaller and the chance of passing lower. A null result here
would not show that the effect is absent; a pass would show that customer
returns carry information about suppliers' next-month returns, not that a
portfolio earns it after costs (the residual sort reports that).
