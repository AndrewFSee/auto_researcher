# Round 3 pre-registration: fundamental factors and an ML valuation gap

Written 2026-10-01, before any factor return, IC or valuation-gap result was
computed. The only related result seen beforehand: across S&P 500 companies,
past and next 3-year revenue growth have an average rank correlation of +0.26
(near zero for 2021–2023 start years) and free-cash-flow growth −0.21.
The analysis is `scripts/fundamental_factor_research.py`; results go to
[fundamental_factors.md](fundamental_factors.md). Nothing below may change
after the first run; deviations, if any, must be listed in the results.

## Data

* Fundamentals: `data/research_cache/sec_fundamentals_pit.parquet` (SEC XBRL,
  first-reported values, usable the day after filing; income and cash-flow
  items trailing twelve months). See
  [sec_fundamentals_validation.md](sec_fundamentals_validation.md).
* Market cap: split-adjusted (not dividend-adjusted) close × split-adjusted
  shares. Returns: dividend-adjusted close (yfinance).
* Universe: current S&P 500 constituents, each from its index add date;
  excluded: GICS Financials, captive-finance industrials (F, GM, CAT, DE,
  PCAR) and companies whose SEC share count fails the Yahoo cross-check.
  Survivorship bias remains (companies that left the index are missing).

## Part A: six published factors

| Factor | Definition | Expected sign | Source |
| --- | --- | --- | --- |
| `gross_profitability` | (revenue − cost of revenue, or gross profit) / total assets | + | Novy-Marx (2013) |
| `accruals` | (net income − operating cash flow) / total assets | − | Sloan (1996) |
| `asset_growth` | total assets / total assets known one year earlier − 1 | − | Cooper, Gulen & Schill (2008) |
| `net_issuance` | log(split-adjusted shares / shares known one year earlier) | − | Pontiff & Woodgate (2008) |
| `fcf_yield` | (operating cash flow − capex) / market cap | + | value literature |
| `ebit_ev` | operating income / enterprise value (EV > 0) | + | value literature |
| `composite` | mean of the six signed cross-sectional percentile ranks | + | pre-specified combination |

* Rows: dates every 21 trading days from 2011; a row is kept only if all six
  factors exist, so every candidate is scored on the same (date, stock) rows.
* Evaluation: the purged walk-forward harness with a fixed-sign single-feature
  score (no fitting), execution one day after the signal date, horizons of 21
  trading days (primary) and 63 (secondary), rebalancing every 21 days, top-50
  equal-weight portfolio, 10 bps costs, compared with SPY and the equal-weight
  universe.
* Periods: development = signal dates through 2022-12-31; holdout = from
  2023-01-01. There is no selection step: every candidate is run once in
  each period.
* A factor is **supported** if its development-period mean IC (21-day) has the
  expected sign with Newey-West |t| ≥ 2.0 **and** its holdout mean IC has the
  same sign.

## Part B: ML growth forecast and valuation gap

**Forecast.** Training rows at quarter-ends from June 2012. Features known at
date *t*: revenue growth over 1 and 3 years, FCF-to-firm margin and its 3-year
change, operating and gross margin, capex/revenue, capex/depreciation, stock
compensation/revenue, log revenue, asset turnover, accruals, asset growth,
sector (categorical) and the sector median of 3-year revenue growth. Targets:
3-year forward revenue CAGR (clipped to −50%…+100%) and FCF-to-firm margin
three years ahead (clipped to −50%…+80%), both from values known at *t* + 3
years.

* Model: scikit-learn `HistGradientBoostingRegressor(max_iter=300,
  learning_rate=0.05, max_depth=3, min_samples_leaf=50, l2_regularization=1.0,
  random_state=0)`, one per target. No tuning.
* Purging: refit every 1 January from 2016 on all rows whose labels were known
  before the refit date (*t* + 3 years < refit date); at least 1,000 rows.
* Baselines: growth B1 = sector median of past 3-year growth; B2 = half own
  past growth + half sector median. Margin M1 = current margin; M2 = half
  current + half sector median.
* The forecast is **useful** if, out of sample, its mean per-date rank
  correlation with realized 3-year revenue growth beats B2's with a paired
  |t| ≥ 2.0.

**Valuation.** Intrinsic enterprise value: revenue grows at the forecast rate
for three years, then the rate fades linearly to 2.5% by year 10; FCF-to-firm
margin moves linearly from today's to the forecast margin over three years,
then stays; 9% discount rate; 2.5% terminal growth. Signal = intrinsic EV /
market EV (EV > 0).

**Candidates** (scored like Part A on identical rows: Part A rows from 2016
with positive trailing FCF-to-firm and both gaps defined):

| Candidate | Sign |
| --- | --- |
| `valuation_gap_ml`: intrinsic EV from the ML forecasts / EV | + |
| `valuation_gap_naive`: intrinsic EV from B2 growth and M2 margin / EV | + |
| `reverse_dcf`: implied growth from the reverse DCF. With positive FCF it ranks stocks exactly like FCF-to-firm / EV, so it is scored as `fcff_ev` | − implied growth (= + yield) |

The ML gap **adds value** if, in development, its mean IC (21-day) exceeds both
`valuation_gap_naive` and `reverse_dcf` with paired per-date |t| ≥ 2.0 against
each, and its holdout mean IC has the same sign and is at least as large as
both.

## Multiple testing

Deflated statistics assume 20 trials: (7 Part A + 3 Part B candidates) × 2
horizons.
