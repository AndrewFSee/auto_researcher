# Moat tagger v3: set `fresh`

Generated 2026-11-07 14:04 by `scripts/moat_pilot.py`, set `fresh`, model `claude-sonnet-5-5`, prompts `claude-sonnet-5-5__ca6bf1d1a__high`. One call per company over Item 1 (business) and five years of financials, prompt v3: quotes checked word for word against the text, each moat source rated from the verified quotes only (`auto_researcher.llm.moat`).

* Companies: 50 (50 rated); chunks read 50 (Item 1 cut at 16 chunks for 0).
* Quotes returned: 395; kept 394; not found in the text 0 (0.0%); duplicates 1; over the cap 0; unparseable replies 0.
* Kept quotes by kind: intangibles 82, switching_costs 60, network_effects 12, cost_advantage 41, efficient_scale 21, against 178.
* Rating fixes (citations of missing or "against" quotes dropped, uncited ratings lowered to 0): 0.
* Model time: 6 min (8 s per company); API cost $2.06; served by another model 0.

## Ratings

0 = none or generic, 1 = specific but modest, 2 = strong and durable.

| Ticker | Sector | Overall | intang | switch | network | cost | eff.scale | Against quotes | Returns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IPG | Communication Services | none | 0 | 0 | 0 | 0 | 0 | 5 | above |
| T | Communication Services | none | 1 | 0 | 0 | 1 | 1 | 4 | below |
| AMZN | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 4 | above |
| AZO | Consumer Discretionary | narrow | 1 | 0 | 0 | 1 | 0 | 3 | above |
| DECK | Consumer Discretionary | narrow | 1 | 0 | 0 | 0 | 0 | 4 | above |
| HLT | Consumer Discretionary | narrow | 1 | 1 | 1 | 0 | 0 | 3 | above |
| KMX | Consumer Discretionary | none | 0 | 0 | 0 | 1 | 0 | 5 | below |
| COST | Consumer Staples | wide | 1 | 1 | 0 | 2 | 0 | 2 | above |
| K | Consumer Staples | narrow | 1 | 0 | 0 | 0 | 0 | 4 | above |
| KMB | Consumer Staples | narrow | 1 | 0 | 0 | 0 | 0 | 3 | above |
| PG | Consumer Staples | narrow | 1 | 0 | 0 | 0 | 0 | 4 | above |
| SLB | Energy | none | 1 | 0 | 0 | 0 | 0 | 4 | near |
| AJG | Financials | none | 0 | 0 | 0 | 0 | 0 | 3 | near |
| BK | Financials | none | 0 | 0 | 0 | 0 | 0 | 3 | near |
| BX | Financials | narrow | 1 | 1 | 0 | 0 | 0 | 5 | above |
| GS | Financials | none | 1 | 0 | 0 | 0 | 0 | 5 | near |
| MET | Financials | none | 1 | 0 | 0 | 0 | 0 | 4 | near |
| RF | Financials | none | 0 | 0 | 0 | 0 | 0 | 5 | near |
| WFC | Financials | none | 0 | 0 | 0 | 0 | 0 | 4 | near |
| BSX | Health Care | none | 1 | 1 | 0 | 0 | 0 | 4 | below |
| CVS | Health Care | none | 0 | 0 | 0 | 1 | 0 | 5 | near |
| GEHC | Health Care | narrow | 0 | 1 | 0 | 0 | 0 | 3 | near |
| MRNA | Health Care | none | 1 | 0 | 0 | 0 | 0 | 6 | below |
| RMD | Health Care | none | 0 | 0 | 0 | 0 | 0 | 0 | above |
| WST | Health Care | narrow | 1 | 1 | 0 | 0 | 0 | 5 | above |
| CPRT | Industrials | narrow | 1 | 1 | 1 | 0 | 0 | 3 | above |
| DOV | Industrials | narrow | 0 | 1 | 0 | 0 | 0 | 3 | above |
| EMR | Industrials | narrow | 1 | 0 | 0 | 0 | 0 | 4 | unclear |
| FTV | Industrials | none | 0 | 0 | 0 | 0 | 0 | 4 | below |
| MMM | Industrials | narrow | 1 | 0 | 0 | 0 | 0 | 2 | unclear |
| TDG | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 2 | above |
| TXT | Industrials | none | 1 | 1 | 0 | 0 | 0 | 2 | near |
| VRSK | Industrials | wide | 2 | 2 | 1 | 0 | 0 | 2 | above |
| AKAM | Information Technology | none | 0 | 0 | 0 | 0 | 0 | 4 | near |
| CTSH | Information Technology | none | 0 | 1 | 0 | 0 | 0 | 4 | above |
| FTNT | Information Technology | narrow | 0 | 0 | 0 | 1 | 0 | 4 | above |
| IBM | Information Technology | none | 0 | 1 | 0 | 0 | 0 | 4 | unclear |
| PTC | Information Technology | narrow | 0 | 1 | 0 | 0 | 0 | 2 | above |
| SMCI | Information Technology | none | 0 | 0 | 0 | 0 | 0 | 5 | above |
| TRMB | Information Technology | none | 1 | 1 | 1 | 0 | 0 | 3 | below |
| TYL | Information Technology | none | 0 | 1 | 0 | 0 | 0 | 5 | below |
| ECL | Materials | narrow | 1 | 1 | 0 | 0 | 1 | 2 | near |
| NUE | Materials | none | 1 | 0 | 0 | 1 | 0 | 5 | near |
| SW | Materials | none | 0 | 1 | 0 | 1 | 0 | 4 | below |
| CSGP | Real Estate | narrow | 1 | 1 | 1 | 0 | 0 | 4 | near |
| MAA | Real Estate | none | 0 | 0 | 0 | 0 | 0 | 4 | near |
| PSA | Real Estate | narrow | 1 | 0 | 0 | 1 | 0 | 5 | above |
| AEP | Utilities | narrow | 1 | 0 | 0 | 0 | 2 | 2 | near |
| FE | Utilities | narrow | 0 | 0 | 0 | 0 | 1 | 1 | near |
| WEC | Utilities | narrow | 1 | 0 | 0 | 0 | 1 | 1 | near |

| Sector | none | narrow | wide |
| --- | ---: | ---: | ---: |
| Communication Services | 2 | 0 | 0 |
| Consumer Discretionary | 2 | 3 | 0 |
| Consumer Staples | 0 | 3 | 1 |
| Energy | 1 | 0 | 0 |
| Financials | 6 | 1 | 0 |
| Health Care | 4 | 2 | 0 |
| Industrials | 2 | 5 | 1 |
| Information Technology | 6 | 2 | 0 |
| Materials | 2 | 1 | 0 |
| Real Estate | 1 | 2 | 0 |
| Utilities | 0 | 3 | 0 |

## Notes

* **What changed from v1.** The prompt adds the company's financial history and a step judging returns on capital before the overall call (`ONE_CALL_SYSTEM_V2`); sources are still rated from quotes only. Item 1 texts that start mid-sentence or hold MD&A or financial statement headings early are not tagged (`item1_problem`).
* **Tuned on the hand-checked set, then frozen.** Against the targets the 47 v1 verdicts imply, v2 (fixed rules: high returns plus a source rated 1 meant narrow, one 8-10% cost of capital) put 34 overall calls on target and moved weak resellers up and utilities down; v3 (returns corroborate the text, a lower cost of capital for utilities and REITs) put 39. A second run of the unchanged v1 prompt put 36, and 4 of 47 calls changed between the two identical v1 runs, so v3's lead is small. v3 was committed (bfcd03b) before the fresh set was tagged once.
* **The fresh set** is 50 companies in no earlier set, drawn the same way (sector-stratified, seed 1) from the 2025 10-Ks that pass the Item 1 check; its tags were checked blind (`moat_v3_fresh.md`).
* **The financial history is point-in-time:** SEC XBRL company facts filed on or before the 10-K's filing date (the 10-K's own figures included), the latest value known then for each fiscal year (`data.sec_fundamentals.build_pit_table`). ROIC is operating income after a 21% tax over average equity + debt - cash; ratios on negative or near-zero capital are shown as n/a.

## Blind check

50 companies marked on a review page that hid the name, ticker and sector and shuffled the order, showing what the tagger saw (marks in `data/moat/blind_verdicts.json`). After marking a company the reviewer revealed it and could change any mark. AMZN, IPG, T had been marked on the named page first, so they are left out of the blind rows.

| Companies | Overall agreed | Too high | Too low | Source ratings agreed | Too high | Too low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all (50) | 42 (84%) | 0 | 8 | 234 of 250 (94%) | 0 | 16 |
| marked blind (47) | 40 (85%) | 0 | 7 | 223 of 235 (95%) | 0 | 12 |
| blind, not recognized (37) | 36 (97%) | 0 | 1 | 182 of 185 (98%) | 0 | 3 |
| blind, recognized (10) | 4 (40%) | 0 | 6 | 41 of 50 (82%) | 0 | 9 |

Marks changed after the reveal: 0 of 50 companies.

| Tag | Marked blind | Agreed |
| --- | ---: | ---: |
| none | 23 | 18 |
| narrow | 22 | 20 |
| wide | 2 | 2 |

Disputed overall calls ("too low" = the moat should be higher):

* **AMZN**: tagged none, returns above, too low (marked on the named page first). WIDE: Differs: recognized company (Amazon); 10-K text understates marketplace network effects, logistics scale and AWS lock-in, and returns are above cost of capital.
* **PG**: tagged narrow, returns above, too low (recognized while blind). WIDE: Disagree, upgrade to wide: recognizably P&G; category-leading brands plus distribution and advertising scale have sustained ~20% ROIC and 22-24% margins for decades despite private label.
* **AJG**: tagged none, returns near, too low. NARROW: Insurance brokers have sticky client relationships and high retention; low ROE reflects acquisition goodwill while FCF margins are strong. Narrow rather than none.
* **BK**: tagged none, returns near, too low (recognized while blind). NARROW: Disagree mildly: recognizably BNY Mellon; custody and servicing for $52T of assets are deeply embedded and scale-driven, though ROE has mostly trailed the cost of equity.
* **BSX**: tagged none, returns below, too low (recognized while blind). NARROW: Disagree with none: recognizably Boston Scientific; FDA approvals, first-in-class devices and physician familiarity give lasting protection, and low ROIC reflects acquisition goodwill while operating margins are rising to 16%.
* **RMD**: tagged none, returns above, too low (recognized while blind). NARROW: Disagree with none: the excerpt is only regulatory, but ROIC of 24-31% and 27-33% operating margins sustained for five years strongly suggest a device franchise (possibly ResMed) with modest regulatory and installed-base advantages.
* **TDG**: tagged narrow, returns above, too low (recognized while blind). WIDE: Disagree, upgrade to wide: recognizably TransDigm; ~90% proprietary, sole-source certified parts with 25-30+ year aftermarket tails and 35-47% operating margins point to very durable pricing power.
* **TYL**: tagged none, returns below, too low (recognized while blind). NARROW: Disagree with none: recognizably Tyler; ~2% annual client turnover on mission-critical government systems is strong switching cost, and low ROIC mainly reflects NIC acquisition goodwill.

Disputed source ratings:

* AMZN: intangibles 0 too low; switching_costs 0 too low; network_effects 0 too low; cost_advantage 0 too low
* PG: intangibles 1 too low; cost_advantage 0 too low
* AJG: switching_costs 0 too low
* BK: switching_costs 0 too low
* RMD: intangibles 0 too low; switching_costs 0 too low
* EMR: switching_costs 0 too low
* TDG: intangibles 1 too low; switching_costs 1 too low
* AKAM: switching_costs 0 too low
* FTNT: switching_costs 0 too low
* TYL: switching_costs 1 too low

### What the blind check shows

* **Blind, v3 agrees with the reviewer as often as v1 did with names shown.** 40 of 47 overall calls and 223 of 235 source ratings, against 40 of 47 and 215 of 235 for the v1 tags of the first set. Every dispute says the moat should be higher (7 overall calls, 12 source ratings); none says lower.
* **The reveal changed no mark** (0 of 50), but the blinding was partial: the notes name 10 of the 47 companies while still blind. Disputes sit almost entirely there: 6 of the 10 recognized companies were called too low (BSX, TDG, PG, RMD, BK, TYL), against 1 of the 37 that were not (AJG). On the text and financials alone, the tagger and the reviewer agree on 36 of 37 overall calls.
* **What that does and does not show.** The recognized companies are the well-known franchises, where Item 1 most understates the moat, so the disputes may be outside knowledge (and hindsight) rather than misreading. The tagger is not meant to use that knowledge: a point-in-time tag should rest on what the filing says. The low-side errors are where a text-only tagger is expected to fall short.
* **Acquisition goodwill.** Three disputes (BSX, TYL, AJG) say low returns reflect acquired goodwill. ROIC over equity + debt - cash counts goodwill as capital, so serial acquirers look as if they barely earn their cost of capital. A return on tangible capital would address it, but as a new prompt it would need freezing and another fresh set.

## Review sheet

For each company: what it does (the opening of Item 1, name hidden), the overall call, then each source rated above 0 with the model's reason and the quotes it cites, and the quotes against a moat. Sources rated 0 are listed by name only.

### IPG: INTERPUBLIC GROUP OF COMPANIES, INC. (Communication Services, 10-K filed 2025-02-20)

> Business The the Company ("the Company," the "Company," "the Company," "we," "us" or "our") was incorporated in Delaware in September 1930 under the name of McCann-Erickson Incorporated as the successor to the advertising agency businesses founded in 1902 by A.W. Erickson and in 1911 by Harrison K. ...

**Overall: none.** The text describes a competitive, relationship-driven services business with short, cancelable contracts and easily switched clients. Good returns do not make this a moat when the text shows no durable mechanism, and revenue is flat to declining.

**Returns: above.** ROIC was 21-32% in 2021-2024, well above the cost of capital, but fell to 11% in 2020 and has declined since 2021. Revenue is flat to declining. Returns rest on a thin capital base typical of service firms, with goodwill-heavy balance sheet effects uncertain.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -11% | 6% | 4% | 19% | 11% | 12% | 2% |
| 2021-12-31 | 13% | 14% | 9% | 18% | 32% | 29% | 5% |
| 2022-12-31 | 7% | 13% | 9% | 4% | 30% | 26% | 5% |
| 2023-12-31 | -0% | 14% | 10% | 3% | 27% | 29% | 6% |
| 2024-12-31 | -2% | 11% | 6% | 9% | 21% | 18% | 4% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Our contracts generally provide for termination by either party on relatively short notice, usually 30 to 90 days"
  * [2] "Any client may transfer its business from one of our agencies to another one of our agencies or to a competing agency, and a client may change its marketing budget at any time."
  * [3] "We operate in a highly competitive advertising and marketing communications industry."
  * [4] "Our operating companies compete against other large multinational advertising and marketing communications companies as well as numerous independent and niche agencies and new forms of market participants to win new clients and maintain existing client relationships."
  * [6] "There is keen competition for qualified employees."

### T: AT&T INC. (Communication Services, 10-K filed 2025-02-12)

> BUSINESS GENERAL the Company (“the Company,” “we” or the “Company”) is a holding company incorporated under the laws of the State of Delaware in 1983 and has its principal executive offices at 208 S. Akard St., Dallas, Texas, 75202 (telephone number 210-821-4105). We maintain an internet website at ...

**Overall: none.** The Company holds spectrum licenses and has scale in a concentrated industry, but it competes on price against cable and other carriers, its legacy business is declining, and customers switch easily. Returns have been weak and below the cost of capital. The text states no clear moat.

**Returns: below.** ROIC was 2%, 6%, -1%, 8% and 6% over five years, below or at the cost of capital. Revenue was shrinking or flat and ROE was volatile. The 2022 loss and the divestitures distort the figures.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -21% | 6% | -4% | 16% | 2% | -3% | -1% |
| 2021-12-31 | -6% | 19% | 15% | 16% | 6% | 11% | 4% |
| 2022-12-31 | -10% | -4% | -7% | 13% | -1% | -6% | -2% |
| 2023-12-31 | 1% | 19% | 12% | 17% | 8% | 13% | 4% |
| 2024-12-31 | -0% | 16% | 9% | 15% | 6% | 9% | 3% |

* **intangibles = 1.** FCC spectrum licenses are a real regulatory barrier, but they are issued for fixed terms and are held by the other national carriers too, so they limit entry without differentiating the Company.
  * [1] "Facilities-based wireless communications providers in the United States, like the Company, must be licensed by the FCC to provide communications services at specified spectrum frequencies within defined geographic areas and must comply with FCC rules and policies governing the use of the spectrum."
  * [2] "Building on that fiber foundation is our solid spectrum portfolio, strengthened through Federal Communications Commission (FCC) auction acquisitions and 5G deployment."
* **cost_advantage = 1.** The Company claims a low-cost converged network and a software-based cost advantage. These are self-assessed claims, not quantified against competitors.
  * [4] "This network approach delivers a demonstrable cost advantage in the deployment of next-generation technology over the traditional, hardware-intensive network approach."
  * [5] "Our concentration is to aggregate the most traffic on the largest, lowest marginal cost, converged network through efficient spectrum deployment and construction of the largest high-capacity broadband solutions in the United States"
* **efficient_scale = 1.** The U.S. wireless market is an oligopoly of three national carriers with high spectrum and capital barriers. The text still lists many competitors, including regional carriers, resellers and cable companies, and does not claim the market supports only a few firms.
  * [3] "Our competitors include two national wireless providers; a larger number of regional providers and resellers of each of those providers’ services; and certain cable companies."
* Rated 0: switching_costs, network_effects.
* Against a moat:
  * [6] "We compete for customers based principally on service/device offerings, price, network quality, coverage area and customer service."
  * [7] "We continue to lose legacy voice and data subscribers due to industry-wide secular declines and competitors (e.g., wireless, cable and VoIP providers) who can provide comparable services at lower prices"
  * [8] "We face substantial competition in our wireless businesses."
  * [9] "We provide our customers the ability to purchase handsets on an installment basis and the opportunity to bring their own device."

### AMZN: AMAZON COM INC (Consumer Discretionary, 10-K filed 2025-02-07)

> Business This Annual Report on Form 10-K and the documents incorporated herein by reference contain forward-looking statements based on expectations, estimates, and projections as of the date of this filing. Actual results and outcomes may differ materially from those expressed in forward-looking ...

**Overall: none.** The text states no moat mechanism directly and stresses intense competition, low barriers to entry on the internet and price-driven retail. Returns are strong in most years but are not enough to infer a moat from this text alone.

**Returns: above.** ROIC was 27%, 17%, 6%, 16% and 24%, mostly above the cost of capital but volatile, with a 2022 dip to 6% and a negative ROE. Returns are good but not consistent.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 38% | 6% | 6% | 7% | 27% | 27% | 8% |
| 2021-12-31 | 22% | 5% | 7% | -3% | 17% | 29% | 9% |
| 2022-12-31 | 9% | 2% | -1% | -3% | 6% | -2% | -1% |
| 2023-12-31 | 12% | 6% | 5% | 6% | 16% | 17% | 6% |
| 2024-12-31 | 11% | 11% | 9% | 5% | 24% | 24% | 10% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The worldwide marketplace in which we compete is evolving rapidly and intensely competitive, and we face a broad array of competitors from many different industry sectors around the world."
  * [2] "We believe that the principal competitive factors in our retail businesses include selection, price, and convenience, including fast and reliable fulfillment."
  * [3] "The internet facilitates competitive entry and comparison shopping, which enhances the ability of new, smaller, or lesser-known businesses to compete against us."
  * [4] "Some of our current and potential competitors have greater resources, longer histories, more customers, greater brand recognition, and greater control over inputs critical to our various businesses."

### AZO: AUTOZONE INC (Consumer Discretionary, 10-K filed 2025-10-27)

> Introduction the Company. (“the Company,” the “Company,” “we,” “our” or “us”) is a leading retailer and distributor of automotive replacement parts and accessories in the Americas. We began operations in 1979 and at August 30, 2025, operated 6,627 stores in the United States (“U.S.”), 883 stores in ...

**Overall: narrow.** The text describes a highly competitive, fragmented market with substitutable supply. Moat sources are modest: exclusive brands and a hub-based distribution network that gives parts availability. Consistently high returns and stable 20% margins, sustained over many years, corroborate a narrow advantage from scale, density and availability, though the text does not quantify it.

**Returns: above.** ROIC was 63% to over 100% over five years, far above the cost of capital. Operating margins of about 20% were steady. Equity is small because of buybacks, which inflates ROIC, but returns are still clearly high, and growth is slowing.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-08-28 | 16% | 20% | 15% | 20% | 90% | n/a | 15% |
| 2022-08-27 | 11% | 20% | 15% | 16% | >100% | n/a | 16% |
| 2023-08-26 | 7% | 20% | 14% | 12% | >100% | n/a | 16% |
| 2024-08-31 | 6% | 20% | 14% | 10% | 85% | n/a | 16% |
| 2025-08-30 | 2% | 19% | 13% | 9% | 63% | n/a | 14% |

* **intangibles = 1.** Exclusive in-house brands such as Duralast are a differentiator, but the text gives no evidence of pricing power, and the market is highly competitive.
  * [2] "A key differentiating component versus our competitors is our exclusive line of in-house brands, which includes Duralast and the family of Duralast brands, Econocraft, ProElite, ShopPro, SureBilt, TotalPro, TruGrade and Valucraft."
* **cost_advantage = 1.** A hub-and-spoke distribution network and centralized support give scale-based availability and cost benefits. The text does not compare costs with competitors, so the advantage is modest and uncertain.
  * [1] "Stores are replenished primarily by the nearest distribution center but also typically have same-day access to one of our 367 domestic and 53 international hub stores’ expanded inventory assortment."
  * [3] "We believe that this centralization enhances consistent execution of our merchandising and marketing strategies at the store level, while reducing expenses and cost of sales."
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [4] "The sale of automotive parts, accessories and maintenance items is highly competitive."
  * [5] "We believe alternative sources of supply exist, at similar costs, for most types of product sold."
  * [6] "Our competitors include national, regional and local auto parts chains, independently owned parts stores, online automotive parts stores or marketplaces, wholesale distributors, jobbers, repair shops, car washes and auto dealers,"

### DECK: DECKERS OUTDOOR CORP (Consumer Discretionary, 10-K filed 2025-05-23)

> GENERAL We are a global leader in designing, marketing, and distributing innovative footwear, apparel, and accessories developed for both everyday casual lifestyle use and high-performance activities. We market our products primarily under five proprietary brands: UGG, HOKA, Teva, AHNU, and ...

**Overall: narrow.** The brand-driven intangible advantage in UGG and HOKA is supported by exceptional, sustained returns, but the text describes a fashion-driven, easy-entry, highly competitive market with no switching costs or other mechanisms. Brand strength is plausible for about a decade, so the moat is narrow.

**Returns: above.** ROIC is 70% to over 100% across all five years, with operating margins of 18-24% and ROE of 30-42%. These are far above the cost of capital, reflecting an asset-light model and strong brand profitability.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-03-31 | 19% | 20% | 15% | 22% | 94% | 30% | 19% |
| 2022-03-31 | 24% | 18% | 14% | 4% | 85% | 30% | 20% |
| 2023-03-31 | 15% | 18% | 14% | 13% | 70% | 31% | 21% |
| 2024-03-31 | 18% | 22% | 18% | 22% | >100% | 39% | 27% |
| 2025-03-31 | 16% | 24% | 19% | 19% | >100% | 42% | 29% |

* **intangibles = 1.** UGG and HOKA are strong consumer brands with loyal followings and high margins, but the text offers little quantified evidence of pricing power or durability. Fashion brands can be fickle, and patents are generic.
  * [1] "The UGG brand is one of the most iconic and recognized footwear brands in our industry, which highlights our successful track record of building niche brands into lifestyle and fashion market leaders."
  * [2] "which has quickly become a leading brand within run and outdoor specialty wholesale accounts and is growing across its global marketplace"
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "In addition, access to offshore manufacturing and the growth of e-commerce has made it easier for new companies to enter the markets in which we compete, further increasing competition in the footwear, apparel, and accessories industry."
  * [5] "Although the industry is fragmented, many of our competitors are larger and have substantially greater resources, several of which compete directly with some of our products."
  * [6] "We generally purchase products from independent manufacturers on the basis of individual purchase orders, rather than maintaining long-term purchase commitments"
  * [7] "we believe our ability to successfully compete depends on numerous factors, including our ability to predict, assess, and respond quickly to changing consumer tastes and preferences"

### HLT: Hilton Worldwide Holdings Inc. (Consumer Discretionary, 10-K filed 2025-02-06)

> Overview the Company is one of the largest global hospitality companies, with 8,447 properties comprising 1,268,206 rooms in 140 countries and territories as of December 31, 2024. Founded in 1919, the Company has been an innovator in the industry for over 105 years, driven by the vision of founder ...

**Overall: narrow.** Long management and franchise contracts, strong brands and a very large loyalty program are real advantages, and ROIC above 25% for three years supports them. The advantages are not quantified, though. Competition is active, and owners can leave on performance failures. That fits a narrow moat rather than a wide one.

**Returns: above.** ROIC fell to -5% in 2020 and then recovered to 13%, 26%, 28% and 31% in 2021-2024. Asset-light franchising gives high returns, and the last three years are well above the cost of capital. Small or negative equity may inflate these figures.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -54% | -10% | -17% | 15% | -5% | n/a | -5% |
| 2021-12-31 | 34% | 17% | 7% | 1% | 13% | n/a | 3% |
| 2022-12-31 | 52% | 24% | 14% | 19% | 26% | n/a | 8% |
| 2023-12-31 | 17% | 22% | 11% | 18% | 28% | n/a | 7% |
| 2024-12-31 | 9% | 21% | 14% | 17% | 31% | n/a | 10% |

* **intangibles = 1.** A large portfolio of brands, with Hampton ranked the #1 lodging brand to franchise for 16 years running, attracts owners and guests. The text gives no pricing evidence, though, and the other quotes are generic.
  * [4] "Hampton has been ranked the #1 lodging brand to franchise by Entrepreneur for 16 consecutive years."
  * [8] "We believe that our capabilities as a multi-branded manager, franchisor, owner and lessee of hotels with an associated global, system-wide guest loyalty program and commercial platform help us continue to maintain our position as one of the largest and most geographically diverse hospitality companies in the world."
* **switching_costs = 1.** Owners are tied to the Company by 20-30 year management contracts and roughly 20-year franchise contracts, and conversion costs are high. Some contracts allow early termination on performance failures, which tempers the lock-in.
  * [2] "The initial terms of our management contracts are typically 20 to 30 years."
  * [3] "Our franchise contracts typically have initial terms of approximately 20 years for new hotels and approximately 10 to 20 years for hotels converting from hotels outside of our system."
  * [7] "HGV has the exclusive right to use our timeshare brands, subject to the terms of a long-term license agreement with us."
* **network_effects = 1.** The 211 million-member loyalty program drives repeat business and draws owners, and members spend more as they move up tiers. This is a loyalty program feeding a booking platform, so the network effect is modest.
  * [1] "As of December 31, 2024, we had 211 million members in our award-winning guest loyalty program, the Company Honors, an increase of 17 percent from December 31, 2023"
  * [5] "Affiliation with our loyalty program encourages members to allocate more of their travel spend to our hotels."
  * [6] "The percentage of travel spend we capture from loyalty members increases as they move up the tiers of our program."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [9] "We encounter active and robust competition as a hotel and resort manager, franchisor, owner and lessee."
  * [10] "Some of our management contracts provide early termination rights to hotel owners upon certain events, including the failure to meet certain financial or performance criteria."
  * [11] "Our properties and brands compete with other hotels, resorts, motels, inns and other accommodation rental services in their respective geographic markets or customer segments"

### KMX: CARMAX INC (Consumer Discretionary, 10-K filed 2025-04-11)

> BUSINESS OVERVIEW the Company Background the Company. delivers an unrivaled customer experience by offering a broad selection of quality used vehicles and related products and services at competitive, no-haggle prices using a customer-friendly sales process. We are the nation’s largest retailer of ...

**Overall: none.** The company has scale and a brand in a highly fragmented, competitive market, with a small share, no switching costs and declining returns. Its advantages are modest and not durable enough to count as a moat.

**Returns: below.** ROIC is not reported. ROE fell from 18-24% to 8-9% in the last three years, with net margins of about 2% and ROA of 2%. Returns are near or below the cost of capital, and a financing arm inflates leverage.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-02-28 | -7% | n/a | 4% | 3% | n/a | 18% | 4% |
| 2022-02-28 | 68% | n/a | 4% | -9% | n/a | 24% | 5% |
| 2023-02-28 | -7% | n/a | 2% | 3% | n/a | 9% | 2% |
| 2024-02-29 | -11% | n/a | 2% | -0% | n/a | 8% | 2% |
| 2025-02-28 | -1% | n/a | 2% | 1% | n/a | 8% | 2% |

* **cost_advantage = 1.** Appraisal-driven sourcing and scale enable its own auctions and a large nationwide inventory, which is a modest advantage, but no cost comparison is quantified.
  * [2] "Our high volume of appraisal purchases, further supported by our online instant appraisal offers and MaxOffer, supplies not only a large portion of our retail inventory, but also provides the scale that enables us to conduct our own wholesale auctions to dispose of vehicles that do not meet our retail standards."
  * [3] "This gives the Company customers access to a much larger selection of vehicles than any traditional auto retailer."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [5] "The U.S. used car marketplace is highly fragmented, and we face competition from franchised dealers, who sell both new and used vehicles; online and mobile sales platforms; independent used car dealers; and private parties."
  * [6] "Competition in our industry has evolved with the adoption of online platforms and marketing tools, all of which facilitate increased competition."
  * [7] "In calendar 2024, we estimate we sold approximately 3.7% of the age 0- to 10-year old vehicles sold on a nationwide basis, consistent with calendar 2023."
  * [8] "All finance offers, whether from CAF or our third-party providers, are backed by a 3-day payoff option, which allows customers to refinance their loan with another finance provider within three business days at no charge."
  * [9] "CAF’s primary competitors are banks and credit unions that offer direct financing to customers purchasing used cars."

### COST: COSTCO WHOLESALE CORP /NEW (Consumer Staples, 10-K filed 2025-10-08)

> the Company Wholesale Corporation and its subsidiaries (the Company or the Company) began operations in 1983, in Seattle, Washington. We are principally engaged in the operation of membership warehouses in the United States (U.S.) and Puerto Rico, Canada, Mexico, Japan, the United Kingdom (U.K.), ...

**Overall: wide.** The scale-based cost advantage, a low-price value proposition and a sticky paid membership base reinforce each other and produce sustained very high returns, despite a competitive industry with Walmart, Amazon and other clubs. Advantages are durable but contested, so narrow is the careful verdict, though the reinforcing sources and returns argue for wide.

**Returns: above.** ROIC of 37-41% and ROE of 27-31% for five years are well above the cost of capital, with stable thin 3-4% operating margins on high turnover. Returns are not driven by small equity.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-08-29 | 17% | 3% | 3% | 3% | 37% | 27% | 9% |
| 2022-08-28 | 16% | 3% | 3% | 2% | 39% | 30% | 9% |
| 2023-09-03 | 7% | 3% | 3% | 3% | 37% | 28% | 9% |
| 2024-09-01 | 5% | 4% | 3% | 3% | 39% | 30% | 11% |
| 2025-08-31 | 8% | 4% | 3% | 3% | 41% | 31% | 11% |

* **intangibles = 1.** Kirkland Signature private label and a well-recognized warehouse brand that supports membership and differentiation, but described mostly in general terms.
  * [7] "We have invested significantly in the development and protection of our well-recognized brands, including the the Company Wholesale trademarks and our private-label brand, Kirkland Signature."
  * [8] "We believe that Kirkland Signature products are high quality, offered at prices that are generally lower than national brands, and help lower costs, differentiate our merchandise offerings, and generally earn higher margins."
* **switching_costs = 1.** Very high membership renewal rates and heavy Executive member penetration show sticky members, though customers face no true contractual or integration lock-in.
  * [6] "Our member renewal rate was 92.3% in the U.S. and Canada and 89.8% worldwide at the end of 2025."
  * [9] "The sales penetration of Executive members represented approximately 73.6% of worldwide net sales in 2025."
* **cost_advantage = 2.** Volume purchasing, limited SKUs, cross-docking, low shrinkage and operating at significantly lower gross margins than most retailers, with float from selling inventory before payment. This is a scale-based structural advantage that is also reflected in the returns.
  * [1] "these volumes and turnover enable us to operate profitably at significantly lower gross margins (net sales less merchandise costs) than most other retailers."
  * [2] "We often sell inventory before we are required to pay for it, even while taking advantage of early payment discounts."
  * [3] "This process creates freight volume and handling efficiencies, lowering"
  * [4] "We carry less than 4,000 active stock keeping units (SKUs) per warehouse in our core warehouse business, significantly less than other broadline retailers."
  * [5] "we believe our inventory losses (shrinkage) are well below those of typical retail operations."
* Rated 0: network_effects, efficient_scale.
* Against a moat:
  * [10] "Our industry is highly competitive, based on factors such as price, merchandise quality and selection, location, convenience, distribution strategy, and customer service."
  * [11] "Many of the metropolitan areas in the U.S. and certain of our Other International locations have multiple competing clubs."

### K: KELLANOVA (Consumer Staples, 10-K filed 2025-02-21)

> The Company. the Company (formerly Kellogg Company), founded in 1906 and incorporated in Delaware in 1922, and its subsidiaries are engaged in the manufacture and marketing of snacks and convenience foods. The address of the principal business office of the Company is 412 N. Wells Street, Chicago, ...

**Overall: narrow.** Established snack brands, with rankings that are not quantified, plus decent returns suggest a modest brand-based advantage. Intense competition, lower-priced private label, and concentrated powerful retail customers limit it. The brand intangible looks real but not strongly evidenced, so the moat is narrow.

**Returns: above.** ROIC was 9-17% over five years, mostly at or modestly above the cost of capital, with a 2024 high of 17%. ROE of 22-39% is inflated by leverage. Margins are steady in the low-to-mid teens.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-02 | 1% | 13% | 9% | 11% | 13% | 36% | 7% |
| 2022-01-01 | -15% | 12% | 13% | 10% | 10% | 38% | 8% |
| 2022-12-31 | 8% | 10% | 8% | 9% | 9% | 22% | 5% |
| 2023-12-30 | 4% | 11% | 7% | 7% | 13% | 25% | 6% |
| 2024-12-28 | -3% | 15% | 11% | 9% | 17% | 39% | 9% |

* **intangibles = 1.** Well-known snack brands (Pringles, Cheez-It, Pop-Tarts, Eggo) support pricing and shelf presence, but the text only describes trademarks generally and gives no figures on share or price premiums. Brand strength is implied rather than stated directly.
  * [1] "Registrations of trademarks can also generally be renewed indefinitely as long as the trademarks are in use."
  * [2] "One of our subsidiaries is also the exclusive licensee of the Carr’s cracker line in the United States."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "Our products compete with advertised and branded products of a similar nature as well as unadvertised and private label products, which are typically distributed at lower prices, and generally with other food products."
  * [4] "Principal methods and factors of competition include new product introductions, product quality, taste, convenience, nutritional value, price, advertising and promotion."
  * [5] "During 2024, our top five customers, collectively, including Wal-Mart, accounted for approximately 29% of our consolidated net sales and approximately 46% of U.S."
  * [6] "There has been significant worldwide consolidation in the grocery industry and we believe that this trend is likely to continue."

### KMB: KIMBERLY CLARK CORP (Consumer Staples, 10-K filed 2025-02-13)

> Description of the Company-Clark the Company-Clark Corporation was founded in 1872 and incorporated in Delaware in 1928. We are a global company focused on delivering products and solutions that provide better care for a better world through product innovation and building o ur brands. We are ...

**Overall: narrow.** The business is in consumer staples with strong brands such as Huggies and Kleenex, and the very high ROIC sustained for five years backs the brand-based advantage. The text only describes the brands and does not quantify them. Competition from larger rivals, price competition and Walmart's 14% share of sales weigh against a wide moat. This supports a narrow moat.

**Returns: above.** ROIC was 22-34% in all five years, well above the cost of capital, with operating margins of 11-17%. ROE is above 100% because equity is small, so it is discounted. ROIC is still strong and consistent. Revenue growth is low.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 4% | 17% | 12% | 13% | 31% | >100% | 14% |
| 2021-12-31 | 2% | 13% | 9% | 9% | 23% | >100% | 10% |
| 2022-12-31 | 4% | 13% | 10% | 9% | 24% | >100% | 11% |
| 2023-12-31 | 1% | 11% | 9% | 14% | 22% | >100% | 10% |
| 2024-12-31 | -2% | 16% | 13% | 13% | 34% | >100% | 15% |

* **intangibles = 1.** The company owns a portfolio of well-known consumer brands, and brand recognition and loyalty is listed as a basis of competition. But the text gives no figures on market share, rank or price premium, and generic patent and trademark statements do not count. The brand strength is only suggested, and high returns point to a real advantage.
  * [4] "Our portfolio of brands, including Huggies, Kleenex, Scott, Kotex, Cottonelle, Poise, Depend, Andrex, Pull-Ups, GoodNites, Intimus, Plenitud, Sweety, Softex, Viva and WypAll, encompass five global daily-need product categories"
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "We have several major competitors in most of our markets, some of which are larger and more diversified than us."
  * [2] "The principal methods and elements of competition include brand recognition and loyalty, product innovation, quality and performance, price, and marketing and distribution capabilities."
  * [3] "Our largest customer, Walmart Inc., represented approximately"

### PG: PROCTER & GAMBLE Co (Consumer Staples, 10-K filed 2025-08-04)

> Business. The the Company Company (the Company) is a world-leading multinational consumer goods company focused on providing trusted, branded products of superior quality, performance and value to improve the lives of consumers around the world - now and for generations to come. Our products are ...

**Overall: narrow.** The text only weakly states the brand and market-share advantage, and it admits intense competition, private-label rivals and powerful retail customers. However, five years of ROIC of about 20% and stable 22-24% operating margins corroborate that its brand portfolio and scale in daily-use categories sustain pricing power. This supports a narrow moat, not a wide one, given private-label pressure and concentrated customers.

**Returns: above.** ROIC steady at 20-21% for five years and ROE about 30%, with operating margins of 22-24%. This is well above the 8-10% cost of capital. ROE is somewhat boosted by buybacks, but ROIC confirms the high returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-06-30 | 7% | 24% | 19% | 20% | 21% | 31% | 12% |
| 2022-06-30 | 5% | 22% | 18% | 17% | 20% | 32% | 12% |
| 2023-06-30 | 2% | 22% | 18% | 17% | 20% | 31% | 12% |
| 2024-06-30 | 2% | 22% | 18% | 20% | 20% | 30% | 12% |
| 2025-06-30 | 0% | 24% | 19% | 17% | 21% | 31% | 13% |

* **intangibles = 1.** Brands and leadership/significant market share positions in many categories are stated, but the text gives no figures and the patent and trademark language is generic. Direct mechanism stated, though not quantified.
  * [1] "In this highly competitive setting, we are well positioned in the industry segments and markets in which we operate, often holding a leadership or significant market share position."
  * [2] "Our patents cover a range of product features, including significant product formulation and processes used to manufacture our products."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "The markets in which our products are sold are highly competitive."
  * [4] "In many of the markets and industry segments, we compete against other branded products as well as retailers' private-label brands."
  * [5] "Our top ten customers accounted for 43% of our total net sales in 2025, 42% in 2024 and 40% in 2023."
  * [6] "Sales to Walmart Inc. and its affiliates represent approximately 16% of our total sales in 2025 and 2024 and 15% in 2023."

### SLB: SCHLUMBERGER LIMITED/NV (Energy, 10-K filed 2025-01-22)

> usiness. All references in this report to “Registrant,” “Company,” “the Company,” “we” or “our” are to the Company (the Company) and its consolidated subsidiaries. We are the Company, a global technology company driving energy innovation for a balanced planet. With a global presence in more than ...

**Overall: none.** The company is a scaled, technology-driven oilfield services leader, but the text gives no strong, directly stated moat source. Competition is on technology, service and price against numerous competitors, and no patent is material. Recent returns are strong, but they follow a cyclical recovery and are not consistent across the five years.

**Returns: near.** ROIC rose from 6% in 2020 to 10%, 15% and 19% in 2021-2023, with operating margins improving to 20%. ROE was about 22%. Returns are above the cost of capital only in the recent cyclical upswing, after a weak 2020.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -28% | 10% | -45% | 8% | 6% | -57% | -21% |
| 2021-12-31 | -3% | 15% | 8% | 15% | 10% | 14% | 4% |
| 2022-12-31 | 23% | 18% | 12% | 7% | 15% | 21% | 8% |
| 2023-12-31 | 18% | 20% | 13% | 14% | 19% | 22% | 9% |
| 2024-12-31 | 10% | n/a | 12% | 13% | n/a | 22% | 9% |

* **intangibles = 1.** A large IP portfolio and an extensive exploration data library are cited, but the company says no single patent or group of patents is material, so the advantage is modest and unquantified.
  * [1] "the Company owns or controls one of the industry’s leading portfolios of intellectual property, including but not limited to patents, proprietary information, trade secrets, and software tools and applications that, in the aggregate, are material to the Company’s business."
  * [6] "includes one of the industry’s most extensive exploration data libraries."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [2] "While the Company seeks and holds a significant number of patents covering various products and processes, no particular patent or group of patents is material to the Company’s business."
  * [3] "The principal methods of competition within the energy services industry are technological innovation, quality of service, and price differentiation."
  * [4] "the Company has numerous competitors, both large and small."
  * [7] "Our software products sold directly to customers, which are agnostic to equipment provider, enable automation and autonomy to reduce cost and improve performance."

### AJG: Arthur J. Gallagher & Co. (Financials, 10-K filed 2025-02-18)

> usiness. Overview the Company and its subsidiaries, collectively referred to herein as we, our, us or Gallagher, are engaged in providing insurance brokerage, reinsurance brokerage, consulting, and third-party property/casualty claims settlement and administration services to entities and ...

**Overall: none.** A large global broker with a diversified client base, but the text describes a highly competitive market with many rivals and no directly stated moat source. Returns are near the cost of capital, so no moat can be confirmed from this text.

**Returns: near.** ROIC is not reported. ROE of 9-14% is near a 10% cost of equity and fell to 9% in 2024, while ROA is low (2-4%). Margins and free cash flow are solid, but returns are not clearly above the cost of capital, and acquisitions distort the figures.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | n/a | 12% | 24% | n/a | 14% | 4% |
| 2021-12-31 | 17% | n/a | 11% | 15% | n/a | 12% | 3% |
| 2022-12-31 | 4% | n/a | 13% | 14% | n/a | 13% | 3% |
| 2023-12-31 | 18% | n/a | 10% | 18% | n/a | 10% | 2% |
| 2024-12-31 | 15% | n/a | 13% | 21% | n/a | 9% | 3% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "The insurance and reinsurance brokerage and consulting businesses are highly competitive and there are many organizations and individuals throughout the world who actively compete with us in every area of our business."
  * [5] "Additionally, we also face competition from insurance and reinsurance carriers that market, distribute and service a portion of their products directly, and in some cases from banks, consulting and accounting firms, and technology companies that can provide alternative risk management products or services."
  * [6] "In 2024, our largest single client represented approximately 1% and our ten largest clients together represented approximately 3%, respectively, of our combined brokerage and risk management segment revenues."

### BK: Bank of New York Mellon Corp (Financials, 10-K filed 2025-02-27)

> Description of Business The the Company of New York Mellon Corporation, a Delaware corporation (NYSE symbol: the Company), is a global company headquartered in New York, New York, with $52.1 trillion in assets under custody and/or administration and $2.0 trillion in assets under management as of ...

**Overall: none.** The text states no moat mechanism directly. It describes intense competition, including from fintechs, and returns are only near the cost of capital. Scale in custody is plausible but not evidenced in the quotes.

**Returns: near.** As a financial company, ROE is the relevant measure. ROE was 6-8% for four years and 11% in 2024, near or below the ~10% cost of equity. ROA is about 1%. ROIC is not reported.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | n/a | 23% | 24% | n/a | 8% | 1% |
| 2021-12-31 | 1% | n/a | 24% | 10% | n/a | 8% | 1% |
| 2022-12-31 | 4% | n/a | 15% | 83% | n/a | 6% | 1% |
| 2023-12-31 | 7% | n/a | 19% | 27% | n/a | 8% | 1% |
| 2024-12-31 | 5% | n/a | 24% | -4% | n/a | 11% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Competition in the financial services industry continues to be intense."
  * [2] "Our competitive position may be affected by institutions that are not similarly subject to extensive regulation, such as financial technology firms, and as further technological advances enable more companies, such as financial technology firms, to provide financial services."
  * [3] "BNY is subject to competition in all aspects and areas of our business."

### BX: Blackstone Inc. (Financials, 10-K filed 2025-02-28)

> Business Overview the Company is the world’s largest alternative asset manager. We seek to deliver compelling returns for institutional and individual investors by strengthening the companies and assets in which we invest. Our more than $1.1 trillion in Total Assets Under Management as of December ...

**Overall: narrow.** The Company is the largest alternative manager and its brand and track record help it raise long-dated capital, and ROE has stayed above the cost of equity every year. Against that, the text stresses intense competition, fee pressure, investors who can terminate or in-source, and a business that relies on talent. The brand and scale advantage looks real but is not quantified, so I rate it narrow.

**Returns: above.** ROIC is not reported. ROE ran 15-73% over the five years, well above 10%, with net margins of 17-26%. Revenue swings widely because carried interest and performance revenues fluctuate, and part of the ROE comes from leverage and a structure that pays out most earnings.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -17% | n/a | 17% | 30% | n/a | 15% | 4% |
| 2021-12-31 | >100% | n/a | 26% | 17% | n/a | 73% | 17% |
| 2022-12-31 | -62% | n/a | 21% | 72% | n/a | 20% | 4% |
| 2023-12-31 | -6% | n/a | 17% | 48% | n/a | 19% | 3% |
| 2024-12-31 | 65% | n/a | 21% | 26% | n/a | 37% | 7% |

* **intangibles = 1.** Claims of being the world's largest alternative manager, with a leading rank in hedge fund allocation and credit/CLO management, point to a brand and track record that attract institutional capital. The text does not quantify pricing power or a lasting edge.
  * [1] "the Company is the world’s largest alternative asset manager."
  * [2] "BXMA, the world’s largest discretionary allocator to hedge funds, seeks to grow investors’ assets through investment strategies designed to deliver, primarily through the public markets, compelling risk-adjusted returns."
  * [3] "BXCI is one of the largest credit managers and CLO managers in the world."
* **switching_costs = 1.** Long-dated fund terms and perpetual capital vehicles lock in fee-paying capital, but some mandates can be terminated on 30 days' notice and investors can remove the general partner. The lock-in is real but partial.
  * [4] "Management fees are generally payable over either the term or life of the fund."
  * [5] "Perpetual Capital strategies represent a significant and growing portion of our overall business, and the management fees and performance revenues we receive."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The asset management industry is intensely competitive, and we expect it to remain so."
  * [7] "Such management fees are generally subject to contractual rights the investor has to terminate our management on generally as short as 30 days’ notice."
  * [8] "Certain institutional investors have demonstrated a preference to in-source their own investment professionals and to make direct investments in alternative assets without the assistance of private equity advisers like us."
  * [9] "Some of these competitors may also have a lower cost of capital and access to funding sources or other resources that are not available to us, which may create competitive disadvantages for us with respect to investment opportunities."
  * [10] "Third-party investors in some of our partnership funds have the right to remove the general partner of the fund or to accelerate the termination of the fund without cause by a majority or supermajority vote."

### GS: GOLDMAN SACHS GROUP INC (Financials, 10-K filed 2025-02-27)

> Introduction the Company is a leading global financial institution that delivers a broad range of financial services to a large and diversified client base that includes corporations, financial institutions, governments and individuals. Our purpose is to advance sustainable economic growth and ...

**Overall: none.** The firm has a strong reputation in advisory and underwriting. But the text describes intense price competition, pricing pressure across most businesses and many large rivals. ROE is near the cost of equity and cyclical, so the moat is thin at best.

**Returns: near.** ROE was 10%, 21%, 10%, 7% and 12% over the five years, averaging about 12%. That is near the roughly 10% cost of equity and volatile. ROA is only 1-2%, and ROIC is not reported.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 22% | n/a | 21% | -56% | n/a | 10% | 1% |
| 2021-12-31 | 33% | n/a | 36% | 3% | n/a | 21% | 2% |
| 2022-12-31 | -20% | n/a | 24% | 10% | n/a | 10% | 1% |
| 2023-12-31 | -2% | n/a | 18% | -32% | n/a | 7% | 1% |
| 2024-12-31 | 16% | n/a | 27% | -29% | n/a | 12% | 1% |

* **intangibles = 1.** The firm cites long-standing leadership in M&A advisory and equity underwriting, which suggests a reputation advantage. The text gives no figures such as league-table ranks, and it admits competition on price and reputation.
  * [1] "We have been a leader for many years in providing advisory services, including strategic advisory assignments with respect to mergers and acquisitions, divestitures, corporate defense activities, restructurings and spin-offs."
  * [2] "We regularly receive mandates for large, complex transactions and have held a leading position in worldwide public common stock offerings and worldwide initial public offerings for many years."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "The financial services industry and all of our businesses are intensely competitive, and we expect them to remain so."
  * [6] "A number of our businesses are subject to intense price competition."
  * [7] "Efforts by our competitors to gain market share have resulted in pricing pressure in our investment banking, market-making, consumer, wealth management and asset management businesses."
  * [8] "We have faced, and expect to continue to face, pressure to retain market share by committing capital to businesses or transactions on terms that offer returns that may not be commensurate with their risks."
  * [9] "The increasing prevalence of passive investment strategies that typically have lower fees than other strategies we offer has affected the competitive and pricing dynamics for our asset management products and services."

### MET: METLIFE INC (Financials, 10-K filed 2025-02-21)

> Index to Business Page Business Overview & Strategy 5 Segments and Corporate & Other 6 Policyholder Liabilities 10 Underwriting and Pricing 11 Reinsurance Activity 12 Regulation 12 Competition 23 Human Capital Resources 23 Information About Our Executive Officers 25 Trademarks 26 Available ...

**Overall: none.** This is a large, diversified insurer in a highly competitive industry with many similar rivals and customers who can move at renewal. Employer relationships and agency networks help a little, but the text shows no strong mechanism, and ROE sits near the cost of capital. No moat.

**Returns: near.** ROIC is high (23%, 46%, 55%) but is inflated by an equity base distorted by unrealized losses. ROE is 5-15%, roughly at the 10% cost of capital, with a 5% low in 2023. Returns are near cost of capital for a financial company.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | n/a | 8% | 17% | n/a | 8% | 1% |
| 2021-12-31 | 1% | n/a | 10% | 18% | n/a | 11% | 1% |
| 2022-12-31 | 0% | 9% | 8% | 19% | 23% | 13% | 1% |
| 2023-12-31 | -3% | 9% | 2% | 21% | 46% | 5% | 0% |
| 2024-12-31 | 6% | 8% | 6% | 21% | 55% | 15% | 1% |

* **intangibles = 1.** Long-standing employer relationships and a captive agency network are noted, but they are modest and not quantified. The text itself says competitors may have better ratings or pricing. Regulatory licensing is a general barrier, not a distinct edge.
  * [1] "We have built a leading position in the U.S. group insurance market through long-standing relationships with many of the largest employers in the U.S."
  * [2] "We have an exclusive and captive agency distribution network which sells a variety of individual life, accident & health, and pension products."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "The life insurance industry remains highly competitive."
  * [4] "Some of our competitors offer a broader array of products, have more competitive pricing or, with respect to other insurance companies, have higher claims paying ability ratings."
  * [5] "Because we and others underwrite many group insurance products annually, our group purchasers may be able to obtain more favorable terms from competitors rather than renewing coverage with us."
  * [6] "We compete globally with a large number of insurance companies and non-insurance financial services companies such as banks, broker-dealers and asset managers."

### RF: REGIONS FINANCIAL CORP (Financials, 10-K filed 2025-02-21)

> Business the Company is a FHC headquartered in Birmingham, Alabama operating in the South, Midwest and Texas. In addition, Regions operates several offices delivering specialty capabilities in New York, Washington D.C., Chicago, Salt Lake City, and other locations nationwide. Regions provides ...

**Overall: none.** The text describes a highly competitive, commoditized banking market with lower entry barriers from fintechs, and states no specific moat mechanism. Returns are only near the cost of equity.

**Returns: near.** ROE fell from 14% in 2021 to 11% in 2024, about equal to the 10% cost of equity. ROA is 1%, and ROIC of 1-2% is not meaningful for a bank. Returns are near the cost of capital, not clearly above it.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 6% | n/a | >100% | >100% | n/a | 6% | 1% |
| 2021-12-31 | 24% | 44% | >100% | >100% | n/a | 14% | 2% |
| 2022-12-31 | n/a | n/a | n/a | n/a | n/a | 13% | 1% |
| 2023-12-31 | n/a | n/a | n/a | n/a | 2% | 12% | 1% |
| 2024-12-31 | n/a | n/a | n/a | n/a | 1% | 11% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "All aspects of our business are highly competitive."
  * [2] "In addition, technology has lowered barriers to entry and made it possible for non-banks to offer traditional bank or bank-like products and services and therefore compete with financial institutions like us in providing electronic, internet-based and mobile phone-based financial solutions."
  * [3] "Many of our non-bank competitors are not subject to the same extensive regulations we are and, therefore, may have greater flexibility in competing for business."
  * [4] "Customers for banking services and other financial services offered by our subsidiaries are generally influenced by convenience, quality of service, price of service, personal contacts, the quality of the technology that supports the customer experience and availability of products."
  * [5] "We expect competition to remain intense among financial services companies."

### WFC: WELLS FARGO & COMPANY/MN (Financials, 10-K filed 2025-02-25)

> BUSINESS the Company Fargo & Company is a corporation organized under the laws of Delaware and a financial holding company and a bank holding company registered under the Bank Holding Company Act of 1956, as amended (BHC Act). Its principal business is to act as a holding company for its ...

**Overall: none.** The text describes a large, highly regulated universal bank in a highly competitive industry, with no stated durable advantages such as switching costs, network effects or cost advantages. ROE is roughly at the cost of equity, and regulatory constraints and consent orders limit it. No moat is evident from the text.

**Returns: near.** ROE was 2%, 12%, 7%, 10% and 11% over 2020-2024, which is near or below the ~10% cost of equity for banks. ROIC is not reported. ROA is about 1%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -14% | n/a | 5% | 3% | n/a | 2% | 0% |
| 2021-12-31 | 7% | n/a | 28% | -15% | n/a | 12% | 1% |
| 2022-12-31 | -6% | n/a | 18% | 36% | n/a | 7% | 1% |
| 2023-12-31 | 11% | n/a | 23% | 49% | n/a | 10% | 1% |
| 2024-12-31 | -0% | n/a | 24% | 4% | n/a | 11% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The financial services industry is highly competitive."
  * [2] "Many of these competitors enjoy fewer regulatory constraints and some may have lower cost structures."
  * [3] "These technological advances may diminish the importance of depository institutions and other financial intermediaries in the transfer of funds between parties."
  * [5] "Given the rule’s requirement to share customer information with authorized third parties, some of whom could be non-financial institutions, the rule could result in increased fraud, data misuse, and competition."

### BSX: BOSTON SCIENTIFIC CORP (Health Care, 10-K filed 2025-02-18)

> Our Company the Company is a global developer, manufacturer and marketer of medical devices that are used in a broad range of interventional medical specialties. Our mission is to transform lives through innovative medical solutions that improve the health of patients around the world. As a medical ...

**Overall: none.** Regulatory barriers, some first-or-only products and physician familiarity give the Company some protection. But competition is intense with large rivals, pricing pressure from group purchasers is rising, and ROIC is below the cost of capital throughout. The text and the returns together do not support a moat.

**Returns: below.** ROIC was 4-7% over 2021-2024 and about 0% in 2020, below the 8-10% cost of capital. ROE reached only 9%. Operating margins have improved to 16%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -8% | -1% | -1% | 11% | -0% | -1% | -0% |
| 2021-12-31 | 20% | 10% | 8% | 11% | 4% | 6% | 3% |
| 2022-12-31 | 7% | 13% | 5% | 7% | 5% | 4% | 2% |
| 2023-12-31 | 12% | 16% | 11% | 13% | 7% | 9% | 5% |
| 2024-12-31 | 18% | 16% | 11% | 16% | 7% | 9% | 5% |

* **intangibles = 1.** Regulatory approvals (PMA, FDA clearances) and first or only positions in some products (Watchman leader, AXIOS, Intracept) give some protection, but the company says no single patent or asset is material and rivals are large.
  * [1] "AXIOSTM Stents and Electrocautery Enhanced Delivery Systems, the first, and currently only stent systems in the U.S. indicated for endoscopic drainage of pancreatic pseudocysts"
  * [2] "WATCHMANTM is the first device to offer a non-pharmacologic alternative to oral anti-coagulants that has been studied in a randomized clinical trial and is the leading device in percutaneous LAAC globally."
  * [3] "our IntraceptTM Intraosseous Nerve Ablation System, the only FDA-cleared system to treat vertebrogenic pain, a form of chronic low back pain and"
  * [4] "The PMA process is generally more detailed, lengthier and more expensive than the 510(k) process."
* **switching_costs = 1.** Physician familiarity and a platform compatible with the company's catheters suggest some stickiness, but the text does not state strong lock-in.
  * [5] "We also compete on ease of use, comparative effectiveness, reliability and physician familiarity."
  * [6] "iLabTM Ultrasound Imaging Systems with Polaris Software, designed to enhance the diagnosis and treatment of blocked vessels and other heart disorders, compatible with our full line of imaging catheters,"
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "we are also required to compete on the basis of price, value, reliability and efficiency."
  * [8] "We encounter significant competition across our product lines and in each market in which we sell our products and solutions, some from companies that may have greater financial, sales and marketing resources than we do."
  * [9] "In addition, large group purchasing organizations, hospital networks and other buying groups are important to our business and represent a substantial portion of our net sales."
  * [10] "we believe that no single patent, trade secret, trademark, intellectual property asset or license is material in relation to our business as a whole."

### CVS: CVS HEALTH Corp (Health Care, 10-K filed 2025-02-12)

> Overview the Company, together with its subsidiaries (collectively, “the Company,” the “Company,” “we,” “our” or “us”), is a leading health solutions company building a world of health around every consumer it serves and connecting care so that it works for people wherever they are. As of December ...

**Overall: none.** The business is large and integrated, with a PBM and pharmacy scale position and a regulated insurance licence. The text describes a highly competitive, price-driven market with annual contracts and low entry barriers in parts of the business. Returns are near the cost of capital, so the evidence does not support a moat.

**Returns: near.** ROIC was 9%, 9%, 5%, 9% and 5% over the five years, around or below the 8-10% cost of capital. ROE was 6-11%, near the 10% benchmark for an insurer. Margins are thin and declining.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 5% | 5% | 3% | 5% | 9% | 11% | 3% |
| 2021-12-31 | 9% | 5% | 3% | 5% | 9% | 11% | 3% |
| 2022-12-31 | 10% | 2% | 1% | 4% | 5% | 6% | 2% |
| 2023-12-31 | 11% | 4% | 2% | 3% | 9% | 11% | 3% |
| 2024-12-31 | 4% | 2% | 1% | 2% | 5% | 6% | 2% |

* **cost_advantage = 1.** Scale in PBM volume and about 27% retail pharmacy share suggest purchasing leverage, but the text does not state a cost advantage, and the retail footprint is being cut.
  * [3] "the Company’s PBM filled or managed 1.9 billion prescriptions on a 30-day equivalent basis."
  * [4] "dispensed approximately 27.2% of total retail pharmacy prescriptions in the U.S."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [7] "The health care benefits industry is highly competitive, primarily due to a large number of for-profit and not-for-profit competitors, competitors’ marketing and pricing and a proliferation of competing products, including new products that are continually being introduced into the marketplace."
  * [8] "Because of the low barriers of entry into the primary care business and the ability of physicians to own primary care centers and/or also be medical directors for their own centers, competition for growth in existing and expanding markets is not limited to large competitors with substantial financial resources."
  * [9] "The retail pharmacy business is highly competitive."
  * [10] "For Commercial Insured plans, contracts containing the pricing and other terms of the relationship are generally established in advance of the policy period and typically have a duration of one year."
  * [11] "Minimum MLR requirements and similar actions further limit the level of margin the Company can earn in its Insured Commercial products while leaving the Company exposed to medical costs that are higher than those reflected in its pricing."

### GEHC: GE HealthCare Technologies Inc. (Health Care, 10-K filed 2025-02-13)

> the Company (“the Company,” the “Company,” “our,” or “we”) is a trusted partner and leading global healthcare solutions provider, innovating medical technology, pharmaceutical diagnostics, and integrated, cloud-first AI-enabled solutions, services, and data analytics. We have approximately 53,000 ...

**Overall: narrow.** The installed base and recurring service and consumables give some stickiness, but the industry is highly competitive with several large rivals, and IP is not material. Returns are only moderately above cost of capital, which supports a narrow moat at best based on installed-base lock-in.

**Returns: near.** ROIC of 12-15% over the period is moderately above a 8-10% cost of capital, with operating margins of 12-16% that have declined slightly. ROE is rising on leverage after the spin-off.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-12-31 | n/a | 16% | 13% | 8% | 15% | 14% | n/a |
| 2022-12-31 | 4% | 14% | 10% | 10% | 12% | 15% | 7% |
| 2023-12-31 | 7% | 12% | 8% | 9% | 13% | 19% | 5% |
| 2024-12-31 | 1% | 13% | 10% | 8% | 15% | 26% | 6% |

* **switching_costs = 1.** Large installed base with recurring consumables, service contracts and proprietary parameters tied to monitoring fleets creates real but moderate lock-in; customers can still choose other vendors at replacement.
  * [1] "We have a large, global installed base of medical imaging, ultrasound, and patient monitoring systems."
  * [2] "Our Patient Monitoring business includes proprietary parameters and complementary consumables as well as original equipment manufacturers’ parameters that are integrated into our monitoring fleet, of which a significant portion represents recurring revenue streams."
  * [3] "Both our consumables and services provide our customers with ongoing clinical impact and protect their capital investment while providing us with recurring revenue streams."
  * [4] "Our equipment sales representatives partner closely with their service sales counterparts to position both equipment contracts and long-term maintenance agreements along with system upgrades and software as a service (“SaaS”) agreements."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The global medical technology industry is highly competitive and includes global and regional participants of all sizes that can vary by product line."
  * [8] "In the industries we serve, we believe our primary global competitors include Siemens Healthineers, Philips Healthcare, Canon, Mindray, and United Imaging, among others."
  * [9] "we do not consider any single IP asset or group of assets to be of material importance to any segment or to the business as a whole"

### MRNA: Moderna, Inc. (Health Care, 10-K filed 2025-02-21)

> the Company is a leader in the creation of the field of messenger RNA (mRNA) medicine. Through the advancement of mRNA technology, we are reimagining how medicines are made and transforming how we treat and prevent disease for everyone. By working at the intersection of science, technology and ...

**Overall: none.** The patent estate and platform know-how give some protection, but the company faces direct mRNA competition, shows no switching costs or network effects, and sells into a seasonal, fragmented market. Pandemic-era profits have vanished and recent returns are deeply negative, so there is no durable moat.

**Returns: below.** Returns swung from a pandemic-driven ROIC of 64% in 2022 to about -25% and -31% in 2023-24 as revenue fell 64% and then 53%. They were not stable or consistent, and the recent losses are well below the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | >100% | -95% | -93% | >100% | n/a | -40% | -17% |
| 2021-12-31 | >100% | 72% | 66% | 72% | n/a | >100% | 76% |
| 2022-12-31 | 4% | 49% | 43% | 24% | 64% | 50% | 33% |
| 2023-12-31 | -64% | -62% | -69% | -56% | -25% | -29% | -21% |
| 2024-12-31 | -53% | <-100% | <-100% | <-100% | -31% | -29% | -22% |

* **intangibles = 1.** A sizable patent estate with named COVID and RSV patents expiring 2031-2041, plus platform know-how and orphan exclusivities. However, a competing mRNA vaccine exists, the key licences are non-exclusive, and the COVID franchise is shrinking. This is a real but uncertain advantage.
  * [1] "With respect to our IP estate, our solely-owned patent portfolio consists of more than 260 issued or allowed U.S. patents or patent applications and more than 140 granted or allowed patents in jurisdictions outside of the U.S."
  * [2] "Most of the patents and applications (if issued) in our portfolio will not expire until 2033 at the earliest."
  * [3] "Any patent that may issue from our most recently filed patent applications is projected to expire between 2043 and 2044, at the earliest."
  * [4] "We also possess substantial proprietary know-how associated with related manufacturing processes and expertise."
  * [5] "If a product that has orphan designation subsequently receives the first FDA approval for the disease or condition for which it has such designation, the product is entitled to orphan drug exclusivity"
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The continued growth of the mRNA field is leading to increased competitive pressure, including from large and more established pharmaceutical companies."
  * [8] "We largely compete against Pfizer and BioNTech for sales of our COVID vaccine, whose vaccine is also based on mRNA technology."
  * [9] "With respect to our RSV vaccine, we compete against Pfizer and GlaxoSmithKline, who entered the U.S. market prior to us and achieved larger market shares in 2024."
  * [10] "The commercial markets for these vaccines are seasonal and characterized, particularly in the U.S. (our largest market), by a fragmented customer base, unpredictability in orders and seasonality of deliveries."
  * [11] "In markets that we enter after competitors have already introduced a competing product, we may have difficulty achieving market share, as was the case in connection with the launch of our RSV vaccine."
  * [12] "The Cellscript-MRT Agreements are non-exclusive, although Cellscript and MRT are subject to certain time restrictions on granting additional sublicenses for"

### RMD: RESMED INC (Health Care, 10-K filed 2025-08-08)

> the Company. AND SUBSIDIARIES provide reasonable assurance of the safety and effectiveness of most devices. Non-compliance with applicable requirements can result in import detentions, fines, civil and administrative penalties, injunctions, suspensions or losses of regulatory approvals, recall or ...

**Overall: none.** The excerpt covers only the regulatory section and states no moat mechanism. The regulatory clearances are standard for medical devices, and the 510(k) pathway is not a strong barrier. High returns are well above the cost of capital, but returns corroborate the text and do not replace it. With no stated mechanism, the call is no moat on this evidence.

**Returns: above.** ROIC of 24-31% in all five years, with operating margins of 27-33% and ROE of 18-26%. This is well above the cost of capital and consistent, indicating a strong business.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-06-30 | 8% | 28% | 15% | 20% | 31% | 18% | 10% |
| 2022-06-30 | 12% | 28% | 22% | 6% | 28% | 25% | 16% |
| 2023-06-30 | 18% | 27% | 21% | 14% | 26% | 24% | 15% |
| 2024-06-30 | 11% | 28% | 22% | 28% | 24% | 23% | 15% |
| 2025-06-30 | 10% | 33% | 27% | 32% | 28% | 26% | 19% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.

### WST: WEST PHARMACEUTICAL SERVICES INC (Health Care, 10-K filed 2025-02-18)

> General We are a leading global manufacturer in the design and production of technologically advanced, high-quality, integrated containment and delivery systems for injectable drugs and healthcare products. Our products include a variety of primary proprietary packaging, containment solutions, ...

**Overall: narrow.** Regulatory qualification and integration into drug filings give the proprietary segment some stickiness, and sustained high returns corroborate this. But the text names no quantified advantage, the contract segment is price-competitive, and customers are concentrated. Narrow is a fair judgment.

**Returns: above.** ROIC was 19-36% over five years, well above the cost of capital, though it has fallen from the 2021 peak to 19% in 2024 with revenue declining. ROE is also high, at 18-32%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 17% | 19% | 16% | 14% | 22% | 20% | 13% |
| 2021-12-31 | 32% | 27% | 23% | 12% | 36% | 32% | 22% |
| 2022-12-31 | 2% | 25% | 20% | 15% | 30% | 23% | 17% |
| 2023-12-31 | 2% | 23% | 20% | 14% | 25% | 21% | 16% |
| 2024-12-31 | -2% | 20% | 17% | 10% | 19% | 18% | 13% |

* **intangibles = 1.** Exclusive licence from Daikyo for key proprietary products is stated, but patents are generic and the company says it is not dependent on any single IP asset; the licence is held from a third party.
  * [1] "Certain key value-added and proprietary products and processes are exclusively licensed from Daikyo."
* **switching_costs = 1.** Components are built into customers' injectable products and manufacturing lines, and drug products require regulatory approval with the packaging, making changes costly. Contract manufacturing is more price-driven, so the advantage is moderate.
  * [3] "Contract-Manufactured Products components generally are incorporated into our customers’ manufacturing lines for further processing or assembly."
  * [4] "which incorporate our components and other offerings into their injectable products for distribution to the point of care and ultimate end-user, the patient"
  * [5] "products that are intended to be used in the packaging and delivery of pharmaceutical products are subject to both customer acceptance of our products and regulatory approval of the customer’s products following our development period."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Our Contract-Manufactured Products business operates in very competitive markets for its products."
  * [7] "Given the cost pressures they face, many of our customers look to reduce costs by sourcing from low-cost locations."
  * [8] "Our ten largest customers accounted for 43.4% of our consolidated net sales in 2024, and one of these customers, individually accounted for more than 10% of consolidated net sales, at 12.3% or $356.4 million"
  * [9] "We believe, however, that neither our business nor any business segment is wholly dependent on a single intellectual property asset, license, or technology, by itself."
  * [10] "Competition for these components is based primarily on product design and performance, quality, regulatory compliance, and scientific expertise, along with total cost."

### CPRT: COPART INC (Industrials, 10-K filed 2025-09-26)

> Business Corporate Information We were incorporated in California in 1982, became a public company in 1994, and were reincorporated in Delaware in January 2012. Our principal executive offices are located at 14185 Dallas Parkway, Suite 300, Dallas, Texas 75254 and our telephone number is (972) ...

**Overall: narrow.** The text describes a buyer-seller marketplace with a global yard network, seller integration and title know-how, backed by sustained returns well above the cost of capital. Competition is significant, though: IAA, Manheim and others are named, some contracts are tendered on price, and the stated mechanisms are not quantified. That supports a narrow moat.

**Returns: above.** ROIC was 34%, 35%, 28%, 22% and 22% over the five years, well above the cost of capital, with operating margins of 37-42%. The trend is declining but returns remain high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-07-31 | 22% | 42% | 35% | 20% | 34% | 31% | 23% |
| 2022-07-31 | 30% | 39% | 31% | 24% | 35% | 27% | 22% |
| 2023-07-31 | 11% | 38% | 32% | 22% | 28% | 23% | 21% |
| 2024-07-31 | 9% | 37% | 32% | 23% | 22% | 20% | 18% |
| 2025-07-31 | 10% | 37% | 33% | 26% | 22% | 19% | 17% |

* **intangibles = 1.** Title-processing know-how and a single VB3 patent are mentioned, but the claims are modest and the patent dates from 2008. No brand or regulatory exclusivity is described.
  * [1] "Our know-how in the area of title processing is a competitive advantage."
  * [2] "In 2008, we obtained a patent issued by the United States Patent and Trademark Office that covers certain aspects of our virtual bidding auction platform, VB3."
* **switching_costs = 1.** Sellers integrate their systems via APIs and the Company cites high client retention. Contracts are customized, but no term lengths or quantified lock-in are given, and sellers can still tender contracts.
  * [5] "We believe that our integrated technology systems generate improved capacity and financial returns for our clients, resulting in high client retention, and allow us to expand our national supply contracts."
  * [6] "Sellers integrate their systems with the Company G2 Platform via B2B Application Programming Interface “APIs”."
* **network_effects = 1.** A two-sided marketplace: a larger buyer pool (about 1 million members, many international) raises sale prices, which attracts sellers. The mechanism is stated directly, but competitors run similar online auctions.
  * [3] "We believe our virtual auction platform increases the pool of available buyers for each sale, which brings added competition and an increase in the amount that buyers are willing to pay for vehicles."
  * [4] "We maintain a database of approximately 1 million registered members (“buyers”) in the vehicle dismantling and recycling, rebuilding, used vehicle dealer and export industries, as well as members that are a part of the general public, where applicable."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [9] "These national, regional, and local competitors may have established relationships with vehicle sellers and buyers and may have financial resources that are greater than ours."
  * [10] "Under these circumstances, insurance companies will generally award the contract to the company that is willing to pay the highest price for the vehicles."
  * [11] "may purchase salvage vehicles directly from insurance companies, thereby bypassing vehicle remarketing companies like the Company entirely."

### DOV: DOVER Corp (Industrials, 10-K filed 2025-02-14)

> Overview the Company is a diversified global manufacturer and solutions provider delivering innovative equipment and components, consumable supplies, aftermarket parts, software and digital solutions and support services through five operating segments: Engineered Products, Clean Energy & Fueling, ...

**Overall: narrow.** A diversified industrial group with niche leadership, embedded components and an aftermarket installed base that give modest switching costs, and steady above-cost-of-capital returns. Competition is partly price-based, there are many named competitors, and product cycles are short in some segments. Overall this is a narrow moat.

**Returns: above.** ROIC of 12-16% over five years is consistently above the 8-10% cost of capital but not far above it. ROE and net margin in 2024 are inflated by the gain on the ESG sale. Operating margin is stable at 16%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -6% | 14% | 10% | 14% | 13% | 21% | 8% |
| 2021-12-31 | 18% | 16% | 14% | 12% | 16% | 30% | 11% |
| 2022-12-31 | -1% | 16% | 14% | 7% | 14% | 25% | 10% |
| 2023-12-31 | -2% | 16% | 14% | 13% | 13% | 23% | 10% |
| 2024-12-31 | 1% | 16% | 35% | 12% | 12% | 45% | 23% |

* **switching_costs = 1.** Components embedded in customers' systems with switching costs and risks exceeding component cost, plus an installed base driving recurring aftermarket revenue (36%). Real but moderate and varies by segment.
  * [1] "Such components typically serve demanding applications where value-in-use and costs and risks of switching far exceed the cost of the component itself."
  * [2] "our marking and coding equipment, plastics and polymer processing equipment and aluminum can-making equipment all derive a significant share of revenue and profits from sale of consumables, parts and services into their large installed base."
  * [3] "Recurring demand, which includes parts, consumables, services and software, represents approximately 36% of our revenue."
  * [4] "their products are predominantly components or parts of larger equipment and production systems with our products often specified by end customers or regulations"
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "the key competitive factors are customer service, product quality, price and innovation."
  * [8] "The result has been downward pricing trends that can only be mitigated with the continuous introduction of innovative product solutions in a market where product life cycles generally average less than seven years."
  * [9] "While our intellectual property and customer relationships are important to our success, the loss or expiration of any of these rights or relationships is not likely to materially affect our results on a consolidated basis."

### EMR: EMERSON ELECTRIC CO (Industrials, 10-K filed 2025-11-10)

> the Company. ("the Company", "we", "us", "our" or the "Company”) is a global technology and software company that provides innovative solutions for customers in a wide range of end markets around the world. Through its leading automation portfolio, the Company helps process, hybrid and discrete ...

**Overall: narrow.** The Company has probable advantages from installed base, brands and market positions in process automation, but the text gives little direct evidence, describes highly competitive markets, and recent returns are modest and distorted by M&A. Quoted evidence supports only a modest brand/leadership position, so the moat is narrow at most.

**Returns: unclear.** ROIC is not reported. ROE was about 25% in 2021-22, inflated in 2023 by divestiture gains (62%), then fell to 8% and 11% in 2024-25 after large acquisitions and goodwill. Recent returns are near the cost of capital and distorted by portfolio changes.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-09-30 | -23% | n/a | 18% | 25% | n/a | 25% | 10% |
| 2022-09-30 | 7% | n/a | 23% | 19% | n/a | 25% | 11% |
| 2023-09-30 | 10% | n/a | 87% | 2% | n/a | 62% | 34% |
| 2024-09-30 | 15% | n/a | 11% | 17% | n/a | 8% | 5% |
| 2025-09-30 | 3% | n/a | 13% | 15% | n/a | 11% | 5% |

* **intangibles = 1.** Well-known brands (Fisher, Rosemount, Micro Motion) and claimed market leadership, but no figures on share or pricing power, and patents are explicitly not critical.
  * [5] "Management believes the Company has a market leadership position in many of its product lines."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The Company's businesses operate in highly competitive markets."
  * [2] "The Company competes based on product performance, quality, branding, service and/or price across the industries and markets served."
  * [3] "Some competitors have substantially greater sales, assets and financial resources than the Company and the Company also competes with many smaller companies."
  * [4] "While this proprietary intellectual property portfolio is important to the Company in the aggregate, management does not regard any of its segments as being dependent on any single patent, trademark registration or license."

### FTV: Fortive Corp (Industrials, 10-K filed 2025-02-25)

> General the Company is a provider of essential technologies for connected workflow solutions across a range of attractive end-markets. Our strategic segments - Intelligent Operating Solutions, Precision Technologies, and Advanced Healthcare Solutions - include well-known brands with leading ...

**Overall: none.** The text gives only generic leadership claims and brand lists, with no stated mechanism that keeps competitors out. It describes highly competitive markets, price as a competitive factor, and low-cost entrants. Returns on capital are at or below the cost of capital. No moat.

**Returns: below.** ROIC rose from 4% to 8% over five years, at or below the 8-10% cost of capital. ROE is 6-9% apart from a one-off 2020 spike. Operating margins of 19% are decent, but returns on capital are modest, partly because of acquisition goodwill.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | 12% | 33% | 29% | 4% | 19% | 9% |
| 2021-12-31 | 13% | 15% | 11% | 17% | 5% | 6% | 4% |
| 2022-12-31 | 11% | 17% | 13% | 21% | 6% | 8% | 5% |
| 2023-12-31 | 4% | 19% | 14% | 21% | 7% | 9% | 5% |
| 2024-12-31 | 3% | 19% | 13% | 23% | 8% | 8% | 5% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Although our businesses generally operate in highly competitive markets, our competitive position cannot be determined accurately in the aggregate or by segment"
  * [2] "We face increased competition in a number of our served markets as a result of the entry of competitors based in low-cost manufacturing locations, and increasing consolidation in particular markets."
  * [3] "Key competitive factors vary among our businesses and product and service lines, but include the specific factors noted above with respect to each particular business and typically also include price, quality, performance, delivery speed, applications expertise, distribution channel access, service and support, technology and innovation, breadth of product, service and software offerings, and brand name recognition."
  * [4] "Although in aggregate our intellectual property is important to our operations, we do not consider any single patent, trademark, copyright, trade secret, or license to be of material importance to any segment or to the business as a whole."

### MMM: 3M CO (Industrials, 10-K filed 2025-02-05)

> 3M Company was incorporated in 1929 under the laws of the State of Delaware to continue operations begun in 1902. The Company’s ticker symbol is the Company. As used herein, the term “3M” or “Company” includes 3M Company and its subsidiaries unless the context indicates otherwise. In this document, ...

**Overall: narrow.** 3M has a diversified portfolio of brands (Post-it, Scotch, Command) and technology, and normal-year returns and margins are high. The text, though, offers only generic claims of advantage, says no single patent is essential, and describes competition from other technology-oriented firms. Brand and technology breadth plausibly support a narrow moat, but the litigation-driven volatility in returns and the weak textual evidence limit conviction.

**Returns: unclear.** ROIC was 20-21% in 2020-21 and 13% in 2022, then -40% in 2023 because of PFAS and earplug litigation charges, and 29% in 2024. Operating margins are around 20% in normal years. ROE is distorted by small equity, and the litigation shocks make the series uneven.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 0% | 22% | 17% | 21% | 20% | 47% | 12% |
| 2021-12-31 | 10% | 21% | 17% | 17% | 21% | 42% | 13% |
| 2022-12-31 | -26% | 17% | 22% | 15% | 13% | 39% | 12% |
| 2023-12-31 | -6% | -43% | -28% | 21% | -40% | -71% | -14% |
| 2024-12-31 | -0% | 20% | 17% | 3% | 29% | 96% | 9% |

* **intangibles = 1.** 3M claims trademarks, patents and trade secrets give an important advantage, and distributor confidence supports its position. However, no single patent is essential and the claims are general and unquantified.
  * [1] "The Company believes that its trademarks, patents, and trade secrets provide an important competitive advantage in many of its businesses."
  * [5] "Management believes the confidence of wholesalers, retailers, jobbers, distributors and dealers in 3M and its products — a confidence developed through long association with skilled marketing and sales representatives — has contributed significantly to 3M’s position in the marketplace and to its growth."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [2] "In general, no single patent or group of related patents is in itself essential to the Company as a whole or to any of the Company’s business segments."
  * [3] "Most 3M products involve expertise in product development, manufacturing and marketing, and are subject to competition from products manufactured and sold by other technologically oriented companies."

### TDG: TransDigm Group INC (Industrials, 10-K filed 2025-11-12)

> The Company TD Group, through its wholly-owned subsidiary, the Company, is a leading global designer, producer and supplier of highly engineered aircraft components that are critical to the safe and effective operation of nearly all commercial and military aircraft worldwide. Our products are ...

**Overall: narrow.** Proprietary, certified, designed-in parts with a long aftermarket tail support switching costs and regulatory barriers, and returns confirm pricing power. Fragmented niches and limited quantification of the advantages keep this at narrow rather than wide.

**Returns: above.** ROIC rose from 12% to 23% over five years with operating margins of 35-47%, which is well above the 8-10% cost of capital throughout. Acquisition goodwill inflates the capital base, so these returns are even more notable.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-09-30 | -6% | 35% | 14% | 17% | 12% | n/a | 4% |
| 2022-09-30 | 13% | 41% | 16% | 15% | 14% | n/a | 5% |
| 2023-09-30 | 21% | 44% | 20% | 19% | 17% | n/a | 7% |
| 2024-09-30 | 21% | 44% | 22% | 24% | 22% | n/a | 8% |
| 2025-09-30 | 11% | 47% | 23% | 21% | 23% | n/a | 9% |

* **intangibles = 1.** Proprietary designs (90% of sales) with FAA/OEM certification requirements create regulatory barriers and pricing power on niche parts. Barriers are stated as only 'may' deter entrants, and individual IP is not critical, so the advantage is moderate.
  * [1] "We estimate that approximately 90% of our net sales for fiscal year 2025 were generated by proprietary products."
  * [4] "The industry’s stringent regulatory, certification and technical requirements and the investments necessary in the development and certification of products may create disincentives for potential new competitors for certain products."
  * [5] "We, and the components we manufacture, are required to be certified by one or more of these entities or agencies, and, in many cases, by individual OEMs, in order to engineer and service parts and components used in specific aircraft models."
  * [7] "This has generally enabled us to price our products to fairly reflect the value we provide and the resources required to do so."
* **switching_costs = 1.** Parts designed into aircraft generate aftermarket revenue for 25-30+ years, and recertifying another supplier is costly and slow. This is a direct, durable mechanism, though stated with hedged language.
  * [2] "Once our parts are designed into and sold on a new aircraft, we generate net sales from aftermarket consumption over the life of that aircraft, which is generally estimated to be approximately 25 to 30 years."
  * [3] "If customers receive products that meet or exceed expectations and performance standards, we believe that they will have a reduced incentive to certify another supplier because of the cost and time of the technical design and testing certification process."
  * [6] "We estimate that approximately 55% of our net sales in fiscal year 2025 were generated from the aftermarket, the vast majority of which come from the commercial and military aftermarkets."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "The niche markets within the aerospace industry that we serve are relatively fragmented and we face several competitors for many of the products and services we provide."
  * [9] "Based on the broad scope of the Company’s product lines, management believes that the loss or expiration of any single intellectual property right would not have a material effect on our consolidated financial statements."

### TXT: TEXTRON INC (Industrials, 10-K filed 2025-02-06)

> the Company. is a multi-industry company that leverages its global network of aircraft, defense, industrial and finance businesses to provide customers with innovative products and services around the world. References to “the Company.,” the “Company,” “we,” “our” and “us” in this Annual Report on ...

**Overall: none.** The company has some regulatory barriers and an aftermarket installed base in aviation and helicopters, but the text offers little quantified advantage. It also owns cyclical, competitive auto-supplier and vehicle businesses, and its returns are only near the cost of capital. This supports no moat.

**Returns: near.** ROE has been about 12-13% for four years with ROA of 5-6% and thin net margins of 6-7%; ROIC isn't reported. Returns are modest and only around the cost of capital, helped by financial leverage.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-02 | -15% | n/a | 3% | 4% | n/a | 5% | 2% |
| 2022-01-01 | 6% | n/a | 6% | 10% | n/a | 12% | 5% |
| 2022-12-31 | 4% | n/a | 7% | 9% | n/a | 12% | 5% |
| 2023-12-30 | 6% | n/a | 7% | 6% | n/a | 13% | 6% |
| 2024-12-28 | 0% | n/a | 6% | 5% | n/a | 12% | 5% |

* **intangibles = 1.** FAA type certification requirements and the FLRAA/V-22 positions create regulatory barriers, but the company itself says no single IP right is material, and brands are not shown to command premium pricing.
  * [1] "For an aircraft to be manufactured and sold, the model must receive a type certificate from the appropriate aviation authority, and each aircraft must receive a certificate of airworthiness."
  * [4] "The FLRAA development contract was awarded to Bell in December 2022."
* **switching_costs = 1.** A 13,000-helicopter installed base and global service networks drive aftermarket lock-in, but the advantage is moderate and not quantified in terms of revenue share.
  * [2] "For both its military programs and its commercial products, Bell provides post-sale support and service for an installed base of approximately 13,000 helicopters."
  * [3] "In support of its family of aircraft, the Company Aviation operates a global network of more than 20 service centers."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Our contracts with the U.S. Government generally may be terminated by the U.S. Government for convenience or if we default in whole or in part by failing to perform under the terms of the applicable contract."
  * [8] "we do not believe that any existing patent, license, trademark or other intellectual property right is of such importance that its loss or termination would have a material adverse effect on our business taken as a whole."

### VRSK: Verisk Analytics, Inc. (Industrials, 10-K filed 2025-02-26)

> Business Our Company the Company is a leading data, analytics, and technology provider serving clients in the insurance ecosystem. Using advanced technologies to collect and analyze billions of records, we draw on unique data assets, insurance industry knowledge, and technological expertise to ...

**Overall: wide.** Proprietary contributory data, regulatory-embedded standard programs and deep workflow integration give high switching costs and intangible advantages, supported by very high margins and returns for five years. Competition exists in newer segments and some customers can build in-house, but the core is durable.

**Returns: above.** ROIC was 12-34% over five years, rising to 34% in 2024, well above the 8-10% cost of capital; operating margins of 37-56% and FCF margins ~31-36% corroborate. ROE inflated by buybacks.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -13% | 42% | 31% | 36% | 14% | 29% | 10% |
| 2021-12-31 | 9% | 37% | 27% | 36% | 12% | 24% | 9% |
| 2022-12-31 | 1% | 56% | 38% | 31% | 20% | 41% | 13% |
| 2023-12-31 | 7% | 42% | 23% | 31% | 22% | 59% | 11% |
| 2024-12-31 | 7% | 44% | 33% | 32% | 34% | >100% | 22% |

* **intangibles = 2.** Proprietary, contributed data assets (36B records), industry-standard forms and loss costs, and state licensing as statistical agent/advisory organization are hard to replicate and have been built over 50 years.
  * [4] "We are the recognized leader in the U.S. for industry-standard insurance programs that help P&C insurers define coverages and issue policies."
  * [7] "We maintain an underwriting database of more than 36.3 billion statistical records, including approximately 9.7 billion commercial lines records and approximately 26 billion personal lines records."
  * [9] "Nearly every property insurer in the U.S. uses our evaluations of community firefighting capabilities to help determine premiums for fire insurance throughout the country."
  * [12] "We are also licensed as a rating, rate service, advisory, or statistical organization under state insurance codes in all 50 states, Puerto Rico, Guam, the U.S. Virgin Islands, and the District of Columbia."
* **switching_costs = 2.** Products are embedded in insurers' rating, underwriting and regulatory reporting workflows; subscription contracts, 100% of top 100 insurers as clients, and estimates 'essential input to rating decisions' make leaving costly.
  * [2] "In 2024, our clients included all of the top 100 property and casualty ("P&C") insurance providers in the U.S. for the lines of P&C services we offer."
  * [3] "We offer our solutions and services primarily through annual subscriptions or long-term agreements, which are typically prepaid (annually and quarterly) and represented approximately 81% of our revenues in 2024."
  * [8] "For most P&C insurers in most lines of business, we believe that our estimates of future costs are an essential input to rating decisions."
  * [13] "A substantial majority of P&C insurance providers in the U.S. use our statistical agent services to report to regulators, and the majority of insurers and reinsurers in the U.S. use our actuarial services and industry-standard insurance programs."
  * [14] "It is very rare that contributors elect not to continue providing us data."
* **network_effects = 1.** Contributory databases and claims data become more valuable with participation, and widespread contractor use of pricing data improves accuracy, but the text states it only partly as a network mechanism.
  * [10] "We estimate that more than 80% of insurance repair contractors and service providers in the U.S. and Canada with computerized estimating systems use our building and repair pricing data."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [15] "In addition, a customer may use its own internal resources rather than engage an outside firm for these services."
  * [16] "In the P&C insurance claims and catastrophe modeling markets, certain products are offered by a number of companies in the areas of catastrophe modeling, repair cost estimating, claims investigative reports, claims fraud analytics, and injury claims analytics."

### AKAM: AKAMAI TECHNOLOGIES INC (Information Technology, 10-K filed 2025-02-24)

> Overview the Company mission is to power and protect life online. Since 1998, the Company has developed and provided solutions for global enterprises to build, secure and accelerate their applications and digital experiences. Our massively distributed global network is comprised of core and ...

**Overall: none.** The text describes an intensely competitive market, price competition, hyperscaler entry and customers who can build in-house. It states no clear moat mechanism. Returns are declining toward the cost of capital, so there is no durable moat.

**Returns: near.** ROIC fell from 15-16% in 2020-21 to 10% in 2024 with falling operating margins (21% to 13%) and slowing growth. It is still near or slightly above the cost of capital but trending down.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 11% | 21% | 17% | 22% | 15% | 14% | 8% |
| 2021-12-31 | 8% | 23% | 19% | 31% | 16% | 15% | 8% |
| 2022-12-31 | 4% | 19% | 14% | 29% | 14% | 12% | 6% |
| 2023-12-31 | 5% | 17% | 14% | 23% | 13% | 12% | 6% |
| 2024-12-31 | 5% | 13% | 13% | 28% | 10% | 11% | 5% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "The market for our solutions is intensely competitive and characterized by rapidly changing technology, evolving industry standards and frequent new product and service innovations."
  * [4] "In addition, existing and potential customers may decide to purchase or develop their own hardware, software or other technology solutions rather than rely on a third-party provider like us."
  * [5] "we anticipate that going forward our compute solutions will increasingly compete with the large so-called “hyper-scaler” cloud computing providers."
  * [6] "We do not believe that the expiration of any particular patent in the near future would be materially detrimental to our business."

### CTSH: COGNIZANT TECHNOLOGY SOLUTIONS CORP (Information Technology, 10-K filed 2025-02-12)

> Overview the Company is one of the world’s leading professional services companies, engineering modern businesses and delivering strategic outcomes for our clients. We help clients modernize technology, reimagine processes and transform experiences so they can stay ahead in today's fast-changing ...

**Overall: none.** IT services is a highly competitive, fragmented market with many large rivals and competition partly on price. The only mechanism is moderate switching costs from embedded client work. Consistently good returns reflect an efficient offshore-heavy model but the text does not establish durable advantages, so no moat.

**Returns: above.** ROIC of 19-23% over five years, consistently above 8-10% cost of capital, with stable operating margins of 13-15%. Growth is low recently.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -1% | 13% | 8% | 17% | 19% | 13% | 8% |
| 2021-12-31 | 11% | 15% | 12% | 12% | 23% | 19% | 12% |
| 2022-12-31 | 5% | 15% | 12% | 12% | 22% | 19% | 13% |
| 2023-12-31 | -0% | 14% | 11% | 10% | 19% | 17% | 12% |
| 2024-12-31 | 2% | 15% | 11% | 9% | 19% | 16% | 12% |

* **switching_costs = 1.** Services are often critical to larger clients and termination requires an extended transition period, a direct but modest statement of switching costs in a competitive market.
  * [1] "The services we provide to our larger clients are often critical to their operations and termination of our services would typically require an extended transition period with gradually declining revenues."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [2] "The markets for our services are highly competitive, characterized by a large number of participants and subject to rapid change."
  * [3] "Our direct competitors include, among others, Accenture, Atos, Capgemini, CGI, Deloitte Digital, DXC Technology, EPAM Systems, Genpact, HCL Technologies, IBM Consulting, Infosys Technologies, Tata Consultancy Services and Wipro."
  * [4] "A loss of a significant client or a few significant clients in a particular segment could materially reduce revenues for that segment."
  * [5] "The principal competitive factors affecting the markets for our services include the provider’s reputation and experience, strategic advisory capabilities, digital services capabilities, performance and reliability, responsiveness to customer needs, financial stability, corporate governance and competitive pricing of services."

### FTNT: Fortinet, Inc. (Information Technology, 10-K filed 2025-02-21)

> Overview the Company is a leader in cybersecurity, driving the convergence of networking and security. Our mission is to secure people, devices and data everywhere. Our integrated platform, the the Company Security Fabric, spans secure networking, unified Secure Access Service Edge (“SASE”) and ...

**Overall: narrow.** Strong returns and proprietary ASICs with an integrated platform give some edge, but the text describes an extremely competitive market with larger rivals, price pressure and no directly stated switching costs or barriers. Proprietary silicon cost/performance advantage supports a narrow moat at most.

**Returns: above.** ROIC is not reported, but operating margins of 19-30%, FCF margins of 32-37%, ROA rising to 21% and ROE of 44-73% indicate strong, consistent profitability, though ROE is inflated by buybacks and small equity.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 20% | 20% | 19% | 37% | n/a | 44% | 12% |
| 2021-12-31 | 29% | 19% | 18% | 36% | n/a | 73% | 12% |
| 2022-12-31 | 32% | 22% | 19% | 33% | n/a | n/a | 14% |
| 2023-12-31 | 20% | 23% | 22% | 33% | n/a | n/a | 17% |
| 2024-12-31 | 12% | 30% | 29% | 32% | n/a | n/a | 21% |

* **cost_advantage = 1.** In-house ASIC design gives performance and efficiency advantages over industry alternatives, but cost advantage vs. competitors is not quantified and manufacturing is outsourced.
  * [2] "Our application-specific integrated circuit (“ASIC”)-based security processing units (“SPUs”) increase the speed, scale, efficiency and value of our solutions while improving user experience, reducing footprint and power requirements."
  * [3] "We develop our own Network Processors, Content Processors and System-on-Chip Application-Specific Integrated Circuits technology in house."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [6] "Several competitors are significantly larger, have greater financial, technical, marketing, distribution, customer support and other resources, are more established than we are, and have significantly better brand recognition."
  * [7] "Based in part on these competitive pressures, we may lower prices or attempt to add incremental features and functionalities to our products."
  * [8] "The markets for our products are extremely competitive and are characterized by rapid technological change."
  * [9] "We have no long-term contracts related to the manufacturing of our ASICs or other components that guarantee any capacity or pricing terms."

### IBM: INTERNATIONAL BUSINESS MACHINES CORP (Information Technology, 10-K filed 2025-02-25)

> the Company (the Company or the company) was incorporated in the State of New York on June 16, 1911, as the Computing-Tabulating-Recording Co. (C-T-R), a consolidation of the Computing Scale Co. of America, the Tabulating Machine Co. and The International Time Recording Co. of New York. Since that ...

**Overall: none.** The company has a large installed base and enterprise incumbency in mission-critical systems, which gives some stickiness. But it faces hundreds of competitors, competes partly on price, and is in segments where hyperscalers, Accenture and others are strong. Growth is low and the text offers no quantified advantage, so there is at most a thin moat, and the evidence does not support one.

**Returns: unclear.** ROIC is not reported. ROE of 24-34% in most years (8% in 2022) is inflated by a small equity base and leverage, including a financing arm. Net margins are about 10%, and revenue growth is low at 1-6%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | n/a | 10% | 28% | n/a | 27% | 4% |
| 2021-12-31 | 4% | n/a | 10% | 19% | n/a | 29% | 4% |
| 2022-12-31 | 6% | n/a | 3% | 15% | n/a | 8% | 1% |
| 2023-12-31 | 2% | n/a | 12% | 21% | n/a | 34% | 6% |
| 2024-12-31 | 1% | n/a | 10% | 20% | n/a | 24% | 4% |

* **switching_costs = 1.** Incumbency in mission-critical infrastructure with enterprises suggests embedded relationships and some stickiness, but it is not quantified (no contract terms or retention data) and the consulting and software markets are contested.
  * [2] "deep incumbency in mission-critical infrastructure"
  * [3] "We differentiate from other providers by bringing together incumbency with enterprises, deep expertise in technology, industries and business processes, a broad go-to-market reach including an ecosystem of partners and alliances, and a strong focus on innovation."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "Across our business segments, we recognize hundreds of competitors worldwide and as we execute our hybrid cloud and AI strategy, we are regularly exposed to new competitors."
  * [6] "Our principal methods of competition are: technology innovation; performance; price; quality; brand; our breadth of capabilities, products and services; talent; client relationships and trust; the ability to deliver business value to clients; and service and support."
  * [7] "Further, cloud service providers are leveraging innovation in technology and service delivery to compete with traditional providers and to offer additional routes to market for server and storage systems."
  * [8] "Companies with which we have strategic partnerships in some areas may be competitors in other areas."

### PTC: PTC INC. (Information Technology, 10-K filed 2025-11-21)

> siness Our Business the Company is a global software company headquartered in Boston, Massachusetts. We employ over 7,000 people and support more than 30,000 customers globally. We primarily serve customers in the following industry verticals: • Industrials • Federal, Aerospace and Defense • ...

**Overall: narrow.** The text states only modest switching costs (recurring revenue, product data embedded in engineering workflows) with no quantified lock-in. Competition is against well-resourced rivals (Siemens, Dassault, Autodesk). Returns well above the cost of capital for five years, and rising, corroborate a narrow moat built on embedded PLM/CAD data, but the text does not support wide.

**Returns: above.** ROIC rose from 11% to 22% over five years, above the 8-10% cost of capital throughout, with operating margins of 21-36% and strong FCF margins. This is consistent with sticky enterprise software.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-09-30 | 24% | 21% | 26% | 19% | 11% | 27% | 12% |
| 2022-09-30 | 7% | 23% | 16% | 22% | 14% | 14% | 7% |
| 2023-09-30 | 8% | 22% | 12% | 28% | 16% | 10% | 4% |
| 2024-09-30 | 10% | 26% | 16% | 32% | 16% | 13% | 6% |
| 2025-09-30 | 19% | 36% | 27% | 31% | 22% | 21% | 11% |

* **switching_costs = 1.** Mostly recurring subscription revenue (95%) and the text's claim of higher retention, with product data embedded in engineering workflows, suggest meaningful switching costs. However, the text does not directly quantify retention or lock-in.
  * [1] "Our business is based on a subscription model and 95% of our 2025 revenue is recurring in nature."
  * [2] "Compared to a perpetual license model, our subscription model naturally drives higher customer engagement and retention and provides better business predictability."
  * [3] "A product data foundation is the backbone of AI-driven transformation."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "For enterprise CAD and PLM solutions, we compete with large established companies including Autodesk, Dassault Systèmes SA, and Siemens AG."
  * [5] "For our SLM products, we compete with enterprise software companies such as Oracle, SAP, IFS AB, Microsoft, and Salesforce, and with companies that offer point solutions."

### SMCI: Super Micro Computer, Inc. (Information Technology, 10-K filed 2025-02-25)

> Our Company We are a Silicon Valley-based provider of total IT solutions which address demanding workloads from the enterprise and cloud to the intelligent edge. We deliver rack-scale solutions optimized for various workloads, including artificial intelligence (“AI”) and high-performance computing ...

**Overall: none.** The text describes a competitive, price- and speed-driven server market with large rivals and low-cost ODMs, and no stated mechanism that keeps customers. Recent high returns come from AI demand and fast execution, not a documented durable advantage, and earlier years show near-cost-of-capital returns.

**Returns: above.** ROIC rose from 9-11% in FY2020-21 to 20%, 34% and 32% in FY2022-24, well above cost of capital recently, but operating margins are thin (3-11%) and FCF is volatile and negative in several years. Returns reflect an AI demand boom, with a short history of high returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-06-30 | -5% | 3% | 3% | -2% | 9% | 8% | 5% |
| 2021-06-30 | 7% | 3% | 3% | 2% | 11% | 10% | 5% |
| 2022-06-30 | 46% | 6% | 5% | -9% | 20% | 23% | 10% |
| 2023-06-30 | 37% | 11% | 9% | 9% | 34% | 38% | 19% |
| 2024-06-30 | >100% | 8% | 8% | -17% | 32% | 31% | 17% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The market for our products is highly competitive, rapidly evolving and subject to new technological developments, changing customer needs and new product introductions."
  * [2] "In recent years, we have experienced increased competition from original design manufacturers ("ODMs”) that benefit from their scale and very low-cost manufacturing and are increasingly offering their own branded products."
  * [3] "However, most of our competitors have longer operating histories, significantly greater resources, greater name recognition and deeper market penetration."
  * [4] "In addition, it is possible that new competitors could emerge and acquire significant market share."
  * [8] "We compete primarily with large vendors of x86-based general purpose servers and components."

### TRMB: TRIMBLE INC. (Information Technology, 10-K filed 2025-04-25)

> Business the Company is a leading technology solutions provider that enables office and field professionals to connect their workflows and asset lifecycles to drive a more productive, sustainable future. With a focus on the industries that build, move, and feed the world, the comprehensive depth ...

**Overall: none.** There are some sources of advantage, namely embedded software workflows, positioning technology and patents, and an early-stage marketplace. None is quantified or strongly proven. Competition is stated to include price and large established players. Returns on capital are below or near the cost of capital, which does not corroborate a moat.

**Returns: below.** ROIC was 5-9% over five years, at or below the cost of capital and falling to 5% in the latest year. ROE of 29% in the latest year comes from a one-off gain (41% net margin) and is not meaningful. Operating margins of 12-15% are moderate.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-01 | -4% | 13% | 12% | 20% | 7% | 12% | 6% |
| 2021-12-31 | 16% | 15% | 13% | 19% | 9% | 13% | 7% |
| 2022-12-30 | 0% | 14% | 12% | 9% | 8% | 11% | 6% |
| 2023-12-29 | 3% | 12% | 8% | 15% | 6% | 7% | 4% |
| 2025-01-03 | -3% | 13% | 41% | 14% | 5% | 29% | 16% |

* **intangibles = 1.** A large patent portfolio on positioning and related technologies is cited, but only in general terms with no expiry dates or named key patents. The brands are not shown to command pricing power.
  * [5] "We hold over 1,000 unique issued and enforceable patents covering key technology areas, including precision GNSS, optical and inertial positioning solutions, AI and machine learning, IoT, cloud computing, laser scanning, 3D modeling, point cloud processing, augmented reality, and many others."
* **switching_costs = 1.** The transportation management system and the integration into customer IT infrastructure imply embedded workflows. No contract terms, retention figures or switching-cost evidence are given.
  * [3] "The transportation management system serves as a central hub from which the core operations of transportation organizations are managed, data is stored and analyzed, and mission-critical business processes are automated."
  * [4] "In addition to cloud-hosted solutions, we also integrate our applications and services directly into the customer’s IT infrastructure."
* **network_effects = 1.** The text explicitly describes a network effect, and the Transporeon marketplace has meaningful daily volume. However, the effect is stated as a strategy rather than demonstrated, and it applies to a small part of the business.
  * [1] "This network effect also means that the willingness of developers, partners, or end users to engage increases as the number of network participants grows, which further enhances the platform experience and end-user value."
  * [2] "our Transporeon business creates a marketplace for shippers, forwarders, carriers, and retailers to connect online and digitize their end-to-end transportation management processes."
  * [7] "With the amount of data we generate every day—up to approximately 200,000 transports and dock scheduling appointments get executed and booked on the Transporeon platform daily—we are able to provide data insights and market benchmarks"
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "we increasingly compete with large established companies that offer similar systems across all industries."
  * [9] "We compete principally on the basis of interoperability, domain expertise, customer support and service, price, innovative product offerings, quality, and the completeness of our solutions."
  * [10] "Major competitors in this segment are typically survey instrument companies that provide software-driven 3D measurement and imaging solutions."

### TYL: TYLER TECHNOLOGIES INC (Information Technology, 10-K filed 2025-02-19)

> DESCRIPTION OF BUSINESS the Company. (“the Company” or “Company”) is a leading provider of integrated software and technology management solutions for the public sector. Our solutions empower local, state, and federal government entities to create smarter, safer, and stronger communities. We offer ...

**Overall: none.** Sticky mission-critical software with very low churn gives some switching-cost protection. But competition is intense and partly on price, RFPs are required, there are no patents, and ROIC is below the cost of capital, so a durable moat is not established.

**Returns: below.** ROIC was 10% in 2020, then 5-7% for 2021-2024 (7% in 2024), below or at the cost of capital. This is partly distorted by the large NIC acquisition and goodwill. Operating margins of 11-15% are modest.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 3% | 15% | 17% | 30% | 10% | 11% | 8% |
| 2021-12-31 | 43% | 11% | 10% | 21% | 6% | 7% | 4% |
| 2022-12-31 | 16% | 12% | 9% | 19% | 5% | 7% | 3% |
| 2023-12-31 | 5% | 11% | 9% | 18% | 5% | 6% | 4% |
| 2024-12-31 | 10% | 14% | 12% | 28% | 7% | 8% | 5% |

* **switching_costs = 1.** Mission-critical government systems with about 2% annual client turnover, auto-renewing maintenance contracts and 84% recurring revenue show real switching costs, though contract terms are short (one to three years) and RFPs are required.
  * [1] "We have historically experienced very low client turnover (approximately 2% annually), and recurring revenues continue to grow as the installed client base increases."
  * [2] "Most maintenance contracts automatically renew unless the client or the Company gives notice of termination prior to expiration."
  * [3] "We have a large recurring revenue base from subscription-based services and maintenance and support, which generated revenues of $1.8 billion, or 84% of total revenues, in 2024."
  * [4] "Our ability to offer an integrated system of applications for several offices or departments is often a competitive advantage."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Many of these competitors are smaller companies that may offer less expensive solutions than ours."
  * [8] "We compete on a variety of factors, including price, service, name recognition, reputation, technological capabilities, and the ability to configure products and services to address the individual requirements of the client."
  * [9] "Governmental units often are required to seek competitive proposals through a request for proposal process, and some prospective clients use consultants to assist them with the proposal and vendor selection process."
  * [10] "We generally do not rely on patents."
  * [11] "We also compete with national firms, some of which have greater financial and technical resources than we do, including Oracle Corporation, Infor, SAP AG, Workday, Inc., CentralSquare Technologies, Thomson Reuters Corporation, Motorola Solutions, Inc., Axon Enterprise, Inc., and Constellation Software, Inc."

### ECL: ECOLAB INC. (Materials, 10-K filed 2025-02-21)

> . ​ General Development of Business. ​ the Company was incorporated as a Delaware corporation in 1924. Our fiscal year is the calendar year ending December 31. International subsidiaries are included in the consolidated financial statements on the basis of their U.S. GAAP (accounting principles ...

**Overall: narrow.** The company is a leading player in concentrated industrial water and hygiene markets, with a service-heavy model, embedded dosing equipment, trademarks and regulatory registrations. Each source is modest and the text does not quantify any of them. Returns were only near the cost of capital until 2024. That supports a narrow moat at best, and the evidence is thin.

**Returns: near.** ROIC was 8-10% in 2020-2023 (near the cost of capital) and rose to 15% in 2024 with the operating margin at 18%. Returns were not consistently well above the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -6% | 12% | -10% | 12% | 8% | -16% | -6% |
| 2021-12-31 | 8% | 13% | 9% | 11% | 9% | 17% | 6% |
| 2022-12-31 | 11% | 11% | 8% | 8% | 8% | 15% | 5% |
| 2023-12-31 | 8% | 13% | 9% | 11% | 10% | 18% | 6% |
| 2024-12-31 | 3% | 18% | 13% | 12% | 15% | 25% | 10% |

* **intangibles = 1.** Trademarks and many EPA/FDA registrations help, but the text only states these generally and does not quantify an advantage; regulation raises entry hurdles moderately.
  * [2] "Trademarks related to the Company, Nalco and 3D TRASAR, which collectively are material to all of our reportable segments."
  * [5] "We maintain several hundred product registrations with the U.S. Environmental Protection Agency (“EPA”)."
  * [6] "Within the Global Healthcare & Life Sciences reportable segment, the Healthcare business competes geographically with companies primarily focused on a smaller range of product categories, with few globally scaled competitors."
* **switching_costs = 1.** Customers embed dispensing and monitoring systems and use multiple segments' offerings, implying some stickiness, but no contract terms or retention figures are given.
  * [3] "Through this strategy and our varied product and service mix, one customer may utilize the offerings of several of our operating segments."
  * [4] "Institutional develops various digital monitoring and chemical dispensing systems which are used by our customers to efficiently and safely dispense our cleaners and sanitizers"
* **efficient_scale = 1.** Industrial markets are led by a few large companies, suggesting an oligopoly, but this does not show that new entrants would earn no return.
  * [1] "In general, the markets in which the businesses in our Global Industrial reportable segment compete are led by a few large companies, with the rest of the market served by smaller entities focusing on more limited geographic regions or a smaller subset of products and services."
* Rated 0: network_effects, cost_advantage.
* Against a moat:
  * [8] "Second, we have numerous smaller regional or local competitors which focus on more limited geographies, product lines and/or end-use customer segments."
  * [9] "While Specialty’s customer base has broadened significantly over the years, Specialty’s business remains largely dependent upon a limited number of major QSR chains and franchisees and large food retail customers."

### NUE: NUCOR CORP (Materials, 10-K filed 2025-02-27)

> usiness. Overview the Company, a Delaware corporation incorporated in 1958, and its affiliates (“the Company,” the “Company,” “we,” “us” or “our”) manufacture steel and steel products. The Company also produces and procures ferrous and non-ferrous materials primarily for use in its steel ...

**Overall: none.** Steel is a cyclical, price-competitive commodity business, and the company says it primarily competes on price and service. It has a lower-cost EAF structure, vertical integration into scrap and DRI, leadership in some niches and trade protection, but these are modest and returns are cyclical, falling to near the cost of capital in 2024. No clear durable moat.

**Returns: near.** ROE was very high in 2021-2022 (53%, 45%), but fell to 22% in 2023 and 10% in 2024; ROA fell to 6%. 2020 ROE was 7%. Returns are cyclical, with a peak in the pricing upcycle and a sharp decline as prices normalize; ROIC is not reported. Low leverage means ROE is not inflated.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -11% | n/a | 4% | 6% | n/a | 7% | 4% |
| 2021-12-31 | 81% | n/a | 19% | 13% | n/a | 53% | 30% |
| 2022-12-31 | 14% | n/a | 18% | 20% | n/a | 45% | 26% |
| 2023-12-31 | -16% | n/a | 13% | 14% | n/a | 22% | 13% |
| 2024-12-31 | -11% | n/a | 7% | 3% | n/a | 10% | 6% |

* **intangibles = 1.** Trade remedies (AD/CVD orders) offer some protection from imports, and one JV is the only North American producer of HSLA beams, but this is a narrow niche and trade protection is policy-dependent. Brands are not quantified.
  * [9] "There are currently 128 AD/CVD orders in place on core steel product lines made by the Company, and sunset reviews are typically held with respect to specific exporters and products every five years."
  * [10] "the Company-Yamato is the only North American producer of high-strength, low-alloy beams."
* **cost_advantage = 1.** EAF mills have lower capital and fixed costs than integrated producers, plus vertical integration into scrap and DRI and a regional logistics advantage for micro mills. But competition among EAF peers and imports is on price, so the advantage is modest.
  * [6] "EAF-based steel producers, such as the Company, are sensitive to increases in scrap prices but tend to have lower capital and fixed operating costs compared with large integrated steel producers."
  * [7] "This makes them suitable for regional markets and enables us to serve these markets with a logistics cost advantage relative to competitors operating from further away."
  * [8] "Having a significant portion of our raw materials supply under our control minimizes risk associated with the global sourcing of raw materials"
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "These markets are highly competitive with many domestic and foreign firms participating, and, as a result of this highly competitive environment, we find that we primarily compete on price and service."
  * [2] "Firms typically compete on price and geographic proximity to the sources of scrap metal."
  * [3] "Fluctuations in the supply of steel from non U.S. competitors are a source of significant volatility in our primary markets."
  * [4] "In each of our product lines, we face competition from well capitalized domestic and international providers offering similar products and services."
  * [5] "The majority of these contracts are firm, fixed-price contracts that are, in most cases, competitively bid against other suppliers."

### SW: Smurfit Westrock plc (Materials, 10-K filed 2025-03-07)

> Overview the Company was created in July 2024 as a strategic combination between the Company Kappa Group plc (“the Company Kappa”), one of the leading integrated corrugated packaging manufacturers in Europe, with a large-scale pan-regional presence in Latin America, and WestRock Company ...

**Overall: none.** The business is a large, integrated producer in largely standardized, price-competitive markets with many rivals and short-term orders. Cost and integration advantages are modest, and returns have dropped below the cost of capital. No durable moat.

**Returns: below.** ROIC fell from 29% (2022) to 16% (2023) to 4% in 2024, with operating margin dropping to 5% after the merger. The latest year is below the cost of capital and the trend is cyclical and volatile.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-12-31 | n/a | 12% | 8% | 4% | 29% | 20% | n/a |
| 2023-12-31 | -10% | 11% | 7% | 5% | 16% | 14% | 6% |
| 2024-12-31 | 75% | 5% | 2% | 0% | 4% | 3% | 1% |

* **switching_costs = 1.** Some one-to-three-year exclusive supply contracts, but most sales are short-term orders at market rates, so switching costs are modest.
  * [5] "These agreements typically provide that the customer will source a specified percentage of its total product requirements from us on an exclusive basis, and typically allow for raw material cost movements to be passed through to end pricing."
* **cost_advantage = 1.** Vertical integration, low-cost kraftliner mills in Europe and owned forest plantations in Latin America give a modest cost position, but the advantage is limited and competitors are also integrated.
  * [6] "Our three kraftliner mills, located at Piteå (Sweden), Facture (France), and Nettingsdorfer (Austria), are among the largest and, we believe, the lowest cost kraftliner mills in Western Europe."
  * [7] "In addition, we own approximately 301,000 acres of forest plantations in Latin America, which support mill operations in addition to propagating trees for planting."
  * [8] "We believe this security of supply to be a significant benefit at times when recovered fiber is in short supply."
* Rated 0: intangibles, network_effects, efficient_scale.
* Against a moat:
  * [1] "As containerboard and corrugated containers products are largely standardized products, competition is primarily based on price."
  * [2] "The industries in which we operate are highly competitive, and no single company dominates any of those industries."
  * [3] "Most sales of converted products are made on the basis of short-term orders for specified volumes at market rates."
  * [4] "In the corrugated packaging and folding carton markets, we compete with a significant number of national, regional and local packaging suppliers in North America."

### CSGP: COSTAR GROUP, INC. (Real Estate, 10-K filed 2025-02-20)

> In this Report, the words “we,” “our,” “us,” “the Company Group,” or the “Company” refer to the Company and its direct and indirect wholly owned subsidiaries. This Report also refers to our websites, but information contained on those sites is not part of this Report. Overview the Company Group is ...

**Overall: narrow.** The proprietary database, leading marketplace traffic and subscription base give some durable advantages in commercial real estate, but competition is intense, residential rivals are incumbents, and returns are modest and declining. A narrow moat is justified by the data and marketplace leadership.

**Returns: near.** ROIC was 9-12% in 2020-2022, fell to 7% in 2023 and 0% in 2024, with FCF turning negative in 2024. Returns are near or below the cost of capital, partly depressed by large acquisition goodwill, and are not clearly above it.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 19% | 17% | 14% | 29% | 9% | 5% | 4% |
| 2021-12-31 | 17% | 22% | 15% | 18% | 12% | 5% | 4% |
| 2022-12-31 | 12% | 21% | 17% | 20% | 12% | 6% | 5% |
| 2023-12-31 | 12% | 11% | 15% | 15% | 7% | 5% | 4% |
| 2024-12-31 | 11% | 0% | 5% | -7% | 0% | 2% | 2% |

* **intangibles = 1.** A decades-built proprietary CRE database, the largest research team and leading traffic-based brands are stated as differentiators, but patents are not material and data itself has limited copyright protection.
  * [1] "We own and operate leading online marketplaces for real estate in the U.S. and the U.K., based on the numbers of unique visitors and site visits per month"
  * [2] "offer the most comprehensive commercial real estate database available; and have the largest commercial real estate research department in the industry."
  * [3] "We have spent more than 35 years building and acquiring databases of real estate information"
* **switching_costs = 1.** Auto-renewing subscriptions of at least one year and workflow integration give modest stickiness, but terms are short and no deep lock-in is described.
  * [5] "Our services are typically distributed to our clients under subscription-based license agreements that typically renew automatically, a majority of which have a term of at least one year."
* **network_effects = 1.** Professionals proactively report listings to the Company's tools and its marketplaces lead on traffic, which suggests a two-sided audience/listing effect, though it is not explicitly stated as a network effect.
  * [4] "Because of the importance commercial real estate professionals place on our data and our prominent position in the industry, many of these professionals routinely take the initiative and proactively report available space and transactions through our online tool"
  * [6] "LoopNet, the most trafficked commercial real estate marketplace"
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "Many of our competitors, especially with respect to our residential real estate businesses, are incumbent players that may have greater name recognition, and resources."
  * [9] "We regard the rights protected by our patents as valuable to our business, but do not believe that our business is materially dependent on any single patent or portfolio of patents as a whole."
  * [10] "We license our services under license agreements that grant our clients non-exclusive, non-transferable rights."
  * [11] "Generally, as markets for real estate-focused online marketplaces, information, and analytics develop, additional competitors (including companies that could have greater access to data, financial, product development, technical, analytic, or marketing resources than we do) may enter a market, and competition may intensify"

### MAA: MID AMERICA APARTMENT COMMUNITIES INC. (Real Estate, 10-K filed 2025-02-07)

> usiness. Overview the Company, an S&P 500 company, is a multifamily-focused, self-administered and self-managed real estate investment trust, or REIT. We own, operate, acquire and selectively develop apartment communities primarily located in the Southeast, Southwest and Mid-Atlantic regions of the ...

**Overall: none.** Intense competition, concessions and low rents, easily available housing alternatives and a lack of any stated moat source, together with modest returns, point to no moat. Scale and capital access help but are not durable advantages.

**Returns: near.** ROE was 4-10% and ROA 2-6%, with no ROIC reported. For a REIT these are near or modestly below the cost of capital and show no excess returns. Margins are decent but returns are only modest.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | n/a | 15% | 36% | n/a | 4% | 2% |
| 2021-12-31 | 6% | n/a | 30% | 35% | n/a | 9% | 5% |
| 2022-12-31 | 14% | n/a | 32% | 38% | n/a | 10% | 6% |
| 2023-12-31 | 6% | n/a | 26% | 37% | n/a | 9% | 5% |
| 2024-12-31 | 2% | n/a | 24% | 35% | n/a | 9% | 5% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Competition for new residents is generally intense across all of our markets."
  * [2] "Competing apartment communities can use concessions or lower rents to obtain temporary competitive advantages."
  * [3] "The owners of competing apartment communities may have greater resources than us, and the managers of these apartment communities may have more experience than our management."
  * [4] "Moreover, single-family rental housing, manufactured housing, condominiums and the new and existing home markets provide housing alternatives to potential residents of apartment communities."

### PSA: Public Storage (Real Estate, 10-K filed 2025-02-24)

> Business Cautionary Statement Regarding Forward-Looking Statements This Annual Report on Form 10-K contains forward-looking statements within the meaning of the Private Securities Litigation Reform Act of 1995. Forward-looking statements include statements relating to our 2025 outlook and all ...

**Overall: narrow.** The Company has the leading brand and scale in a fragmented industry, with cost and marketing advantages that are real but modest, since small operators compete locally with month-to-month leases. Consistently high returns corroborate some advantage, but the text describes a competitive, low-barrier market, so the moat is narrow at best.

**Returns: above.** ROE was 15-22% in most years and 45% in 2022, with a one-off spike; it is 21% in 2024. ROIC is not reported. Net and free cash flow margins are very high, and ROA is 10-13% excluding 2022. The REIT's returns are well above the roughly 5-7% cost of capital for real estate, and not much distorted by leverage.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | n/a | 47% | 70% | n/a | 15% | 12% |
| 2021-12-31 | 17% | n/a | 57% | 74% | n/a | 22% | 13% |
| 2022-12-31 | 22% | n/a | >100% | 75% | n/a | 45% | 25% |
| 2023-12-31 | 8% | n/a | 48% | 72% | n/a | 21% | 12% |
| 2024-12-31 | 4% | n/a | 44% | 67% | n/a | 21% | 10% |

* **intangibles = 1.** The Company claims the most recognized brand in self-storage and a brand-driven advantage in acquiring customers, but this is management belief and the market is fragmented with local competitors offering similar services.
  * [1] "We are the industry leading owner of self-storage properties, with the most recognized brand in the self-storage industry, including our ubiquitous orange color."
  * [2] "We believe our the Company® brand awareness, as well as the innovative improvements we have made to the customer experience described below, provide us with a competitive advantage in acquiring and retaining customers relative to other self-storage operators."
* **cost_advantage = 1.** Management says scale gives materially higher cash flow per square foot and lower admin costs, and online customer acquisition is centralized. However, the Company owns only about 9% of the market and the claim is not quantified.
  * [3] "However, we believe that the economies of scale inherent in this business allow us to operate self-storage facilities at a materially higher level of cash flow per square foot than other operators without our scale."
  * [4] "The size and scope of our operations have enabled us to achieve high operating margins and a low level of administrative costs relative to revenues through the centralization of many functions"
  * [5] "Approximately 83% of our move-ins in 2024 were sourced through our website"
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [6] "Ownership and operation of self-storage facilities is highly fragmented."
  * [7] "The high level of ownership fragmentation in the industry is partially attributable to the relative simplicity of managing a local self-storage facility, such that small-scale owners can operate self-storage facilities at a basic level of profitability without significant managerial or operational infrastructure."
  * [8] "As a result, competition is significant and affects the occupancy levels, rental rates, rental income, and operating expenses of our facilities."
  * [9] "We are the largest owner of self-storage facilities in the United States (“U.S.”), with physical presence in most major markets and 40 states."
  * [10] "which offer storage spaces for lease on a month-to-month basis, for personal and business use."

### AEP: AMERICAN ELECTRIC POWER CO INC (Utilities, 10-K filed 2025-02-13)

> GENERAL Overview and Description of Major Subsidiaries the Company was incorporated under the laws of the State of New York in 1906 and reorganized in 1925. It is a public utility holding company that owns, directly or indirectly, all of the outstanding common stock of its public utility ...

**Overall: narrow.** The moat is the regulated exclusive franchise, which is the type of efficient scale that lasts for decades. Returns are only modest and capped by regulators, and the Generation & Marketing business is competitive. The company has a narrow moat, because regulation limits returns and carries regulatory risk, though the franchise itself is durable.

**Returns: near.** ROIC is 4-5% and ROE 9-11% across five years, in line with regulated utility cost of capital (5-7%) and authorized ROEs of about 9.5-10.5%. These are the capped returns of a regulated franchise.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | 20% | 15% | 26% | 5% | 11% | 3% |
| 2021-12-31 | 13% | 20% | 15% | 23% | 5% | 11% | 3% |
| 2022-12-31 | 17% | 18% | 12% | 27% | 5% | 10% | 3% |
| 2023-12-31 | -3% | 19% | 12% | 26% | 4% | 9% | 2% |
| 2024-12-31 | 4% | 22% | 15% | 35% | 5% | 11% | 3% |

* **intangibles = 1.** Exclusive franchises and regulatory approvals protect the territories, but this is mostly captured as regulated efficient scale; the franchises have varying expiration dates and the company only calls them 'adequate'.
  * [3] "In some cases, these franchises provide the utility with the exclusive right to provide electric service within a specific territory."
  * [4] "The year of expiration of each NRC Operating License is 2034 for Unit 1 and 2037 for Unit 2."
* **efficient_scale = 2.** Regulated, franchised monopoly territories with cost-of-service rates, and transmission with FERC formula rates, mean that new entrants cannot compete in the service areas. The regulated returns are capped, but the franchise is durable.
  * [1] "Other than AEGCo, the Company’s vertically integrated public utility subsidiaries hold franchises or other rights that effectively grant the exclusive ability to provide electric service in various municipalities and regions in their service areas."
  * [2] "These sales are made at rates approved by the state utility commissions of the states in which they operate, and in some instances, approved by the FERC, and are not subject to competition from other vertically integrated public utilities."
  * [3] "In some cases, these franchises provide the utility with the exclusive right to provide electric service within a specific territory."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [5] "the Company’s vertically integrated public utility subsidiaries compete with self-generation and with distributors of other energy sources, such as natural gas, fuel oil, renewables and coal, within their service areas."
  * [6] "The Company Generation & Marketing segment subsidiaries face competition for the sale of available power, capacity and ancillary services."

### FE: FIRSTENERGY CORP (Utilities, 10-K filed 2025-02-27)

> The Companies the Company and its subsidiaries are principally involved in the transmission, distribution, and generation of electricity. the Company electric operating companies comprise one of the nation’s largest investor-owned electric systems, serving over six million customers in the Midwest ...

**Overall: narrow.** The moat is the regulated franchise: distribution has no competition and rates are set to recover costs plus an allowed return. Returns are capped and only near the cost of capital, and transmission faces some competition. This is a narrow moat.

**Returns: near.** ROIC is about 5-6% for all five years, which is in line with the 5-7% cost of capital for regulated utilities. ROE ranges from 4% to 16% and was 9% in 2024. FCF is negative because of heavy capital spending.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -2% | 20% | 10% | -11% | 6% | 15% | 2% |
| 2021-12-31 | 3% | 16% | 12% | 3% | 5% | 16% | 3% |
| 2022-12-31 | 12% | 15% | 3% | -1% | 5% | 4% | 1% |
| 2023-12-31 | 3% | 18% | 9% | -15% | 5% | 10% | 2% |
| 2024-12-31 | 5% | 18% | 7% | -8% | 5% | 9% | 2% |

* **efficient_scale = 1.** Regulated monopoly distribution territories with no competition and cost-based formula rates form a durable franchise. Some transmission competition under Order 1000 and allowed ROEs that are often not earned (Ohio 4.7% actual vs 10.8% allowed) limit the advantage.
  * [1] "Within the Company Electric Companies' distribution business, there generally is no competition for electric distribution service in the Electric Companies’ respective service territories in Ohio, Pennsylvania, West Virginia, Maryland, New Jersey and New York."
  * [2] "the Company accounts for the effects of regulation through the application of regulatory accounting to the Electric Companies and the Transmission Companies as their rates are established by third-party regulators with the authority to set binding rates that are cost-based and can be charged to and collected from customers."
  * [3] "The segment’s revenues are primarily derived from forward-looking formula rates, pursuant to which the revenue requirement is updated annually based on a projected rate base and projected costs, which is subject to an annual true-up based on actual rate base and costs."
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [4] "This has resulted in additional competition to build transmission facilities in the respective service territories while also allowing the opportunity to seek to build facilities in non-incumbent service territories."

### WEC: WEC ENERGY GROUP, INC. (Utilities, 10-K filed 2025-02-21)

> A. INTRODUCTION In this report, when we refer to "the Company Group," "the Company," "us," "we," "our," or "ours," we are referring to the Company and all of its subsidiaries. The term "utility" refers to the regulated activities of the electric and natural gas utility companies, while the term ...

**Overall: narrow.** The moat is the regulated franchise, with exclusive territories and cost pass-through, which is durable but yields only capped, utility-level returns near the cost of capital. This supports a narrow moat.

**Returns: near.** ROIC is 5-6% across all five years, which is near or slightly above the roughly 5-7% cost of capital for regulated utilities. ROE is 11-12%, which is steady but is regulator-set and levered.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | 24% | 17% | -9% | 6% | 11% | 3% |
| 2021-12-31 | 15% | 21% | 16% | -4% | 6% | 12% | 3% |
| 2022-12-31 | 15% | 20% | 15% | -7% | 6% | 12% | 3% |
| 2023-12-31 | -7% | 21% | 15% | 34% | 5% | 11% | 3% |
| 2024-12-31 | -3% | 25% | 18% | 37% | 6% | 12% | 3% |

* **intangibles = 1.** Exclusive service territories via permits and franchises act as regulatory licences that bar competitors, though they are standard utility rights and returns are regulator-capped.
  * [1] "Our electric utilities are authorized to provide retail electric service in designated territories in the state of Wisconsin, as established by indeterminate permits and boundary agreements with other utilities, and in certain territories in the state of Michigan pursuant to franchises granted by municipalities."
* **efficient_scale = 1.** Regulated monopoly service territories with a 1.68 million customer base and a stake in a transmission company; the market supports only one provider, and cost recovery mechanisms protect returns.
  * [1] "Our electric utilities are authorized to provide retail electric service in designated territories in the state of Wisconsin, as established by indeterminate permits and boundary agreements with other utilities, and in certain territories in the state of Michigan pursuant to franchises granted by municipalities."
  * [3] "We have an approximately 60% equity interest in ATC (an electric transmission company operating in Illinois, Michigan, Minnesota, and Wisconsin)."
  * [4] "Prudently incurred fuel and purchased power costs are recovered dollar-for-dollar from our Michigan retail electric customers."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [5] "The majority of our sales for resale are sold into an energy market operated by MISO at market rates based on the availability of our generation and market demand."
