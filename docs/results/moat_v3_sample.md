# Moat tagger v3: set `sample`

Generated 2026-11-02 18:03 by `scripts/moat_pilot.py`, set `sample`, model `claude-sonnet-5-5`, prompts `claude-sonnet-5-5__ca6bf1d1a__high`. One call per company over Item 1 (business) and five years of financials, prompt v3: quotes checked word for word against the text, each moat source rated from the verified quotes only (`auto_researcher.llm.moat`).

* Companies: 50 (47 rated); chunks read 47 (Item 1 cut at 16 chunks for 0).
* Quotes returned: 411; kept 407; not found in the text 4 (1.0%); duplicates 0; over the cap 0; unparseable replies 0.
* Kept quotes by kind: intangibles 82, switching_costs 49, network_effects 15, cost_advantage 41, efficient_scale 29, against 191.
* Rating fixes (citations of missing or "against" quotes dropped, uncited ratings lowered to 0): 2.
* Model time: 6 min (8 s per company); API cost $2.18; served by another model 0.

## Ratings

0 = none or generic, 1 = specific but modest, 2 = strong and durable.

| Ticker | Sector | Overall | intang | switch | network | cost | eff.scale | Against quotes | Returns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MTCH | Communication Services | narrow | 1 | 0 | 1 | 0 | 0 | 4 | above |
| TTWO | Communication Services | ? | ? | ? | ? | ? | ? | 0 | ? |
| ABNB | Consumer Discretionary | narrow | 1 | 0 | 1 | 0 | 0 | 4 | above |
| CCL | Consumer Discretionary | none | 1 | 0 | 0 | 1 | 1 | 3 | below |
| CMG | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 4 | above |
| GM | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 4 | below |
| MAR | Consumer Discretionary | narrow | 1 | 1 | 1 | 0 | 0 | 3 | above |
| CAG | Consumer Staples | none | 0 | 0 | 0 | 0 | 0 | 5 | below |
| DG | Consumer Staples | narrow | 0 | 0 | 0 | 1 | 1 | 3 | near |
| MO | Consumer Staples | narrow | 2 | 0 | 0 | 0 | 0 | 2 | above |
| TSN | Consumer Staples | none | 1 | 0 | 0 | 0 | 0 | 5 | below |
| EOG | Energy | none | 0 | 0 | 0 | 1 | 0 | 5 | above |
| BBT | Financials | none | 0 | 0 | 0 | 0 | 1 | 4 | below |
| COF | Financials | none | 0 | 0 | 0 | 0 | 0 | 4 | near |
| EG | Financials | none | 0 | 0 | 0 | 0 | 0 | 6 | near |
| MCO | Financials | narrow | 2 | 1 | 0 | 0 | 0 | 2 | above |
| NTRS | Financials | none | 0 | 0 | 0 | 0 | 0 | 2 | near |
| PYPL | Financials | narrow | 1 | 0 | 1 | 0 | 0 | 4 | above |
| SCHW | Financials | narrow | 1 | 0 | 0 | 1 | 0 | 2 | near |
| ABBV | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 | above |
| DGX | Health Care | ? | ? | ? | ? | ? | ? | 0 | ? |
| GILD | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 | unclear |
| HUM | Health Care | none | 0 | 0 | 0 | 1 | 0 | 4 | near |
| IDXX | Health Care | narrow | 1 | 1 | 0 | 0 | 0 | 3 | above |
| MOH | Health Care | none | 0 | 0 | 0 | 0 | 0 | 7 | above |
| REGN | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 6 | above |
| ALLE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 5 | above |
| AME | Industrials | none | 1 | 1 | 0 | 0 | 0 | 4 | above |
| CHRW | Industrials | none | 0 | 0 | 1 | 1 | 0 | 6 | above |
| DE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 3 | above |
| LII | Industrials | narrow | 0 | 0 | 0 | 0 | 0 | 4 | above |
| PWR | Industrials | none | 0 | 1 | 0 | 0 | 0 | 4 | near |
| RSG | Industrials | narrow | 1 | 1 | 0 | 1 | 1 | 3 | near |
| UAL | Industrials | none | 0 | 0 | 1 | 0 | 0 | 5 | near |
| CDW | Information Technology | none | 0 | 0 | 0 | 0 | 0 | 6 | above |
| DELL | Information Technology | none | 0 | 0 | 0 | 1 | 0 | 4 | above |
| FICO | Information Technology | narrow | 2 | 1 | 1 | 0 | 0 | 4 | above |
| NTAP | Information Technology | narrow | 1 | 1 | 0 | 0 | 0 | 6 | above |
| ORCL | Information Technology | narrow | 0 | 1 | 0 | 0 | 0 | 4 | above |
| PANW | Information Technology | narrow | 0 | 1 | 0 | 0 | 0 | 5 | near |
| VRSN | Information Technology | wide | 2 | 0 | 0 | 0 | 2 | 3 | above |
| ALB | Materials | none | 0 | 0 | 0 | 1 | 0 | 5 | below |
| MOS | Materials | none | 1 | 0 | 0 | 1 | 0 | 4 | below |
| PKG | Materials | none | 0 | 0 | 0 | 0 | 0 | 6 | above |
| ESS | Real Estate | none | 0 | 0 | 0 | 0 | 1 | 2 | near |
| IRM | Real Estate | none | 0 | 1 | 0 | 0 | 0 | 3 | near |
| O | Real Estate | ? | ? | ? | ? | ? | ? | 0 | ? |
| CNP | Utilities | narrow | 1 | 0 | 0 | 0 | 1 | 4 | near |
| LNT | Utilities | narrow | 0 | 0 | 0 | 0 | 1 | 2 | near |
| PEG | Utilities | narrow | 0 | 0 | 0 | 0 | 2 | 3 | near |

Not tagged, because the located Item 1 is another part of the 10-K: TTWO (starts mid-sentence); DGX ('Critical Accounting Policies' in the first 10,000 characters); O ('CONSOLIDATED BALANCE SHEETS' in the first 10,000 characters).

| Sector | none | narrow | wide |
| --- | ---: | ---: | ---: |
| Communication Services | 0 | 1 | 0 |
| Consumer Discretionary | 3 | 2 | 0 |
| Consumer Staples | 2 | 2 | 0 |
| Energy | 1 | 0 | 0 |
| Financials | 4 | 3 | 0 |
| Health Care | 2 | 4 | 0 |
| Industrials | 4 | 4 | 0 |
| Information Technology | 2 | 4 | 1 |
| Materials | 3 | 0 | 0 |
| Real Estate | 2 | 0 | 0 |
| Utilities | 0 | 3 | 0 |

## Against the hand check of v1

The reviewer judged the v1 tags; each verdict implies a target for the new tags ("too low" is met by any higher call, "too high" by any lower one, "agree" by the same call). 47 companies with the right Item 1 text in both runs. The reviewed v1 run is favoured by construction (the verdicts were written about its exact calls); a second run of the same v1 prompt shows how many targets a run misses by chance.

| Tags | Overall calls on target | Source ratings on target |
| --- | ---: | ---: |
| v1, the reviewed run | 40 (85%) | 215 of 235 (91%) |
| v1, run 2 (same prompt) | 36 (77%) | 207 of 235 (88%) |
| v3 (with financials) | 39 (83%) | 199 of 235 (85%) |

Companies whose overall call changed or misses its target:

| Ticker | v1 | Verdict on v1 | v3 | Returns (v3) | On target |
| --- | --- | --- | --- | --- | --- |
| CCL | narrow | high | none | below | yes |
| CMG | none | low | none | above | no |
| MCO | wide | agree | narrow | above | no |
| NTRS | none | low | none | near | no |
| IDXX | narrow | low | narrow | above | no |
| AME | none | low | none | above | no |
| LII | none | low | narrow | above | yes |
| FICO | narrow | low | narrow | above | no |
| NTAP | none | agree | narrow | above | no |
| IRM | narrow | agree | none | near | no |

## Notes

* **What changed from v1.** The prompt adds the company's financial history and a step judging returns on capital before the overall call (`ONE_CALL_SYSTEM_V2`); sources are still rated from quotes only. Item 1 texts that start mid-sentence or hold MD&A or financial statement headings early are not tagged (`item1_problem`).
* **The financial history is point-in-time:** SEC XBRL company facts filed on or before the 10-K's filing date (the 10-K's own figures included), the latest value known then for each fiscal year (`data.sec_fundamentals.build_pit_table`). ROIC is operating income after a 21% tax over average equity + debt - cash; ratios on negative or near-zero capital are shown as n/a.

## Review sheet

For each company: what it does (the opening of Item 1, name hidden), the overall call, then each source rated above 0 with the model's reason and the quotes it cites, and the quotes against a moat. Sources rated 0 are listed by name only.

### MTCH: Match Group, Inc. (Communication Services, 10-K filed 2025-02-27)

> Who we are the Company, through its portfolio companies, is a leading provider of digital technologies designed to help people make meaningful connections. Our global portfolio of brands includes Tinder ® , Hinge ® , the Company ® , Meetic ® , OkCupid ® , PairsTM, Plenty Of Fish ® , Azar ® , BLK ® ...

**Overall: narrow.** The brands, especially Tinder and Hinge, and the local network effects of large user pools support a narrow moat, and consistently high returns corroborate this. Against it, the market is fragmented with low switching costs and multi-homing, there is no dominant brand, app stores take a large revenue share, and growth is slowing. The advantages are therefore narrow rather than wide.

**Returns: above.** ROIC was 14-29% over five years, above a cost of capital of about 8-10% every year. Operating margins were 16-31% and free cash flow margins were solid. Growth has slowed to 3% in 2024.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 17% | 31% | 7% | 31% | 15% | n/a | 3% |
| 2021-12-31 | 25% | 29% | 9% | 28% | 29% | n/a | 7% |
| 2022-12-31 | 7% | 16% | 11% | 15% | 14% | n/a | 8% |
| 2023-12-31 | 6% | 27% | 19% | 25% | 25% | n/a | 15% |
| 2024-12-31 | 3% | 24% | 16% | 25% | 23% | n/a | 12% |

* **intangibles = 1.** Tinder and Hinge are strong consumer brands, with Tinder the fastest-scaling in its category, and Meetic and Pairs hold leading regional positions. Users cite brand as very important. But the market is fragmented with no dominant brand, and the claims are not quantified.
  * [1] "Tinder® was launched in 2012 and has since risen to scale and popularity faster than any other service in the online dating category."
  * [2] "Meetic is the most recognized dating app for singles over age 35 in Europe."
  * [5] "Pairs. The PairsTM app was launched in 2012 and is a leading provider of online dating services in Japan, with a presence in Taiwan and South Korea."
* **network_effects = 1.** The text says successful brands depend on large, active user communities, which implies a local network effect in dating apps. But the mechanism is stated only generally, and the market is fragmented with multi-homing.
  * [3] "Generally, successful brands depend on large, active communities of users, strong algorithmic filtering technology, and awareness of successful usage among similar users."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The industry for social connection apps is competitive and has no single, dominant brand globally."
  * [8] "As a result, the market for social connection apps is fragmented, and no single service has been able to effectively serve all of those seeking social connections."
  * [9] "We pay Apple and Google a meaningful share of the revenue we receive from in-app transactions."
  * [10] "Our brands compete with each other and with third-party businesses on brand characteristics, service features, and business model"

### TTWO: TAKE TWO INTERACTIVE SOFTWARE INC (Communication Services, 10-K filed 2025-05-20)

> for additional discussion. Impairments During the fiscal year ended March 31, 2025, we recognized Goodwill impairment charges of $3,545.2, representing a partial impairment related to one of our reporting units, and we recognized impairment charges of $137.0 for acquisition-related Developed Game ...

Not tagged: the located Item 1 starts mid-sentence.

### ABNB: Airbnb, Inc. (Consumer Discretionary, 10-K filed 2025-02-13)

> Overview We operate a global marketplace, where hosts offer guests stays and experiences on our platform. We are a community based on connection and belonging—a community that was born in 2007 when two hosts welcomed three guests to their San Francisco home, and has since grown to over 5 million ...

**Overall: narrow.** The marketplace has real scale, a strong brand and a two-sided network. Very high returns corroborate this. The text, though, shows low switching costs, many alternatives for hosts and guests, intense competition and regulatory risk. That supports a narrow moat, not a wide one.

**Returns: above.** ROIC was 59% in 2024, with operating margins of 15-23% and FCF margins around 40% from 2021 to 2024. ROE was volatile, and 2023 was inflated by a tax benefit. Returns are well above the cost of capital after the 2020 pandemic loss.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -30% | <-100% | <-100% | -23% | n/a | n/a | -44% |
| 2021-12-31 | 77% | 7% | -6% | 38% | n/a | -9% | -3% |
| 2022-12-31 | 40% | 21% | 23% | 41% | n/a | 37% | 13% |
| 2023-12-31 | 18% | 15% | 48% | 39% | n/a | 70% | 26% |
| 2024-12-31 | 12% | 23% | 24% | 41% | 59% | 32% | 13% |

* **intangibles = 1.** Brand is said to reduce reliance on performance marketing, which suggests a real brand pull, but it is not quantified and patents and trademarks are described only in general terms.
  * [3] "the strength of the the Company brand and our communications strategy allows us to be less reliant on performance marketing."
* **network_effects = 1.** A two-sided marketplace with over 5 million hosts and 2 billion guest arrivals implies a host and guest network effect. The text says little about it, and hosts can list on other platforms.
  * [1] "Our technology platform powers our two-sided marketplace and enables our global network of hosts and guests."
  * [2] "has since grown to over 5 million hosts who have welcomed over 2 billion guest arrivals in almost every country and region across the globe"
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "We compete to attract and retain hosts to and on our platform to list their homes and experiences, as hosts have a range of options for doing so."
  * [5] "We compete to attract and retain guests to and on our platform, as guests have a range of options to find and book accommodations and experiences."
  * [6] "We operate in a highly competitive environment."
  * [7] "For example, New York City passed regulations in 2023, which resulted in a de facto ban of short-term rental activities."

### CCL: CARNIVAL CORP (Consumer Discretionary, 10-K filed 2025-01-27)

> Business . A. Overview I. Summary the Company was incorporated in Panama in 1974 and the Company was incorporated in England and Wales in 2000. the Company and the Company operate a dual listed company (“DLC”), whereby the businesses of the Company and the Company are combined through a number of ...

**Overall: none.** The company is the scale leader in a concentrated industry with strong brands and owned destinations, but the text shows no switching costs or network effects, competition from land vacations and rivals, and the financials show returns at or below the cost of capital. Scale and brands give at most a modest advantage that is not yet proven in returns.

**Returns: below.** Returns were deeply negative from 2020 to 2022 (pandemic), ROIC 4% in 2023 and 8% in 2024, only now reaching the cost of capital. There is no history of high returns in the five years.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-11-30 | -73% | <-100% | <-100% | <-100% | -20% | -45% | -21% |
| 2021-11-30 | -66% | <-100% | <-100% | <-100% | -16% | -58% | -18% |
| 2022-11-30 | >100% | -36% | -50% | -54% | -10% | -63% | -12% |
| 2023-11-30 | 77% | 9% | -0% | 5% | 4% | -1% | -0% |
| 2024-11-30 | 16% | 14% | 8% | 5% | 8% | 24% | 4% |

* **intangibles = 1.** Several well-known brands with long heritage and leading positions (World's Most Popular Cruise Line, AIDA, P&O UK), but the text gives no pricing power evidence and brands compete with land vacations.
  * [2] "the Company Cruise Line is “The World’s Most Popular Cruise Line®”"
  * [3] "AIDA is the most recognized brand in the German cruise market."
  * [4] "P&O Cruises (UK) is Britain’s largest cruise line and its heritage can be traced back over 185 years."
* **cost_advantage = 1.** Largest global cruise company with scale purchasing, owned private port destinations and berthing access, which give some cost and experience advantages, but no quantified cost gap versus Royal Caribbean or MSC.
  * [5] "We are the largest global cruise company, and among the largest leisure travel companies"
  * [6] "leveraging our enterprise scale through corporate-wide agreements"
  * [7] "to secure preferential berth access to third-party ports, we enter into berthing agreements and commitments."
  * [8] "Celebration Key will be able to accommodate up to four of our cruise ships simultaneously."
* **efficient_scale = 1.** Four players hold about 80% of capacity, an oligopoly, but the market is contestable and competes with land vacations; newbuild capacity is added and not capped.
  * [1] "we, along with our principal cruise competitors Royal Caribbean Group, Norwegian Cruise Line Holdings, Ltd. and MSC Cruises, represented approximately 80% of the cruise industry capacity."
* Rated 0: switching_costs, network_effects.
* Against a moat:
  * [10] "We compete with land-based vacation alternatives throughout the world, such as hotels, resorts (including all-inclusive resorts), theme parks, organized tours, casinos, vacation ownership properties, and other internet-based alternative lodging sites."
  * [11] "Our travel agents’ relationships are generally not exclusive"
  * [12] "We offer a variety of special promotions, including early booking, past guest recognition and travel agent programs."

### CMG: CHIPOTLE MEXICAN GRILL INC (Consumer Discretionary, 10-K filed 2025-02-05)

> General the Company., a Delaware corporation, together with its subsidiaries (“the Company,” “we,” “us,” or “our”) owns and operates the Company restaurants, which feature a relevant menu of burritos, burrito bowls (a burrito without the tortilla), quesadillas, tacos, and salads. We strive to ...

**Overall: none.** The returns are excellent, but the text describes a highly competitive, low-barrier industry and gives no stated mechanism such as switching costs, network effects, cost advantage or a quantified brand edge. The only support is a brand that the text asserts but does not substantiate. Returns alone cannot establish a moat when the market is competitive and price-driven.

**Returns: above.** ROIC rose from 18% to 56% over five years, far above the cost of capital, with operating margin up to 17%. ROE is also high, though partly lifted by buybacks.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 7% | 5% | 6% | 5% | 18% | 19% | 6% |
| 2021-12-31 | 26% | 11% | 9% | 11% | 45% | 30% | 10% |
| 2022-12-31 | 14% | 13% | 10% | 10% | 54% | 39% | 13% |
| 2023-12-31 | 14% | 16% | 12% | 12% | 55% | 45% | 16% |
| 2024-12-31 | 15% | 17% | 14% | 13% | 56% | 46% | 18% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The fast-casual, quick-service, and casual dining segments of the restaurant industry are highly competitive with respect to, among other things, taste, price, food quality and presentation, service, location, convenience, brand reputation, and cleanliness and ambience of each restaurant."
  * [2] "Among our main competitors are restaurant formats that claim to serve higher quality ingredients without artificial flavors, colors and preservatives, and that serve food quickly and at a reasonable price."
  * [3] "Certain key ingredients are purchased from a small number of suppliers."
  * [4] "Our competition includes a variety of restaurants in each of these segments, including locally-owned restaurants, as well as national and regional chains."

### GM: General Motors Co (Consumer Discretionary, 10-K filed 2025-01-28)

> the Company Company (sometimes referred to as we, our, us, ourselves, the Company, the Company or the Company) was incorporated as a Delaware corporation in 2009. We design, build and sell trucks, crossovers, cars and automobile parts and provide software-enabled services and subscriptions ...

**Overall: none.** A cyclical, capital-intensive automaker that competes on price, design and incentives against many rivals. The strong U.S. truck position is not enough to offset low ROIC, so there is no moat.

**Returns: below.** ROIC of 4-6% over five years is below the cost of capital; ROE of 9-17% is boosted by leverage from the finance arm. Margins are thin and the business is cyclical.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -11% | 5% | 5% | 9% | 4% | 13% | 3% |
| 2021-12-31 | 4% | 7% | 8% | 6% | 5% | 17% | 4% |
| 2022-12-31 | 23% | 7% | 6% | 4% | 5% | 14% | 4% |
| 2023-12-31 | 10% | 5% | 6% | 6% | 4% | 15% | 4% |
| 2024-12-31 | 9% | 7% | 3% | 5% | 6% | 9% | 2% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The principal factors that determine consumer vehicle preferences in the markets in which we operate include overall vehicle design, price, quality, available options, safety, reliability, fuel economy or range and functionality."
  * [2] "The level of incentives is dependent upon the level of competition in the markets in which we operate and the level of demand for our products."
  * [3] "Certain fleet transactions, particularly sales to daily rental car companies, are generally less profitable than retail sales to end customers."
  * [4] "Market leadership in individual countries in which we compete varies widely."

### MAR: MARRIOTT INTERNATIONAL INC /MD/ (Consumer Discretionary, 10-K filed 2025-02-11)

> Overview We are a worldwide operator, franchisor, and licensor of hotel, residential, timeshare, and other lodging properties under more than 30 brand names at different price and service points. Consistent with our focus on management, franchising, and licensing, we own or lease very few of our ...

**Overall: narrow.** Asset-light franchisor with a large global brand portfolio, long contracts and a large loyalty program that reinforces owner and guest demand; very high ROIC supports narrow moat. Intense competition from similarly branded chains and Airbnb, plus owner termination rights, keep it from wide.

**Returns: above.** ROIC was 27%, 29%, 27% in 2022-2024 and 14% in 2021, well above cost of capital after the 2020 pandemic trough; asset-light model with negative-ish equity inflates ROE but ROIC is still high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -50% | 1% | -3% | 14% | 1% | -47% | -1% |
| 2021-12-31 | 31% | 13% | 8% | 7% | 14% | >100% | 4% |
| 2022-12-31 | 50% | 17% | 11% | 10% | 27% | >100% | 9% |
| 2023-12-31 | 14% | 16% | 13% | 11% | 29% | n/a | 12% |
| 2024-12-31 | 6% | 15% | 9% | 8% | 27% | n/a | 9% |

* **intangibles = 1.** Large brand portfolio with 17% US share and heavy brand affiliation, but text mostly asserts brand recognition generically and faces many strong rivals.
  * [5] "Based on lodging industry data, we have an approximately 17 percent share of the U.S. hotel market and a four percent share of the hotel market outside the U.S. (based on number of rooms)."
  * [7] "In 2024, approximately 73 percent of U.S. hotel rooms were brand-affiliated."
* **switching_costs = 1.** Long management (20-30 yr) and franchise (10-20 yr) contracts and loyalty-driven repeat business lock in owners and guests, though owners can terminate for poor performance.
  * [1] "Such agreements are generally for initial periods of 20 to 30 years, with options for us to renew for up to 10 or more additional years."
  * [2] "Such agreements are generally for periods of 10 to 20 years."
  * [4] "We believe that our Loyalty Program generates substantial repeat business that might otherwise go to competing properties."
* **network_effects = 1.** Loyalty program with a majority of room nights booked by members and a large system attract owners and guests, a modest two-sided reinforcement; not strongly quantified as network value.
  * [3] "In 2024, 72 percent of our U.S. hotel room nights and 65 percent of our global hotel room nights were booked by Loyalty Program members."
  * [6] "We believe that our brands are attractive to hotel owners seeking a management company, franchise, or other licensing affiliation because of the benefits of our Loyalty Program, centralized reservation systems, marketing programs, and other offerings, and our emphasis on guest service."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "Many of our management agreements also permit the hotel owners to terminate the agreement if we do not meet certain performance metrics, financial returns fail to meet defined levels for a period of time, and we have not cured those deficiencies."
  * [9] "we compete against many other companies with strong brands and guest appeal, including Hilton, IHG Hotels & Resorts, Hyatt, Wyndham Hotels & Resorts, Accor, Choice Hotels, Best Western Hotels & Resorts, and others."
  * [10] "We encounter strong competition in the short-term lodging market from regional, national, and international chains that operate lodging properties or franchise their brands, lodging properties that are not affiliated with a chain, and online platforms, including Airbnb and Vrbo"

### CAG: CONAGRA BRANDS INC. (Consumer Staples, 10-K filed 2025-07-10)

> General Development of Business the Company (the “Company”, “the Company”, “we”, “us”, or “our”), headquartered in Chicago, is one of North America’s leading branded food companies. We combine a 100-year history of making quality food with agility and a relentless focus on collaboration and ...

**Overall: none.** The company owns well-known branded food products, but the text shows no quantified advantage, and competition is intense, with private label and larger rivals. Walmart accounts for 29% of sales, which gives it bargaining power. Returns are below the cost of capital, so the evidence points to no moat.

**Returns: below.** ROIC fell from 10% to 4-5% in fiscal 2023-24 and recovered to 7% in fiscal 2025. That is at or below the 8-10% cost of capital for most of the period. Revenue has declined for two years.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-05-30 | 1% | 19% | 12% | 9% | 10% | 16% | 6% |
| 2022-05-29 | 3% | 14% | 8% | 6% | 7% | 10% | 4% |
| 2023-05-28 | 6% | 9% | 6% | 5% | 5% | 8% | 3% |
| 2024-05-26 | -2% | 7% | 3% | 14% | 4% | 4% | 2% |
| 2025-05-25 | -4% | 12% | 10% | 11% | 7% | 13% | 6% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "We experience intense competition for sales of our food items in our major markets."
  * [2] "Some of our competitors are larger and have greater resources than we have."
  * [3] "Our food items compete with widely advertised, well-known, branded food, as well as private branded and customized food items."
  * [4] "Our largest customer, Walmart, Inc. and its affiliates, accounted for approximately 29% of consolidated net sales for fiscal 2025 and 28% for fiscal 2024 and 2023."
  * [5] "We use many different raw materials, most of which are commodities, to make and package our products."

### DG: DOLLAR GENERAL CORP (Consumer Staples, 10-K filed 2025-03-21)

> ​ General ​ We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025, with the greatest concentration of stores in the southern, southwestern, midwestern and eastern United States. Our first stores ...

**Overall: narrow.** The company has real scale, a dense small-town footprint and a low-cost model, but it competes in a price-driven, intensely competitive market against larger rivals. It has no switching costs, network effects or strong brand. ROIC has dropped to about 10%, so the advantage is modest. A narrow moat rests on its rural density and purchasing scale.

**Returns: near.** ROIC fell from 30% to 10% over five years, and operating margin fell from 11% to 4%. It was well above the cost of capital earlier but is now only near it, and the trend is down.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-29 | 22% | 11% | 8% | 8% | 30% | 40% | 11% |
| 2022-01-28 | 1% | 9% | 7% | 5% | 26% | 37% | 9% |
| 2023-02-03 | 11% | 9% | 6% | 1% | 24% | 41% | 9% |
| 2024-02-02 | 2% | 6% | 4% | 2% | 15% | 27% | 6% |
| 2025-01-31 | 5% | 4% | 3% | 4% | 10% | 16% | 4% |

* **cost_advantage = 1.** The Company claims purchasing power and a low-cost model from its scale and limited assortment, but it concedes that rivals such as Walmart have greater resources and may secure better supplier terms. The advantage appears modest.
  * [1] "Our ability to offer everyday low prices on quality merchandise is supported by our low-cost operating approach and our strategy to maintain a limited number of items per merchandise category, which we believe helps us maintain strong purchasing power."
  * [2] "Purchasing large volumes of merchandise within our focused assortment in each merchandise category allows us to keep our average product costs low, contributing to our ability to offer competitive everyday prices to our customers."
  * [6] "We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025"
* **efficient_scale = 1.** Many stores sit in small towns with limited alternatives, which limits local competition. But the market is crowded with big competitors, so this is only a partial local advantage.
  * [3] "approximately 80% of our stores are located in towns of 20,000 or fewer people."
  * [4] "We believe our ability to effectively deliver both value and convenience allows us to succeed in small markets with limited shopping alternatives, as well as in larger and more competitive markets."
* Rated 0: intangibles, switching_costs, network_effects.
* Against a moat:
  * [7] "We operate in the basic discount consumer goods market, which is highly competitive with respect to price, customers, store location, merchandise quality, assortment and presentation, service offerings, in-stock consistency, customer service, promotional activity, employees, and market share."
  * [8] "Certain of our competitors have greater financial, distribution, marketing and other resources than we do and may be able to secure better arrangements from suppliers than we can."
  * [9] "Competition is intense and we believe it will continue to be so"

### MO: ALTRIA GROUP, INC. (Consumer Staples, 10-K filed 2025-02-26)

> General Development of Business When used in this Annual Report on Form 10-K (“Form 10-K”), the terms “the Company,” “we,” “us” and “our” refer to either (i) the Company and its consolidated subsidiaries or (ii) the Company only and not its consolidated subsidiaries, as appropriate in the context. ...

**Overall: narrow.** The Marlboro and Copenhagen brands, the leading U.S. share, loyalty, and FDA regulatory barriers support a durable brand-based moat. Returns are very high and sustained. Volume decline of about 10% a year in cigarettes and reduced social acceptance mean the moat is eroding in a shrinking market. This supports narrow rather than wide.

**Returns: above.** ROIC rose from 29% to 46-48% over the five years, far above the cost of capital, with operating margins of 42-47% and FCF margins above 30%. This holds despite revenue declining for four years.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 4% | 42% | 17% | 31% | 29% | 97% | 9% |
| 2021-12-31 | -1% | 44% | 10% | 32% | 37% | n/a | 6% |
| 2022-12-31 | -4% | 47% | 23% | 32% | 46% | n/a | 15% |
| 2023-12-31 | -2% | 47% | 33% | 37% | 48% | n/a | 22% |
| 2024-12-31 | -2% | 47% | 47% | 36% | 46% | n/a | 31% |

* **intangibles = 2.** Marlboro has been the top-selling U.S. cigarette brand for 50 years, and the text cites brand loyalty in a market where PM USA is the largest company. The FDA also restricts new entrants. Declining volumes limit how durable this is, but the brand lets the company price well.
  * [1] "PM USA is the largest cigarette company in the United States and substantially all cigarettes are manufactured and sold to customers in the United States."
  * [3] "USSTC is the leading producer and marketer of MST products."
  * [4] "NJOY’s commercialized product portfolio of tobacco and menthol e-vapor products is fully covered by marketing granted orders (“MGO”) from the U.S. Food and Drug Administration (“FDA”)."
  * [5] "The market for tobacco products is highly competitive, characterized by brand recognition and loyalty, with product quality, taste, price, product innovation, marketing, packaging and distribution constituting the significant methods of competition."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Total smokeable products segment’s cigarettes shipment volume in the United States was 68.6 billion units in 2024, a decrease of 10.2% from 2023."
  * [7] "we recognize that the decreasing social acceptance of tobacco usage may impact our ability to attract and retain talent with skills necessary for us to achieve our Vision."
* Fixes: intangibles: dropped citations [2].

### TSN: TYSON FOODS, INC. (Consumer Staples, 10-K filed 2025-11-10)

> GENERAL the Company and its subsidiaries (collectively, the “Company,” “we,” “us,” “our,” “the Company” or “the Company”) (NYSE: the Company) is a world-class food company and recognized leader in protein. Founded in 1935 by John W. the Company, it has grown under four generations of family ...

**Overall: none.** A commodity protein processor with short-term contracts, price-based competition, a concentrated large customer (Walmart) and commodity inputs. Brands and scale offer little demonstrable durable advantage, and recent returns are below the cost of capital.

**Returns: below.** ROIC was 13-14% in fiscal 2021-22 but fell to -1% in 2023 and 3-4% in 2024-25, well below the cost of capital. Returns are cyclical and recently weak.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-10-02 | 9% | 9% | 6% | 6% | 14% | 18% | 9% |
| 2022-10-01 | 13% | 8% | 6% | 2% | 13% | 17% | 9% |
| 2023-09-30 | -1% | -1% | -1% | -0% | -1% | -3% | -2% |
| 2024-09-28 | 1% | 3% | 2% | 3% | 4% | 4% | 2% |
| 2025-09-27 | 2% | 2% | 1% | 2% | 3% | 3% | 1% |

* **intangibles = 1.** Brands like Jimmy Dean and Hillshire Farm and the Cobb breeding stock business exist, but the text gives no market share figures or pricing power evidence. Brand claims are generic.
  * [6] "Through our wholly-owned subsidiary, Cobb-Vantress, we are one of the leading poultry breeding stock suppliers in the world."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Our agreements with customers are generally short-term, primarily due to the nature of our products, industry practices and fluctuations in supply, demand and price for such products."
  * [2] "The principal competitive elements are advertising, consumer and trade promotions, price, product safety and quality, brand identification, innovation, breadth and depth of product offerings, availability of products, customer service and credit terms."
  * [3] "Walmart Inc. accounted for approximately"
  * [4] "The primary raw materials used in our prepared foods operations are commodity-based raw materials, including beef, pork, chicken, turkey, flour, vegetables, cheese, eggs, seasonings and other cooking ingredients."
  * [5] "We do not have facilities of our own to raise cattle but employ cattle buyers located throughout cattle producing areas who visit independent feed yards and public auctions to buy live cattle on the open spot market."

### EOG: EOG RESOURCES INC (Energy, 10-K filed 2025-02-27)

> Business General the Company., a Delaware corporation organized in 1985, together with its subsidiaries (collectively, the Company), explores for, develops, produces and markets crude oil, natural gas liquids (NGLs) and natural gas primarily in major producing basins in the United States of America ...

**Overall: none.** The Company is a price taker in commodity markets with powerful competitors. Its low-cost claim and quality acreage are modest, unquantified advantages. Returns are cyclical and the 2020 loss shows no durable structural edge.

**Returns: above.** ROIC was 21-34% in 2021-2024 but -2% in 2020, showing high cyclicality. Returns follow commodity prices and have fallen to 24% as prices declined.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -37% | -5% | -5% | 45% | -2% | -3% | -2% |
| 2021-12-31 | 69% | 33% | 25% | 47% | 21% | 22% | 13% |
| 2022-12-31 | 38% | 39% | 30% | 43% | 34% | 33% | 19% |
| 2023-12-31 | -6% | 40% | 31% | 47% | 30% | 29% | 18% |
| 2024-12-31 | -2% | 34% | 27% | 51% | 24% | 22% | 14% |

* **cost_advantage = 1.** The Company claims to be a low-cost producer with a large, stacked-pay Delaware Basin position and owned infrastructure, but the claim is not quantified against peers. Resource quality may give a modest advantage.
  * [1] "the Company is focused on being among the highest return and lowest cost producers, committed to strong environmental performance and playing a significant role in the long-term future of energy."
  * [2] "key gathering, processing and transportation infrastructure was added in order to lower operating costs and increase price realizations."
  * [3] "The Delaware Basin consists of approximately 4,800 feet of liquids-rich stacked pay potential offering the Company multiple co-development opportunities throughout its 395,000 net acre position."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [4] "In each case, the price received was based on market prices at that specific sales point or based on the price index applicable for that location."
  * [5] "Conversely, sales of crude oil and condensate and NGLs by the Company are made at unregulated market prices."
  * [6] "Certain of the Company's competitors have financial and other resources substantially greater than those the Company possesses and have established strategic long-term positions or strong governmental relationships in countries or areas in which the Company may seek new or expanded entry."
  * [7] "During 2024, three purchasers each accounted for more than 10% of the Company's total crude oil and condensate, NGLs and natural gas revenues and gathering, processing and marketing revenues."
  * [8] "Average natural gas prices received by the Company for production in the United States resulted in a 26% decrease in 2024, a 63% decrease in 2023, and a 49% increase in 2022"

### BBT: TRUIST FINANCIAL CORP (Financials, 10-K filed 2025-02-25)

> the Company is a purpose-driven financial services company committed to inspiring and building better lives and communities. Headquartered in Charlotte, North Carolina, the Company has leading market share in many of the high-growth markets in the U.S. and offers a wide range of products and ...

**Overall: none.** The company is a large regional bank in an intensely competitive, heavily regulated industry. It has meaningful deposit share, but the text gives no direct moat mechanism, and returns on equity are at or below the cost of equity. The scale and regulatory barriers are shared by all large banks and do not translate into excess returns.

**Returns: below.** ROE was 6%, 9%, 9%, -2% and 7% over five years, which is below or near the roughly 10% cost of equity. 2023 was negative, and the figures show no consistent excess returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | n/a | n/a | n/a | n/a | n/a | 6% | 1% |
| 2021-12-31 | n/a | n/a | n/a | n/a | n/a | 9% | 1% |
| 2022-12-31 | n/a | n/a | n/a | n/a | n/a | 9% | 1% |
| 2023-12-31 | n/a | n/a | n/a | n/a | n/a | -2% | -0% |
| 2024-12-31 | n/a | n/a | n/a | n/a | n/a | 7% | 1% |

* **efficient_scale = 1.** Regional deposit rankings and deposit-share limits on mergers suggest a concentrated industry. But the text does not say that new entrants would earn inadequate returns, and competition is described as intense.
  * [1] "the Company Bank is one of the 10 largest commercial banks in the U.S. and provides banking and trust services for clients through 1,928 offices as of December 31, 2024 and its digital platform."
  * [2] "The market share limitations impose conditions that the acquiring BHC, after and as a result of the acquisition, control no more than 10% of the total amount of deposits of IDIs in the U.S. and no more than 30%, subject to variation by state law, of such deposits in applicable states."
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [3] "The financial services industry is intensely competitive and constantly evolving."
  * [4] "Some also have greater scale, financial and operational resources, investment capacity, product and service offerings, and brand recognition."
  * [5] "Competition affects every aspect of our business, including product and service offerings, rates, pricing and fees, credit limits, and client service."
  * [6] "We expect that competition will only intensify in the future."

### COF: CAPITAL ONE FINANCIAL CORP (Financials, 10-K filed 2025-02-20)

> OVERVIEW General the Company, a Delaware corporation established in 1994 and headquartered in McLean, Virginia, is a diversified financial services holding company with banking and non-banking subsidiaries. the Company and its subsidiaries (the “Company” or “the Company One”) offer a broad array of ...

**Overall: none.** The text describes a highly competitive, price-driven market with many bank and non-bank rivals. No mechanism is stated directly, and recent ROE is at or below the cost of capital. Scale and regulation protect the franchise, but they do not amount to a moat.

**Returns: near.** ROE fell from 20% in 2021 to 13%, 9% and 8% in 2024, and was only 5% in 2020. That is at or below the roughly 10% cost of equity and is volatile. ROIC is not available.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -0% | n/a | 10% | 56% | n/a | 5% | 1% |
| 2021-12-31 | 7% | n/a | 41% | 38% | n/a | 20% | 3% |
| 2022-12-31 | 13% | n/a | 21% | 38% | n/a | 13% | 2% |
| 2023-12-31 | 7% | n/a | 13% | 53% | n/a | 9% | 1% |
| 2024-12-31 | 6% | n/a | 12% | 43% | n/a | 8% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "Each of our business segments operates in a highly competitive environment, and we face competition in all aspects of our business from numerous bank and non-bank providers of financial services."
  * [4] "In general, customers are attracted to credit card issuers largely on the basis of price, credit limit, reward programs, customer experience and other product features."
  * [5] "Our businesses generally compete on the basis of the quality and range of their products and services, transaction execution, innovation and price."
  * [6] "On May 21, 2024, our credit card program agreement with Walmart terminated (“Walmart Program Termination”)."

### EG: EVEREST GROUP, LTD. (Financials, 10-K filed 2025-02-27)

> The Company. the Company is a Bermuda-based reinsurance and insurance organization. As part of the Standard & Poor’s (“S&P”) 500 Index, we are a leading financial services institution focused on diversifying our portfolio and geographic presence. Through our direct and indirect subsidiaries ...

**Overall: none.** The text describes a mature, highly competitive, broker-intermediated market where pricing competes and alternative capital enters. Strong ratings and licenses give only modest support. Returns are volatile and near the cost of capital, so no moat is evident.

**Returns: near.** ROE was 3%, 14%, 6%, 23% and 10% over 2020-2024, averaging about 11%, and it swung widely with catastrophe losses and investment results. That is around the 10% cost of equity, not consistently above it. ROIC is not reported.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 17% | n/a | 3% | 30% | n/a | 3% | 1% |
| 2021-12-31 | 24% | n/a | 12% | 32% | n/a | 14% | 4% |
| 2022-12-31 | 2% | n/a | 5% | 31% | n/a | 6% | 2% |
| 2023-12-31 | 21% | n/a | 17% | 31% | n/a | 23% | 6% |
| 2024-12-31 | 18% | n/a | 8% | 29% | n/a | 10% | 3% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "The global reinsurance market is highly competitive and mature."
  * [4] "Competition for clients might be based on pricing, capacity, coverage terms, conditions or other factors."
  * [5] "The global insurance market is highly competitive."
  * [6] "In addition, we also compete with new companies and existing companies that move into the insurance industry."
  * [7] "The Reinsurance segment's ten largest brokers accounted for an aggregate of approximately 60.5% of gross written premiums in 2024."
  * [8] "In addition, competitors might include investment companies, mutual companies, insurance companies, alternative risk providers (such as captives, catastrophe bonds and pools) and others"

### MCO: MOODYS CORP /DE/ (Financials, 10-K filed 2025-02-14)

> Background As used in this report, except where the context indicates otherwise, the terms “Moody’s” or the “Company” refer to Moody’s Corporation, a Delaware corporation, and its subsidiaries. The Company’s executive offices are located at 7 World Trade Center at 250 Greenwich Street, New York, NY ...

**Overall: narrow.** MIS's regulatory registration, 115-year reputation and use by investors worldwide create durable intangible assets in a concentrated industry, reinforced by sticky embedded MA software. Returns are consistently far above the cost of capital. Competition from other agencies and in-house research, plus regulatory risk, keeps the case at narrow in the text provided, though the economics are strong.

**Returns: above.** ROIC was 18-37% over five years, well above the 8-10% cost of capital, with operating margins of 34-46%. The 2022 dip reflects a revenue decline in a weak debt market. ROE is inflated by buybacks, but ROIC alone is clearly high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 11% | 44% | 33% | 38% | 37% | >100% | 16% |
| 2021-12-31 | 16% | 46% | 36% | 30% | 32% | 95% | 16% |
| 2022-12-31 | -12% | 34% | 25% | 22% | 18% | 49% | 9% |
| 2023-12-31 | 8% | 36% | 27% | 32% | 20% | 54% | 11% |
| 2024-12-31 | 20% | 41% | 29% | 36% | 27% | 60% | 14% |

* **intangibles = 2.** MIS is a 115-year-old NRSRO-registered rating agency whose opinions are used by institutional investors worldwide, and MA owns the world's largest database on companies and credit. Regulatory registration, reputation and the investor-use dynamic form a strong brand/regulatory intangible, though the text does not quantify market share.
  * [1] "MIS is registered with the SEC as an NRSRO and is subject to the SEC's oversight and examination authority."
  * [2] "For more than 115 years, MIS has been a leading provider of credit ratings, research, and risk analysis helping businesses, governments, and other entities around the globe."
  * [3] "Moody’s opinions on credit are used by institutional investors throughout the world, making an issuer’s debt potentially more attractive to a wide range of buyers."
  * [4] "Among ratings advisors, Moody’s has a strong position and is well-recognized for the depth and breadth of its analytical capabilities."
  * [8] "a data business powered by the world’s largest database on companies and credit (Data & Information)"
* **switching_costs = 1.** MA solutions are embedded in customer workflows and contractually tied to regulated customers' compliance, with subscription retention. Annual fee arrangements and monitoring fees add stickiness. The text does not quantify retention.
  * [5] "MA's integrated and technology-enabled solutions provide unique capabilities and insights that are embedded in customer workflows."
  * [6] "MA also derives a significant amount of its sales from banks and other financial services providers who are subject to regulatory oversight and who are required to conduct due diligence and pass through certain regulatory requirements to key suppliers such as MA by contract."
  * [7] "However, annual fee arrangements with frequent debt issuers, annual fees from debt monitoring, commercial paper and medium-term note programs, bank deposit ratings, insurance company financial strength ratings, mutual fund ratings, and other areas partially mitigate MIS’s dependence on the volume or number of new debt securities issued in the global fixed-income markets."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [9] "MIS competes with other CRAs and with investment banks and brokerage firms that offer credit opinions and research. Many users of MIS’s ratings also have in-house credit research capabilities."
  * [10] "MA competes broadly in the financial information and enterprise risk software industries against various diversified competitors."

### NTRS: NORTHERN TRUST CORP (Financials, 10-K filed 2025-02-24)

> the Company the Company (the “Corporation”) is a leading provider of wealth management, asset servicing, asset management and banking solutions to corporations, institutions, families and individuals. The Corporation is a financial holding company conducting business through various U.S. and ...

**Overall: none.** The text states no moat mechanism directly and describes intense competition from many types of firms. Returns are only near the cost of equity. Scale in custody is plausible but unquantified, so the evidence does not support a moat.

**Returns: near.** ROIC is not reported. ROE ran 10-16% against a roughly 10% hurdle, so it was only modestly above the cost of capital and uneven. 2024 FCF was negative. Bank leverage means ROA of 1% is the better indicator.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 0% | n/a | 20% | 29% | n/a | 11% | 1% |
| 2021-12-31 | 6% | n/a | 24% | 19% | n/a | 13% | 1% |
| 2022-12-31 | 5% | n/a | 20% | 33% | n/a | 11% | 1% |
| 2023-12-31 | 0% | n/a | 16% | 37% | n/a | 10% | 1% |
| 2024-12-31 | 22% | n/a | 24% | -7% | n/a | 16% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "the Company faces intense competition in all aspects and areas of its business."
  * [5] "Our competitors include a broad range of financial institutions and service companies, including other custodial banks, investment counseling firms, deposit-taking institutions, asset management firms, benefits consultants, trust companies, investment banking firms, insurance companies, and various financial technology companies, including software providers and data services firms."

### PYPL: PayPal Holdings, Inc. (Financials, 10-K filed 2025-02-04)

> OVERVIEW At the Company, our mission is to revolutionize commerce globally. Our products are designed to enable digital payments and simplify commerce experiences for consumers and merchants to make selling, shopping, and sending and receiving money simple, personalized, and secure, online or ...

**Overall: narrow.** The two-sided network and brands (PayPal, Venmo) support a narrow moat, and ROIC has stayed above the cost of capital for five years. The text however describes a highly competitive, price-sensitive market with low switching costs and slowing user growth, which keeps the moat from being wide.

**Returns: above.** ROIC was 30% in 2020-21 and fell to 17-19% in 2022-24, still well above the ~8-10% cost of capital. ROE is 20%, operating margins are 14-17%, and free cash flow is positive. Growth has slowed to 7%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 21% | 15% | 20% | 25% | 30% | 23% | 7% |
| 2021-12-31 | 18% | 17% | 16% | 19% | 30% | 20% | 6% |
| 2022-12-31 | 8% | 14% | 9% | 19% | 17% | 12% | 3% |
| 2023-12-31 | 8% | 17% | 14% | 14% | 18% | 21% | 5% |
| 2024-12-31 | 7% | 17% | 13% | 21% | 19% | 20% | 5% |

* **intangibles = 1.** Licenses and the Venmo/brand position are cited, but they are asserted generally and not quantified; licenses are available to competitors who meet the same regulatory requirements.
  * [4] "we believe that our regulatory licenses, which enable us to operate in markets around the world, are a distinct advantage and help support business growth."
  * [5] "Our Venmo digital wallet in the United States (“U.S.”) is a leading mobile application used to move money between our customers."
* **network_effects = 1.** A two-sided network of 434M accounts with merchants is stated directly, and P2P drives acquisition. However, growth is only 2%, and competition from cards, wallets and other apps is intense, so the advantage looks modest.
  * [1] "We operate a global, two-sided network at scale that connects consumers and merchants with 434 million active accounts across approximately 200 markets as of December 31, 2024."
  * [2] "Our relationship on both sides of a transaction enables us to offer unique product experiences designed to remove friction, drive sales, and enhance shopping experiences."
  * [3] "P2P is an important source of customer engagement and also serves as a customer acquisition channel that facilitates organic growth by enabling potential users to establish active accounts with the Company or Venmo at the time they make or receive a P2P payment."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The global payments industry is highly competitive, dynamic, and innovative, and subject to regulatory scrutiny and oversight."
  * [8] "Many of the areas in which we compete evolve rapidly with innovative and disruptive technologies, shifting user preferences and needs, price sensitivity of consumers and merchants, and frequent introductions of new products and services."
  * [9] "Competition also may intensify as new competitors emerge, businesses combine or enter into new partnerships, and established companies in other segments expand to become competitive with various aspects of our business."
  * [10] "A user may register on our platform to access different products and may register more than one account to access a product."

### SCHW: SCHWAB CHARLES CORP (Financials, 10-K filed 2025-02-26)

> General Corporate Overview The Charles the Company (CSC) is a savings and loan holding company. CSC engages, through its subsidiaries (collectively referred to as the Company or the Company), in wealth management, securities brokerage, banking, asset management, custody, and financial advisory ...

**Overall: narrow.** A scale-driven cost advantage and a trusted brand support some durability, but the competitive field is broad (brokers, banks, fintechs, custodians), and the text gives no direct switching-cost evidence. ROE is near the cost of equity rather than clearly above it. The scale position is real, but the evidence is thin, so the moat is rated narrow.

**Returns: near.** As a financial company, it is judged on ROE: 8%, 10%, 15%, 13% and 13% over 2020-2024. These are around or modestly above the 10% cost of equity, and ROA is only 1%. The 2020 and 2021 figures were depressed by the Ameritrade acquisition. Net margins are high at 27-35%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 9% | n/a | 28% | 53% | n/a | 8% | 1% |
| 2021-12-31 | 58% | n/a | 32% | 6% | n/a | 10% | 1% |
| 2022-12-31 | 12% | n/a | 35% | 5% | n/a | 15% | 1% |
| 2023-12-31 | -9% | n/a | 27% | >100% | n/a | 13% | 1% |
| 2024-12-31 | 4% | n/a | 30% | 10% | n/a | 13% | 1% |

* **intangibles = 1.** The Company cites brand and reputation in a trust-dependent industry, but without figures or evidence of pricing power; regulatory burden is not an exclusive licence.
  * [3] "In an industry dependent on trust, the Company reputation and brand across multiple constituents enable us to attract clients and employees while credibly introducing new products to the market."
* **cost_advantage = 1.** The Company directly states a scale-based cost advantage from spreading costs over a very large client base ($10.1 trillion of assets) and shared infrastructure, letting it price competitively while earning profits. It is durable, but not quantified against peers.
  * [1] "As one of the largest investment services firms in the U.S., we are able to spread operating costs and amortize new investments over a large base of clients, and harness the resources to evolve capabilities to meet client needs."
  * [2] "Coupled with scale, our operating efficiency and sharing of infrastructure across different businesses creates a cost advantage that enables us to competitively price products and services while profitably serving clients of various sizes across multiple channels."
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [6] "Within Investor Services, our competition in serving individual investors spans brokerage, wealth management, and asset management firms, as well as banks, trust companies, financial technology companies, and retirement service providers."
  * [7] "In the Advisor Services arena, we compete with institutional custodians, wirehouses, regional and independent broker-dealers, banks, and trust companies."

### ABBV: AbbVie Inc. (Health Care, 10-K filed 2025-02-14)

> Overview the Company or "the company" refer to the Company, or the Company and its consolidated subsidiaries, as the context requires. the Company is a global, diversified research-based biopharmaceutical company positioned for success with a comprehensive product portfolio that has leadership ...

**Overall: narrow.** The moat rests on patents and regulatory exclusivity for specific drugs, plus leading aesthetics brands. These protections are finite: Humira is eroding under biosimilars, Imbruvica is subject to government-set prices, and the Skyrizi and Rinvoq patents expire in 2033. Returns have been above the cost of capital but are declining. Pharma IP with a diversified portfolio supports a narrow moat, not a wide one.

**Returns: above.** ROIC was 16%, 16%, 19%, 16% and 12% over the five years, above an 8-10% cost of capital through 2023 and only modestly above it in 2024. Operating margin fell from 32% to 16% as Humira eroded and acquisition costs rose. ROE is inflated by thin equity.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 38% | 25% | 10% | 37% | 16% | n/a | 4% |
| 2021-12-31 | 23% | 32% | 21% | 39% | 16% | 81% | 8% |
| 2022-12-31 | 3% | 31% | 20% | 42% | 19% | 72% | 8% |
| 2023-12-31 | -6% | 23% | 9% | 41% | 16% | 35% | 4% |
| 2024-12-31 | 4% | 16% | 8% | 32% | 12% | 63% | 3% |

* **intangibles = 1.** Patents and regulatory exclusivity protect key products (Skyrizi and Rinvoq to 2033, a portfolio of patents into the mid 2040s), and aesthetics holds market-leading positions. But Humira already faces biosimilars, Imbruvica faces government price setting, and the patent cliff is approaching, so the advantage is real but time-limited.
  * [1] "The United States composition of matter patents covering risankizumab and upadacitinib are expected to expire in 2033."
  * [2] "These patents and applications, including various patents that expire during the period 2025 to the mid 2040s, in aggregate are believed to be of material importance in the operation of the Company business."
  * [3] "The law provides a pathway for approval of biosimilars following the expiration of 12 years of regulatory exclusivity for the innovator biologic and a potential additional 180 day-extension term for conducting pediatric studies."
  * [4] "which hold market-leading positions in the U.S. and in key markets around the world."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Humira faces direct biosimilar competition globally, and the Company will continue to face competitive pressure from these biologics and from orally administered products."
  * [8] "The markets for the Company products are highly competitive."
  * [9] "In addition, the substitution of generic and biosimilar pharmaceutical products for branded pharmaceutical products creates competitive pressures on the Company products that do not have patent protection."
  * [10] "In 2024, three wholesale distributors (McKesson Corporation, Cardinal Health, Inc. and Cencora, Inc.) accounted for substantially all of the Company pharmaceutical product sales in the United States."
  * [11] "In August 2023, the U.S. Department of Health and Human Services (HHS), through the Centers for Medicare & Medicaid Services (the CMS), selected Imbruvica as one of the first 10 medicines subject to government-set prices beginning in 2026."

### DGX: QUEST DIAGNOSTICS INC (Health Care, 10-K filed 2025-02-20)

> ." Critical Accounting Policies The preparation of financial statements in conformity with accounting principles generally accepted in the United States requires us to make estimates and assumptions and select accounting policies that affect our reported financial results and the disclosure of ...

Not tagged: the located Item 1 'Critical Accounting Policies' in the first 10,000 characters.

### GILD: GILEAD SCIENCES, INC. (Health Care, 10-K filed 2025-02-28)

> the Company (including its consolidated subsidiaries, referred to as “the Company,” the “company,” “we,” “our” or “us”) is a biopharmaceutical company that has pursued and achieved breakthroughs in medicine for more than three decades, with the goal of creating a healthier world for all people. We ...

**Overall: narrow.** The moat is patent- and regulation-based and finite. Key HIV franchises are protected into the 2030s, and returns have been high in most years. But patents expire, pricing pressure is heavy, wholesaler concentration is high, and the company must keep replenishing its pipeline. Narrow.

**Returns: unclear.** ROIC was 14-18% in 2021-2023, above the cost of capital, but 8% in 2020 and 3% in 2024, apparently hit by acquired in-process R&D charges. Returns are mixed and distorted by one-off items.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 10% | 16% | 0% | 30% | 8% | 1% | 0% |
| 2021-12-31 | 11% | 36% | 23% | 40% | 18% | 32% | 9% |
| 2022-12-31 | -0% | 27% | 17% | 31% | 14% | 22% | 7% |
| 2023-12-31 | -1% | 28% | 21% | 27% | 15% | 26% | 9% |
| 2024-12-31 | 6% | 6% | 2% | 36% | 3% | 2% | 1% |

* **intangibles = 1.** Patents on key HIV products like Biktarvy run to 2033, Descovy/Odefsey to 2031-2032, Sunlenca to 2037, with regulatory exclusivity and heavy FDA barriers. The protection is finite, though, and the portfolio faces patent cliffs and generic settlements.
  * [2] "If we have a properly drafted and enforceable patent, it can be more difficult for our competitors to use our technology to create competitive products"
  * [3] "As a result of these regulations, product development and product approval processes are very expensive and time consuming, which has a significant impact on our capital expenditures and results of operations."
  * [4] "Ten years of regulatory/market exclusivity expected on approval."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "We operate in a highly competitive environment."
  * [8] "approximately 91% of our product sales in the U.S. and approximately 65% of our total worldwide revenues were from three large wholesalers"
  * [9] "pricing pressures from private insurers and government payers as our products mature, which often result in a reduction of the net product prices"
  * [10] "new branded or generic products introduced into major markets, which may impact our ability to maintain pricing and market share."
  * [11] "As a result, the price increases we implement from time to time on certain products may have a limited effect on our net product sales in certain markets."

### HUM: HUMANA INC (Health Care, 10-K filed 2025-02-20)

> General Headquartered in Louisville, Kentucky, the Company. and its subsidiaries, referred to throughout this document as “we,” “us,” “our,” the “Company” or “the Company,” is committed to putting health first – for our teammates, our customers, and our company. Through our the Company insurance ...

**Overall: none.** Competition is intense, revenue depends on the federal government (85%) with annual contract renewals and rate-setting by CMS, and rivals are often larger. Scale yields some provider discounts, but the text shows no durable mechanism, and returns have fallen to about the cost of capital.

**Returns: near.** ROIC fell from 27% in 2020 to 12-14% in 2021-23 and 8% in 2024, with operating margin down to 2%. Returns are near the cost of capital and declining.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 19% | 6% | 4% | 6% | 27% | 26% | 11% |
| 2021-12-31 | 8% | 4% | 4% | 1% | 12% | 20% | 7% |
| 2022-12-31 | 12% | 4% | 3% | 4% | 13% | 18% | 6% |
| 2023-12-31 | 15% | 4% | 2% | 3% | 14% | 16% | 6% |
| 2024-12-31 | 11% | 2% | 1% | 2% | 8% | 7% | 3% |

* **cost_advantage = 1.** Membership scale gives some provider discounts, but many competitors are larger, so the advantage is modest and not shown to be structural.
  * [5] "Our membership base and the ability to influence where our members seek care generally enable us to obtain contractual discounts with providers."
  * [6] "CenterWell Home Health has locations in 40 states, providing extensive geographic coverage with approximately 65% overlap with our individual Medicare Advantage membership."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "The health benefits industry is highly competitive."
  * [2] "Many of our competitors have a larger membership base and/or greater financial resources than our health plans in the markets in which we compete."
  * [3] "During 2024, 85% of our total premiums and services revenue were derived from contracts with the federal government"
  * [7] "Our individual Medicare Advantage products covered under Medicare Advantage contracts with CMS are renewed generally for a calendar year term unless CMS notifies us of its decision not to renew by May 1 of the calendar year in which the contract would end"

### IDXX: IDEXX LABORATORIES INC /DE (Health Care, 10-K filed 2025-02-21)

> COMPANY OVERVIEW the Company was incorporated in Delaware in 1983. We develop, manufacture, and distribute products and provide services primarily for the companion animal veterinary, livestock and poultry, dairy and water testing industries. We also provide human medical point-of-care and ...

**Overall: narrow.** The installed base of analyzers brings recurring consumable revenue, and the company has integrated software and exclusive slide supply. Regulatory approvals add protection in the water and livestock businesses. Consistently very high returns corroborate narrow advantages. The text states each mechanism modestly, and competition is robust with well-funded rivals, so narrow rather than wide.

**Returns: above.** ROIC was 49-58% in all five years, with operating margins of 26-30% and free cash flow margins of 12-21%. These are far above the cost of capital and are not driven by leverage.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 12% | 26% | 21% | 20% | 56% | >100% | 28% |
| 2021-12-31 | 19% | 29% | 23% | 20% | 58% | >100% | 31% |
| 2022-12-31 | 5% | 27% | 20% | 12% | 53% | >100% | 26% |
| 2023-12-31 | 9% | 30% | 23% | 21% | 58% | 81% | 28% |
| 2024-12-31 | 6% | 29% | 23% | 21% | 49% | 58% | 27% |

* **intangibles = 1.** Exclusive Catalyst slide supply through 2044 and regulatory approvals (EPA, USDA, FDA) for water, dairy and livestock tests provide some protection. However, the company says no single patent is material and it faces robust competition.
  * [4] "The agreements also prohibit Ortho from promoting and selling these chemistry slides in the veterinary sector, excluding the EU, other than to the Company."
  * [5] "These products must be approved by APHIS before they may be sold in and from the U.S."
  * [6] "before a test can be used as part of a water quality monitoring program in the U.S. that is regulated by the EPA, the test method must first be approved by the EPA."
  * [7] "All test methods used for drug residue testing require validation by the FDA and acceptance by the NCIMS."
* **switching_costs = 1.** A large installed base of analyzers drives recurring consumable sales, and the IVLS, practice management and cloud software are integrated into clinic workflows. Competitors have comparable offerings, so the lock-in is moderate rather than overwhelming.
  * [1] "Sales of consumables to customers who use our chemistry analyzers provide the majority of our instrument consumables revenues from our installed base of the Company VetLab instruments."
  * [2] "We sell IVLS as an integral component for our in-clinic analyzer suite."
  * [3] "Our large practice management systems installed base provides access to veterinary channel transaction activity, enabling a syndicated data offering."
  * [12] "Third-party integrations strengthen our practice management systems value proposition by improving user workflows and can quickly add new functionality to the practice management systems."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [9] "we face robust competition as other companies have been successful in bringing competitive products and services to market, despite the protections afforded by these patent and license rights."
  * [10] "Several of our direct and potential competitors have substantially greater financial and managerial resources than us, as well as greater experience in manufacturing, marketing, research and development, and obtaining regulatory approvals than we do."
  * [11] "In some cases, these third parties are sole or single-source suppliers."

### MOH: MOLINA HEALTHCARE, INC. (Health Care, 10-K filed 2025-02-11)

> OVERVIEW ABOUT the Company the Company, a FORTUNE 500 company, provides managed healthcare services under the Medicaid and Medicare programs, and through the state insurance marketplaces (the “Marketplace”). the Company was founded in 1980 as a provider organization serving low-income families in ...

**Overall: none.** Text describes a competitive, contract-driven government market with rebids, short terms, and large rivals; the barriers are only partial. Strong ROE reflects leverage and underwriting, and the stated mechanisms do not support a durable moat.

**Returns: above.** ROIC is very high (63% to over 100%) but inflated by thin equity capital; ROE of 27-33% is well above 10%, yet net margins are only 2-3%, and FCF margins are thin. Returns are consistently above the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 15% | 6% | 3% | 9% | n/a | 33% | 8% |
| 2021-12-31 | 43% | 4% | 2% | 7% | n/a | 28% | 6% |
| 2022-12-31 | 15% | 4% | 2% | 2% | >100% | 28% | 6% |
| 2023-12-31 | 7% | 5% | 3% | 5% | 93% | 30% | 8% |
| 2024-12-31 | 19% | 4% | 3% | 1% | 63% | 27% | 8% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The competition is fierce, as our markets are attractive."
  * [2] "Such contracts are subject to risk of loss in states that issue RFPs open to competitive bidding by other health plans."
  * [3] "Our state Medicaid contracts typically have terms of three to five years, contain renewal options exercisable by the state Medicaid agency, and allow either the state or the health plan to terminate the contract with or without cause."
  * [4] "Competition can vary considerably from state to state, and some of our competitors have larger membership bases and/or greater financial resources than our health plans in the markets in which we compete."
  * [5] "The Medicare market is highly competitive across the country, with large competitors, such as CVS Health Corporation, Humana Inc., and UnitedHealth Group Inc."
  * [8] "A loss of any of our significant Medicaid contracts could have a material adverse effect on our business, financial condition, cash flows, or results of operations."
  * [10] "There is increasing competition driven by renewed interest from large national health plans."

### REGN: REGENERON PHARMACEUTICALS, INC. (Health Care, 10-K filed 2025-02-05)

> This Annual Report on Form 10-K contains forward-looking statements that involve risks and uncertainties relating to future events and the future performance of the Company (where applicable, together with its subsidiaries, "the Company," "Company," "we," "us," and "our"), and actual events or ...

**Overall: narrow.** The moat rests on patent-protected drugs, with Dupixent protected into the 2030s, and on a proprietary antibody discovery platform. Returns have stayed above the cost of capital. Against this, the text describes biosimilar erosion of EYLEA, a crowded field of competitors in each indication, 74% sales concentration in wholesalers, pricing pressure from payors and the IRA, and finite patent lives. That supports a narrow moat of about 10 years.

**Returns: above.** ROIC was 28%, 49%, 19%, 14% and 12% over 2020-2024, well above the 8-10% cost of capital throughout, but trending down as margins fell from 56% to 28%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 30% | 42% | 41% | 24% | 28% | 32% | 22% |
| 2021-12-31 | 89% | 56% | 50% | 41% | 49% | 54% | 38% |
| 2022-12-31 | -24% | 39% | 36% | 36% | 19% | 21% | 16% |
| 2023-12-31 | 8% | 31% | 30% | 30% | 14% | 16% | 13% |
| 2024-12-31 | 8% | 28% | 31% | 26% | 12% | 16% | 12% |

* **intangibles = 1.** Patent-protected biologics (Dupixent patents into the 2030s and beyond, Libtayo to 2035+) and FDA-approved facilities give exclusivity, but EYLEA's key patents expire around 2027 and face biosimilar competition. Patent terms are finite and uncertain, so the advantage is real but time-limited.
  * [1] "Our remaining issued patents covering these technologies will expire between 2025 and 2032."
  * [2] "A patent term extension has been granted by the U.S. Patent and Trademark Office, extending the original patent term (October 2, 2027), insofar as it covers Dupixent, to March 28, 2031."
  * [3] "Each of these facilities is approved by the FDA and certain other regulatory agencies to manufacture our bulk drug materials and products."
  * [4] "These efforts at RGC have led to the identification of more than 30 novel genetic targets."
  * [5] "RGC utilizes laboratory automation and innovative approaches to cloud computing to achieve high-quality throughput, attaining nearly 3 million samples sequenced to date."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Competition among products approved for sale is based on efficacy, safety, reliability, ease of administration, dosing frequency, availability, price, patent and other intellectual property position, and other factors."
  * [7] "On a combined basis, our product sales to these customers accounted for 74% of our total gross product revenue for the year ended December 31, 2024."
  * [8] "There is no certainty that our existing patents or others, if obtained, will provide us protection from competition or provide commercial benefit."
  * [9] "Others may independently develop similar products or processes to those developed by us, duplicate any of our products or processes or, if patents are issued to us, design around any products and processes covered by our patents."
  * [10] "Private payor healthcare and insurance providers, health maintenance organizations, and pharmacy benefit managers in the United States are adopting more aggressive utilization management techniques and are increasingly requiring significant discounts and rebates from manufacturers as a condition to including products on formulary with favorable coverage and copayment/coinsurance."
  * [11] "The IRA also created a drug price negotiation program requiring the government to set prices for select high-expenditure drugs covered under Medicare Parts B and D."

### ALLE: Allegion plc (Industrials, 10-K filed 2025-02-18)

> BUSINESS Overview the Company ("the Company," "we," "us" or "the Company") is a leading global provider of security products and solutions that keep people and assets safe and secure in the places they live, learn, work and connect. We pioneer safety and security to create a safer and more ...

**Overall: narrow.** Strong brands and specification relationships, together with consistently high ROIC and rising margins, point to a real but modest advantage. The text describes a fragmented, competitive market where it competes partly on price, and no mechanism is quantified. Narrow fits better than wide.

**Returns: above.** ROIC was 18-23% over five years, well above the 8-10% cost of capital. Operating margins rose from 15% to 21%. ROE is inflated by leverage, but ROIC alone is clearly high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -5% | 15% | 12% | 16% | 18% | 39% | 10% |
| 2021-12-31 | 5% | 18% | 17% | 15% | 23% | 61% | 16% |
| 2022-12-31 | 14% | 18% | 14% | 12% | 20% | 54% | 13% |
| 2023-12-31 | 12% | 19% | 15% | 14% | 20% | 48% | 13% |
| 2024-12-31 | 3% | 21% | 16% | 15% | 21% | 42% | 14% |

* **intangibles = 1.** Century-old brands (Schlage, Von Duprin, LCN) are called its most valuable assets and are specified by architects, but the text gives no pricing power or share figures, and it says it does not depend on any particular patent.
  * [3] "We consider our CISA, Interflex, LCN, Schlage, SimonsVoss, Von Duprin and other associated trademarks to be among our most valuable assets, and we have registered these trademarks in a number of countries."
  * [4] "Several of our brands were established more than 100 years ago, and many originally created their categories:"
* **switching_costs = 1.** Specification work with architects and recurring service on automatic entrances suggest some stickiness. The text does not say that customers are locked in.
  * [6] "Our Stanley Access Technologies business offers extensive planned inspection, maintenance and repair services for its automatic entrance solutions throughout the U.S. and Canada."
  * [7] "Our field sales professionals are assisted by specification writers who work with architects, engineers and consultants to help design door openings and security systems to meet end-users’ functional, aesthetic and regulatory requirements."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The security products markets are highly competitive and fragmented throughout the world, with a number of large multi-national companies and thousands of smaller regional and local companies."
  * [2] "We compete based on the breadth, innovation and quality of our products and solutions, pricing, our ability to custom-configure solutions to meet individual end-user requirements and our global supply chain."
  * [5] "Although certain proprietary intellectual property rights are important to our success, we do not believe we are materially dependent on any particular patent or license, or any particular group of patents or licenses."
  * [8] "Our 10 largest customers represented approximately 27% of our total Net revenues in 2024."
  * [9] "As we move into more technologically advanced product categories, we may also compete against new, more specialized competitors."

### AME: AMETEK INC/ (Industrials, 10-K filed 2025-02-20)

> General Development of Business the Company (“the Company” or the “Company”) is incorporated in Delaware. Its predecessor was originally incorporated in Delaware in 1930 under the name American Machine and Metals, Inc. the Company is a leading global manufacturer of electronic instruments and ...

**Overall: none.** A diversified niche instrument and components maker with high, stable margins and a very diversified customer base. The text describes a competitive market with many rivals and gives no strong, quantified moat sources. Aerospace reputation, custom-engineered products and niche positions add some durability. Returns are only moderately above the cost of capital, so the evidence does not support a moat.

**Returns: above.** ROIC of 11-13% is modestly above the 8-10% cost of capital. Operating margins of 23-26% are high and steady, and ROE is 15-16%. Returns are consistent but not exceptional.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -12% | 23% | 19% | 27% | 11% | 16% | 9% |
| 2021-12-31 | 22% | 24% | 18% | 19% | 13% | 15% | 9% |
| 2022-12-31 | 11% | 24% | 19% | 16% | 13% | 16% | 10% |
| 2023-12-31 | 7% | 26% | 20% | 24% | 13% | 16% | 10% |
| 2024-12-31 | 5% | 26% | 20% | 25% | 12% | 15% | 9% |

* **intangibles = 1.** A long aerospace reputation is cited, but the company says no patent or trademark is essential, and the brand claims are generic.
  * [2] "Among its more significant competitive advantages is its 70-year-plus reputation as an established aerospace supplier."
* **switching_costs = 1.** Products designed to customer specifications and long-standing aerospace OEM relationships suggest some stickiness, but there are no contract terms or quantified lock-in.
  * [4] "the Company aerospace products are designed to customer specifications and manufactured to stringent operational and reliability requirements."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "In general, the Company markets are highly competitive with competition based on technology, performance, quality, service and price."
  * [6] "In process and analytical instruments, numerous companies compete in each market on the basis of product quality, performance and innovation."
  * [7] "the Company, however, does not consider any single patent or trademark, or any group of them, essential either to its business as a whole or to either one of its reportable segments."
  * [8] "EMG’s businesses compete with a number of companies in each of its markets."
* Fixes: switching_costs: dropped citations [3].

### CHRW: C. H. ROBINSON WORLDWIDE, INC. (Industrials, 10-K filed 2025-02-14)

> Overview the Company (“the Company,” “the company,” “we,” “us,” or “our”) is one of the largest global logistics providers in the world, with consolidated total revenues of $17.7 billion in 2024. We deliver logistics like no one else. Companies around the world look to us to reimagine supply ...

**Overall: none.** Returns are decent but come from an asset-light model in a fragmented, price-competitive market with short contracts and no volume commitments. Scale, data and the carrier network are modest advantages that could be copied, and the large 2023 decline shows cyclicality. A narrow moat is plausible from scale and network, but the text does not support durable advantages strongly enough.

**Returns: above.** ROIC was 21-29% in 2020-2022, dropped to 14% in 2023 and recovered to 18% in 2024, always above the cost of capital. Margins are thin (3-5% operating) and the capital base is light, as for an asset-light broker, so the returns are partly a function of that model and cyclical gross profit.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 6% | 4% | 3% | 3% | 21% | 29% | 10% |
| 2021-12-31 | 43% | 5% | 4% | 0% | 27% | 43% | 14% |
| 2022-12-31 | 7% | 5% | 4% | 6% | 29% | 56% | 14% |
| 2023-12-31 | -29% | 3% | 2% | 4% | 14% | 23% | 6% |
| 2024-12-31 | 1% | 4% | 3% | 3% | 18% | 30% | 9% |

* **network_effects = 1.** A platform of 83,000 customers and 450,000 carriers is described and suggests some two-sided matching benefit, but the text does not state that value grows with users; carriers are fragmented and multi-home.
  * [6] "As one of the world’s largest logistics platforms, our proprietary technology connects"
  * [7] "In 2024, we managed approximately 37 million shipments for 83,000 customers utilizing the more than 450,000 contract carriers on our platform."
* **cost_advantage = 1.** Scale and one of the largest freight datasets are claimed to aid pricing and purchasing, but there is no quantified cost advantage over competitors in a fragmented market.
  * [9] "We have one of the largest datasets of shipments, routings, and carriers in the world."
* Rated 0: intangibles, switching_costs, efficient_scale.
* Against a moat:
  * [1] "The transportation services industry is highly competitive and fragmented."
  * [2] "We often compete with respect to price, scope of services, or a combination thereof, but believe that our most significant competitive advantages are:"
  * [3] "Most of our contractual rate commitments are for one year or less and allow for renegotiation."
  * [4] "As is typical in the transportation industry, most of these contracts do not include specific volume commitments."
  * [5] "Our standard contracts do not include volume commitments, and typically, the initial contract rate is modified each time we confirm an individual shipment with a contracted motor carrier."
  * [10] "During 2024, our largest customer accounted for approximately two percent of our consolidated total revenues."

### DE: DEERE & CO (Industrials, 10-K filed 2025-12-18)

> BUSINESS. This Annual Report on Form 10-K contains forward-looking statements that are subject to risks and uncertainties. All statements other than statements of historical fact included in this Annual Report on Form 10-K are forward-looking statements. Forward-looking statements provide our ...

**Overall: narrow.** The brand, the large dealer network and proprietary parts and aftermarket give a modest installed-base advantage. Returns have been consistently high. The text describes a competitive industry with many rivals and new entrants. Pricing power is plausible but not quantified, so the moat is narrow.

**Returns: above.** ROIC was 13%, 18%, 34% and 23% in the four years reported, and ROE was 21-48%. All are above the cost of capital, though they peaked in the cycle and are now declining with revenue. Financial Services leverage inflates ROE.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-10-31 | 24% | 18% | 14% | 16% | 13% | 38% | 7% |
| 2022-10-30 | 19% | 18% | 14% | 7% | 18% | 37% | 8% |
| 2023-10-29 | 16% | 21% | 17% | 12% | 34% | 48% | 10% |
| 2024-10-27 | -16% | 17% | 14% | 15% | 23% | 32% | 7% |
| 2025-11-02 | -12% | n/a | 11% | 13% | n/a | 21% | 5% |

* **intangibles = 1.** Brand recognition is cited as a competitive factor and the dealer network is large, but the text gives no figures on pricing power or share. Patents are explicitly not critical.
  * [1] "John the Company brand recognition is a competitive factor in North America and many other parts of the world."
  * [4] "Through the U.S. and Canada, we market products to approximately 2,050 independent dealer locations."
* **switching_costs = 1.** Proprietary parts for the installed base and connected-machine telematics create some lock-in, but the text does not show strong contractual or integration costs.
  * [2] "We supply parts, many of which are proprietary, to support the current product lines as well as products we have sold in the past."
  * [3] "We support our connected machines, which integrate technology with our equipment through our John the Company Operations CenterTM with service alerts, operator insights, and predictive repairs and maintenance."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "The agricultural equipment industry continues to change and is becoming even more competitive through the emergence and global expansion of many competitors."
  * [6] "As technology increasingly enables enhanced productivity in agriculture, the industry is also attracting non-traditional competitors, including technology-focused companies and start-up ventures."
  * [7] "The principal competitive factors in all markets include product performance, technology features, innovation, quality, distribution, sustainability, financing, customer service, and value."

### LII: LENNOX INTERNATIONAL INC (Industrials, 10-K filed 2025-02-11)

> Business References in this Annual Report on Form 10-K to “we,” “our,” “us,” “the Company,” “the Company” or the “Company” refer to the Company. and its subsidiaries, unless the context requires otherwise. The Company We are a global leader in energy-efficient climate-control solutions. We design, ...

**Overall: narrow.** The text lists no explicit moat mechanisms; the market is competitive with a handful of large rivals competing on price, availability and efficiency. Very high returns for five years and a sizable owned distribution network (company stores, direct-to-dealer) in a concentrated oligopoly industry suggest durable advantages, but the text does not quantify them. Given the returns, this supports a narrow moat.

**Returns: above.** ROIC of 42-55% for five years with rising operating margins (13% to 19%) is far above cost of capital, though equity is small due to buybacks, which inflates ROIC; ROE above 100% is leverage-driven. Even so, returns are clearly high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -5% | 13% | 10% | 15% | 42% | n/a | 18% |
| 2021-12-31 | 15% | 14% | 11% | 10% | 52% | n/a | 22% |
| 2022-12-31 | 13% | 14% | 11% | 4% | 47% | n/a | 21% |
| 2023-12-31 | 6% | 16% | 12% | 10% | 47% | n/a | 22% |
| 2024-12-31 | 7% | 19% | 15% | 15% | 55% | >100% | 26% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "Substantially all markets in which we participate are competitive."
  * [5] "The most significant competitive factors we face are product availability, reliability, energy efficiency, product performance, service, and price, with the relative importance of these factors varying among customer applications."
  * [6] "We consider these patents important, but no single patent is material to the overall conduct of our business."
  * [7] "We also sell our products through our the Company Stores."

### PWR: QUANTA SERVICES, INC. (Industrials, 10-K filed 2025-02-20)

> Business OVERVIEW the Company (together with its subsidiaries, “the Company,” “we,” “us” or “our”) is a leading provider of comprehensive infrastructure solutions for the electric and gas utility, renewable energy, technology, communications, pipeline and energy industries in the United States, ...

**Overall: none.** This is a large contractor with strong utility relationships and scale, but the text describes a competitive, price-driven bidding market with low entry barriers in places. Returns are near the cost of capital, and no moat source is clearly stated.

**Returns: near.** ROIC was 7-11% over five years, rising to 10-11% in the last two. That is roughly at the cost of capital, with only recent improvement. ROE of 9-13% is moderate.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -8% | 5% | 4% | 8% | 9% | 11% | 5% |
| 2021-12-31 | 16% | 5% | 4% | 2% | 7% | 10% | 5% |
| 2022-12-31 | 32% | 5% | 3% | 4% | 8% | 9% | 4% |
| 2023-12-31 | 22% | 5% | 4% | 5% | 10% | 13% | 5% |
| 2024-12-31 | 13% | 6% | 4% | 6% | 11% | 13% | 5% |

* **switching_costs = 1.** Preferred-provider status and alliance agreements give some preferential bidding, but contracts are mostly bid on price and customers can switch, so the effect is modest.
  * [1] "We are a preferred service provider for many of our customers, which generally means we have met minimum standards for a specific category of service, maintained a high level of performance and agreed to certain payment terms and negotiated rates."
  * [2] "Strategic alliance agreements also generally state an intention to work together over a period of time and/or on specific types of projects, and many provide us with preferential bidding procedures."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "In addition, there are relatively few barriers to entry into some of the industries in which we operate and, as a result, organizations that have adequate financial resources and access to technical expertise may become a competitor."
  * [6] "A significant portion of our revenues is currently derived from unit price or fixed price agreements, and price is often an important factor in the award of such agreements."
  * [7] "The industries and geographic markets in which we operate are highly competitive, and several of our competitors are large companies that have significant financial, technical and marketing resources."
  * [8] "The in-house service organizations of our existing or prospective customers employ personnel who perform some of the same types of services we provide."

### RSG: REPUBLIC SERVICES, INC. (Industrials, 10-K filed 2025-02-14)

> BUSINESS Overview Republic is one of the largest providers of environmental services in the United States, as measured by revenue. We operate across the United States and Canada through 367 collection operations, 248 transfer stations, 75 recycling centers, 208 active landfills, 2 treatment, ...

**Overall: narrow.** Owned landfills, permitting hurdles, route density and municipal contracts give a local-scale advantage in a fragmented, local industry. Returns are only modestly above the cost of capital, so the evidence supports a narrow moat at most. Narrow fits the integrated landfill-to-collection model.

**Returns: near.** ROIC rose from 8% to 11% over five years, roughly at or slightly above the 8-10% cost of capital. ROE is 12-19%, partly lifted by leverage. Margins are steady and FCF margin is 11%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -1% | 14% | 8% | 11% | 8% | 12% | 4% |
| 2021-12-31 | 11% | 16% | 10% | 11% | 9% | 15% | 5% |
| 2022-12-31 | 17% | 15% | 10% | 11% | 10% | 16% | 6% |
| 2023-12-31 | 10% | 16% | 10% | 11% | 10% | 17% | 6% |
| 2024-12-31 | 7% | 17% | 11% | 11% | 11% | 19% | 6% |

* **intangibles = 1.** Landfill and facility permits are hard to obtain, but the text only notes the permitting burden for new development, not a defended exclusive position. The 'sole provider' claim is vague.
  * [2] "Development projects, while generally less capital intensive than acquisitions, typically require extensive permitting efforts that can take years to complete with no assurance of success."
  * [6] "We offer a broad set of environmental services across the United States and Canada as the sole provider, which we believe sets us apart in the industry."
* **switching_costs = 1.** Long-term municipal contracts and exclusive franchises lock in volume, but the text gives no renewal terms or figures, and customers can rebid contracts.
  * [1] "We seek to obtain long-term contracts for collecting recyclable, solid waste and industrial waste material under residential collection contracts with municipalities, exclusive franchise agreements, small-container and large-container contracts and environmental solutions service contracts."
* **cost_advantage = 1.** Route density, vertical integration and fleet automation point to local scale advantages, but the text offers no quantified comparison with competitors.
  * [3] "We realize synergies from consolidating businesses into our existing operations, whether through acquisitions or public-private partnerships, which allows us to reduce capital expenditures and expenses associated with truck routing, personnel, fleet maintenance, inventories and back-office administration."
  * [4] "Our goal is to develop the best vertically integrated market position to enable us to build density and improve returns."
  * [5] "Approximately 77% of our residential routes have been converted to automated single-driver trucks."
* **efficient_scale = 1.** Landfills and transfer stations are hard to permit, and local markets support few disposal sites. The text only hints at this and does not state that markets hold only a few firms.
  * [2] "Development projects, while generally less capital intensive than acquisitions, typically require extensive permitting efforts that can take years to complete with no assurance of success."
  * [4] "Our goal is to develop the best vertically integrated market position to enable us to build density and improve returns."
* Rated 0: network_effects.
* Against a moat:
  * [7] "We operate throughout North America, but the physical collection and recycling or disposal of material is very much a local business, and the dynamics and opportunities differ in each of the markets we serve."
  * [8] "We believe our ability to successfully complete these acquisitions is enhanced by the challenges facing many privately-held companies, including increasing competition in the environmental services industry"
  * [9] "Environmental solutions remains fragmented, which provides consolidation opportunities to drive scale."

### UAL: United Airlines Holdings, Inc. (Industrials, 10-K filed 2025-02-27)

> Overview the Company (together with its consolidated subsidiaries, "the Company" or the "Company") is a holding company and its wholly-owned subsidiary is the Company (together with its consolidated subsidiaries, "United"). United's shared purpose is "Connecting People. Uniting the World." United ...

**Overall: none.** The airline industry is highly competitive, with price matching and easy entry. The network and hub scale give some advantage but not a durable moat, and returns were below the cost of capital for several years.

**Returns: near.** ROIC was deeply negative in 2020-21 and reached 7% in 2022, 11% in 2023 and 13% in 2024. Recent returns are above the cost of capital but the history is volatile and short of consistent.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -65% | -41% | -46% | -38% | -23% | -81% | -13% |
| 2021-12-31 | 60% | -4% | -8% | -0% | -4% | -36% | -3% |
| 2022-12-31 | 82% | 5% | 2% | 3% | 7% | 12% | 1% |
| 2023-12-31 | 19% | 8% | 5% | -0% | 11% | 32% | 4% |
| 2024-12-31 | 6% | 9% | 6% | 7% | 13% | 29% | 4% |

* **network_effects = 1.** Hub-and-spoke networks and the Star Alliance give scale benefits in connectivity, but this is a network-scale advantage rather than a user-to-user effect, and it is modest.
  * [2] "The hub and spoke system allows us to transport passengers between a large number of destinations with substantially more frequent service than if each route were served directly."
  * [3] "United is a member of Star Alliance, a global integrated airline network and the largest and most comprehensive airline alliance in the world."
* Rated 0: intangibles, switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The domestic airline industry is highly competitive and dynamic."
  * [7] "Currently, any U.S. carrier deemed fit by the U.S. Department of Transportation (the "DOT") is largely free to operate scheduled passenger service between any two points within the United States."
  * [8] "Domestic pricing decisions are impacted by intense competitive pressure exerted on the Company by other U.S. airlines."
  * [9] "In order to remain competitive and maintain passenger traffic levels, we often find it necessary to match competitors' discounted fares."
  * [10] "Carriers with lower costs may offer lower fares to passengers, which could have a potential negative impact on the Company's revenues."

### CDW: CDW Corp (Information Technology, 10-K filed 2025-02-21)

> Our Company the Company (together with its subsidiaries, the “Company,” “the Company” or “we”), a Fortune 500 company and member of the S&P 500 Index, is a leading multi-brand provider of information technology (“IT”) solutions to business, government, education and healthcare customers in the ...

**Overall: none.** The text describes a competitive, fragmented reseller market in which supply is available from many sources and the company depends on a few large vendors. No mechanism is stated directly. Solid ROIC does not turn scale and relationships into a moat in a price-driven distribution business.

**Returns: above.** ROIC was 18-23% over five years, above the cost of capital. ROE of 49-99% is inflated by leverage and buybacks. Operating margins of 6-8% are thin, typical of a reseller, and the high ROIC comes from a thin capital base.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | 6% | 4% | 6% | 23% | 70% | 9% |
| 2021-12-31 | 13% | 7% | 5% | 3% | 20% | 99% | 9% |
| 2022-12-31 | 14% | 7% | 5% | 5% | 19% | 97% | 8% |
| 2023-12-31 | -10% | 8% | 5% | 7% | 19% | 61% | 8% |
| 2024-12-31 | -2% | 8% | 5% | 5% | 18% | 49% | 8% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "These are highly fragmented markets served by thousands of IT resellers and solutions providers."
  * [2] "The market for technology products, solutions and services is highly competitive and subject to economic conditions and rapid technological changes."
  * [3] "We believe competitive sources of supply are available in substantially all of the product categories that we offer."
  * [4] "While innovation can help our business as it creates new offerings for us to sell, it can also disrupt our business model and create new and stronger competitors."
  * [8] "These programs are at the discretion of our vendor partners and are typically tied to sales or other commitments to be met by us within a specified period."
  * [9] "In 2024, we generated $2.0 billion of Net sales from each of our three largest vendor partners."

### DELL: Dell Technologies Inc. (Information Technology, 10-K filed 2025-03-25)

> Company Overview the Company is a leader in the global technology industry focused on providing broad and innovative technology solutions for the data and artificial intelligence (“AI”) era. These solutions range from client devices and peripherals to infrastructure solutions across servers, ...

**Overall: none.** The text describes a price-competitive, commodity-like hardware market with low barriers to entry from ODMs and cloud providers. The scale and direct-sales advantages are modest and unquantified. High ROIC reflects a thin capital base and thin margins, not a stated moat, so no moat.

**Returns: above.** ROIC rose from 7% to 25-28% over the last three years, well above the cost of capital. However, operating margins are only 4-7%, and the capital base is thin and shrunk by buybacks, the VMware spin-off and negative equity. Returns come from a low-margin hardware model and are likely overstated.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-29 | 2% | 4% | 4% | 11% | 7% | 61% | 3% |
| 2022-01-28 | 17% | 5% | 5% | 7% | 16% | n/a | 5% |
| 2023-02-03 | 1% | 6% | 2% | 1% | 28% | n/a | 3% |
| 2024-02-02 | -14% | 6% | 4% | 7% | 25% | n/a | 4% |
| 2025-01-31 | 8% | 7% | 5% | 2% | 27% | n/a | 6% |

* **cost_advantage = 1.** The company claims a large-scale supply chain and a direct sales model, but gives no comparison with competitors, and the market is competitive on price. The claims are modest and unquantified.
  * [5] "Our world-class supply chain operates at a significant scale with the ability to remain agile in a variety of environments."
  * [8] "We believe our direct sales channel is a significant competitive advantage and emphasizes direct communication with customers, allowing us to refine our products and marketing programs while providing insight to better navigate through supply chain challenges and complexity."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "We face ongoing product and price competition in all areas of our business from both branded and generic competitors, including companies that specialize in one or more of our product or service lines."
  * [2] "We believe that new businesses will continue to enter these markets and develop technologies that, if successfully commercialized, may compete with our products and services."
  * [3] "We also face competition from non-traditional IT companies, including large Infrastructure-as-a-Service providers, that often buy their infrastructure directly from original design manufacturers."
  * [7] "Although we use our patented inventions and license some of them to others, we are not substantially dependent on any single patent or group of related patents."

### FICO: FAIR ISAAC CORP (Information Technology, 10-K filed 2025-11-07)

> Business GENERAL the Company (NYSE: the Company) (together with its consolidated subsidiaries, the “Company,” which may also be referred to in this report as “we,” “us,” “our,” and “the Company”) is a global analytics software leader. We were founded in 1956 on the premise that data, used ...

**Overall: narrow.** The score is the entrenched U.S. credit standard embedded in lending, GSE and underwriting workflows, supported by sticky software and a fraud data network. Very high returns sustained for five years corroborate this. Offsets: intense competition, VantageScore, and reliance on bureaus for distribution. This supports a narrow moat, with the score standard possibly making it wider, but the text flags risks, so narrow.

**Returns: above.** ROIC is 41% to 64% over five years, rising, with operating margins of 38% to 46%, far above the cost of capital. Some inflation from buybacks and leverage lowering the capital base, but margins confirm high profitability.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-09-30 | 2% | 38% | 30% | 32% | 41% | n/a | 25% |
| 2022-09-30 | 5% | 39% | 27% | 37% | 46% | n/a | 25% |
| 2023-09-30 | 10% | 42% | 28% | 31% | 52% | n/a | 28% |
| 2024-09-30 | 13% | 43% | 30% | 36% | 54% | n/a | 31% |
| 2025-09-30 | 16% | 46% | 33% | 39% | 64% | n/a | 36% |

* **intangibles = 2.** The credit score is the de facto U.S. standard, used in most credit decisions by nearly all major lenders, a durable brand/standard position. Weakened somewhat by VantageScore competition and dependence on bureaus.
  * [1] "the standard measure of consumer credit risk in the United States (“U.S.”)"
  * [2] "It is used in most U.S. credit decisions, by nearly all major banks, credit card issuers, mortgage lenders, and auto loan originators."
  * [7] "End users of our products include three-quarters of each of the largest 100 financial institutions in the U.S. and the largest 100 banks in the world."
* **switching_costs = 1.** Software is embedded in mission-critical systems under multi-year subscriptions with minimums, and score-to-risk relationships are built into lenders' underwriting models. Reasonably sticky but not quantified.
  * [4] "Our predictive models are frequently used in mission-critical transactional systems and drive decisions and actions in near real time."
  * [5] "We typically sell our software as multi-year subscriptions, with payments based on usage metrics such as the number of accounts, transactions or decisioning use cases deployed, often subject to contracted minimum payments."
  * [6] "Both scores maintain the same score to risk relationship as standard the Company® Scores, enhancing their compatibility with existing credit underwriting systems and models."
* **network_effects = 1.** The Falcon Intelligence Network pools transaction data from over 10,000 institutions to improve fraud models, a data-network benefit, but limited to one product line.
  * [3] "Our models are continually improved using a proprietary, global data set of transaction data contributed by more than 10,000 institutions that participate in the the Company® Falcon® Intelligence Network."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "The market for our solutions is intensely competitive and is constantly changing."
  * [9] "However, many of our competitors are larger than the Company, have more development, sales and marketing resources than the Company, and some have larger shares of our target geographic or product markets."
  * [10] "During fiscal 2025, 2024 and 2023, revenues generated from our agreements with Experian, TransUnion and Equifax collectively accounted for 51%, 45% and 41% of our total revenues, respectively."
  * [11] "Primary competitors among outside suppliers of scoring models are the three major consumer reporting agencies in the U.S. and Canada, which are also our partners in offering our scoring solutions, and VantageScore (a joint venture entity established by the three major U.S. consumer reporting agencies), which is selling a credit scoring product competitive with our products."

### NTAP: NetApp, Inc. (Information Technology, 10-K filed 2025-06-09)

> Business Overview the Company (the Company, we, us, or the Company) helps customers make their data infrastructure more seamless, more dynamic, and higher performing. We were incorporated in 1992, are headquartered in San Jose, California, and provide a full range of enterprise-class software, ...

**Overall: narrow.** ONTAP has a large installed base and the company holds a unique position embedded in the major public clouds, and returns are very high. But the text describes price competition, a fragmented market, cancelable orders, no long-term commitments and heavy distributor concentration. Switching costs and the cloud position are plausible but not quantified. Sustained very high returns on a mechanism that is described but not quantified support a narrow moat.

**Returns: above.** ROIC is about 67% in the two years reported, with operating margins of 16-20% and ROE above 100%. ROE is inflated by buybacks and a small equity base, but returns are still clearly above the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-04-29 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2022-04-29 | n/a | 18% | 15% | 16% | n/a | >100% | 9% |
| 2023-04-28 | 1% | 16% | 20% | 14% | n/a | >100% | 13% |
| 2024-04-26 | -1% | 19% | 16% | 24% | 67% | 86% | 10% |
| 2025-04-25 | 5% | 20% | 18% | 20% | 67% | >100% | 11% |

* **intangibles = 1.** Embedded status in the major public clouds is a distinctive position, but the company says it is not dependent on any specific IP right, and competitors can develop similar technology.
  * [1] "As the only enterprise-grade storage service natively embedded in the world’s largest clouds, we power data across Amazon AWS, Microsoft Azure, and Google Cloud."
* **switching_costs = 1.** ONTAP is the foundation of customers' storage environments across on-premises and cloud, which implies a sticky installed base. However, the text does not explicitly state switching costs, and customers have no long-term purchase commitments.
  * [2] "ONTAP software is our foundational technology that underpins the Company critical storage solutions in the on-premises data center and in private and public clouds."
  * [3] "Our cloud storage services are based on the same ONTAP data management software that underpins our on-premises ONTAP storage infrastructure offerings."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "We face ongoing product and price competition in all areas of our business, including from both branded- and generic-product competitors."
  * [5] "Our markets are highly competitive, fragmented, and characterized by rapidly changing technology."
  * [6] "We typically do not enter into binding long-term purchase commitments with our customers, resellers, and distributors, meaning there is no guarantee that we will continue to receive large, recurring orders from them."
  * [7] "A substantial portion of our products is sold on the basis of standard purchase orders that are cancelable prior to shipment without penalty."
  * [8] "Sales to customers Arrow Electronics, Inc. and TD Synnex Corporation accounted for 21% and 24% of our net revenues, respectively, in fiscal 2025."
  * [9] "we believe that our business is not materially dependent on any particular patent, trademark, copyright, license or other individual intellectual property right."

### ORCL: ORACLE CORP (Information Technology, 10-K filed 2025-06-18)

> Business the Company provides products and services that address enterprise information technology (IT) needs. Our products and services include enterprise applications and infrastructure offerings that are delivered worldwide through a variety of flexible and interoperable IT deployment models. ...

**Overall: narrow.** The installed base of database and applications customers who renew support gives sticky revenue, and returns are very high. But the company describes intense competition and low barriers in many segments, and it is moving into capital-heavy cloud infrastructure against hyperscalers. That supports a narrow moat built on switching costs.

**Returns: above.** Operating margins of 26-38% and net margins of 16-34% are high, and ROIC is above 100% in the one year reported. ROE is above 100% in some years, but that is inflated by buybacks and low equity. FCF turned negative in 2025 because of heavy cloud capex.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-05-31 | 4% | 38% | 34% | 34% | n/a | >100% | 11% |
| 2022-05-31 | 5% | 26% | 16% | 12% | n/a | n/a | 6% |
| 2023-05-31 | 18% | 26% | 17% | 17% | n/a | n/a | 7% |
| 2024-05-31 | 6% | 29% | 20% | 22% | n/a | >100% | 8% |
| 2025-05-31 | 8% | 31% | 22% | -1% | >100% | 85% | 8% |

* **switching_costs = 1.** Substantially all license customers buy and renew support, and customers run core operations on the Company's database and applications, which points to a large installed base with high switching costs. The text does not quantify this and it states that cloud and open source competition is growing.
  * [1] "Substantially all license support customers renew their support contracts with us upon expiration in order to continue to benefit from technical support services and the periodic issuance of unspecified updates and enhancements, which current license support customers are entitled to receive."
  * [2] "Substantially all customers opt to purchase license support contracts when they purchase an the Company license."
  * [3] "We believe that we can market and sell our the Company Cloud Services offerings together to help new and existing customers migrate their extensive installed base of on-premise and cloud-based applications and infrastructure technologies to the the Company Cloud"
  * [4] "Using the Company technologies, our customers build, deploy, run, manage and support their internal and external products, services and business operations"
  * [7] "Java has been adopted by both independent software vendors (ISVs) that have built their products using Java and by enterprise organizations building custom applications or consuming Java-based ISV products."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "In addition, due to the low barriers to entry in many of our market segments, new technologies and new and growing competitors frequently emerge to challenge our offerings."
  * [9] "We face intense competition in all aspects of our business."
  * [10] "Our enterprise cloud, license and hardware offerings compete directly with certain offerings from some of the largest and most competitive companies in the world"
  * [11] "Our customers are demanding less complexity and lower total cost in the implementation, sourcing, integration and ongoing maintenance of their IT environments."

### PANW: Palo Alto Networks Inc (Information Technology, 10-K filed 2025-08-29)

> General the Company. is a global cybersecurity provider and our vision is a world where each day is safer and more secure than the one before. We were incorporated in 2005 and are headquartered in Santa Clara, California. Our mission is to be the cybersecurity partner of choice for enterprises, ...

**Overall: narrow.** The company has a sticky installed base with renewing subscriptions and a platform strategy that raises integration costs. However, the text describes an intensely competitive market with large rivals and a heavy distributor concentration, and it provides no quantified advantage. Returns have only recently exceeded the cost of capital. This supports a narrow moat at best, driven by modest switching costs.

**Returns: near.** ROIC was negative in FY21-22, then rose from 14% to 17% to 21% by FY25. FCF margins are very high at 33-39%, but returns were only recently above the cost of capital and are partly inflated by stock-based compensation and a history of negative equity.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-07-31 | 25% | -7% | -12% | 33% | -15% | -57% | -5% |
| 2022-07-31 | 29% | -3% | -5% | 33% | -8% | -63% | -2% |
| 2023-07-31 | 25% | 6% | 6% | 38% | 14% | 45% | 3% |
| 2024-07-31 | 16% | 9% | 32% | 39% | 17% | 75% | 15% |
| 2025-07-31 | 15% | 13% | 12% | 38% | 21% | 17% | 5% |

* **switching_costs = 1.** Renewing support contracts and a common PAN-OS across the product line, with an installed base using Panorama, suggest moderate stickiness. Switching costs are not described explicitly and competition is intense.
  * [1] "End-customers typically purchase these services for a one-year or longer term at the time of the initial product sale and typically renew for successive one-year or longer periods."
  * [2] "All of our hardware and software firewalls incorporate the PAN-OS® operating system and include the same rich set of features, ensuring consistent operation across our entire product line."
  * [3] "Many of our existing deployments continue to use Panorama as the security management solution."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "We operate in the intensely competitive enterprise security industry that is characterized by constant change and innovation."
  * [7] "However, some of our competitors may have substantially greater financial, technical, and other resources, greater name recognition, larger sales and marketing budgets, broader distribution, more diversified product lines, and larger and more mature intellectual property portfolios."
  * [8] "For fiscal 2025, 44.2% of our total revenue was derived from sales to three distributors."
  * [9] "As our market grows, it will attract more highly specialized vendors, as well as larger vendors that may continue to acquire or bundle their products more effectively."
  * [10] "Our manufacturing and supply contracts, generally, do not guarantee a certain level of supply or fixed pricing, which increases our exposure to supply shortages or price increases."

### VRSN: VERISIGN INC/CA (Information Technology, 10-K filed 2025-02-13)

> BUSINESS Overview We are a global provider of critical internet infrastructure and domain name registry services, enabling internet navigation for many of the world’s most recognized domain names. We help enable the security, stability, and resiliency of the Domain Name System (“DNS”) and the ...

**Overall: wide.** The Company is the exclusive registry operator for .com and .net under government-linked agreements with presumptive renewal and contractual price-increase rights, plus root server roles. These are durable, structurally protected monopoly positions, and the sustained 65%+ operating margins confirm it. Risks are the ICANN/DOC pricing caps, a theoretical non-renewal, and substitution by social media and platforms, but the franchise should last decades.

**Returns: above.** Operating margins are 65-68% and FCF margins 54-57% for five years, and ROA of 36-50% is very high. ROIC is not reported, but returns are clearly far above the cost of capital, although buybacks and a thin or negative equity base inflate the figures.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 3% | 65% | 64% | 54% | n/a | n/a | 45% |
| 2021-12-31 | 5% | 65% | 59% | 57% | n/a | n/a | 42% |
| 2022-12-31 | 7% | 66% | 47% | 56% | n/a | n/a | 36% |
| 2023-12-31 | 5% | 67% | 55% | 54% | n/a | n/a | 47% |
| 2024-12-31 | 4% | 68% | 50% | 56% | n/a | n/a | 50% |

* **intangibles = 2.** The Company holds the exclusive registry rights for .com and .net under ICANN and DOC agreements, with presumptive renewal and automatic six-year renewals. This is a regulatory franchise competitors cannot replicate, though ICANN can in principle refuse renewal and pricing is capped.
  * [2] "We operate the .com, .net, and .name gTLDs and the IDN gTLDs under registry agreements with ICANN and also, with respect to the .com gTLD, a Cooperative Agreement with the U.S. Department of Commerce (“DOC”)."
  * [3] "Although the .com Registry Agreement contains a “presumptive” right of renewal, ICANN could terminate or refuse to renew the Registry Agreement in certain prescribed circumstances."
  * [4] "On November 30, 2024, the Cooperative Agreement was automatically renewed on the same terms for a successive six-year term and will automatically renew on November 30, 2030, unless the DOC provides written notice of non-renewal within 120 days prior to the end of the then-current term."
  * [5] "The .com Registry Agreement permits an increase to the Maximum Price (as defined in the .com Registry Agreement) of .com domain name registrations by up to 7% over the previous year in each of the final four years of each six-year period."
* **efficient_scale = 2.** It operates two of thirteen root servers and the Root Zone Maintainer function, and the .com/.net registries are single-operator by design. The market supports only one operator for each TLD, and the RZMA runs to 2032 with automatic renewal.
  * [1] "We help enable the security, stability, and resiliency of the Domain Name System (“DNS”) and the internet by providing Root Zone Maintainer services, operating two of the thirteen global internet root servers, and providing registration services and authoritative resolution for the .com and .net top-level domains (“TLDs”), which support the majority of global e-commerce."
  * [6] "The RZMA was renewed on October 20, 2024 and the current term of the RZMA ends on October 20, 2032."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [8] "Under our agreements with ICANN, we are subject to certain restrictions in the operation of .com, .net and .name on pricing, bundling, marketing, methods of distribution, introduction of new registry services, and use of registrars, that do not apply to ccTLDs and other gTLDs and therefore may create a competitive disadvantage."
  * [9] "Demand for domain names could be negatively impacted to the extent end-users establish their online identities using social media (such as Facebook, Instagram or Tiktok) or transact business on e-commerce platforms (such as Amazon, Etsy and Taobao) instead of registering domain names."
  * [10] "Some of the software and protocols used in our business are in the public domain or are otherwise available to our competitors, and some are based on open standards set by organizations such as the Internet Engineering Task Force."

### ALB: ALBEMARLE CORP (Materials, 10-K filed 2025-02-12)

> Business. the Company was incorporated in Virginia in 1993. Our principal executive offices are located at 4250 Congress Street, Suite 900, Charlotte, North Carolina 28209. Unless the context otherwise indicates, the terms “the Company,” “we,” “us,” “our” or “the Company” mean the Company and its ...

**Overall: none.** Resource access (Greenbushes, Atacama, bromine brines) is a plausible but unquantified cost position, and lithium competition is now index-priced with heavy new entry. Catalysts and specialties compete on price and performance. Returns were below the cost of capital in four of five years, with a single cyclical peak, so there is no durable moat shown.

**Returns: below.** ROIC was 6%, 8%, 22%, 2% and -11% over the five years: a lithium price spike in 2022 and losses by 2024. This is volatile and below the cost of capital in most years.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -13% | 16% | 12% | -2% | 6% | 9% | 4% |
| 2021-12-31 | 6% | 24% | 4% | -18% | 8% | 2% | 1% |
| 2022-12-31 | >100% | 34% | 37% | 9% | 22% | 38% | 20% |
| 2023-12-31 | 31% | 3% | 16% | -9% | 2% | 18% | 9% |
| 2024-12-31 | -44% | -33% | -22% | -18% | -11% | -12% | -7% |

* **cost_advantage = 1.** Part-ownership of the Greenbushes mine (a major hard-rock resource) and Atacama brine plus Arkansas brine rights give access to owned resources, which plausibly lowers costs, but the text does not compare costs and lithium is increasingly priced on indices with many new entrants.
  * [4] "We obtain lithium: (a) by purchasing lithium concentrate from our 49%-owned joint venture, Windfield Holdings Pty. Ltd. (“Windfield”), which directly owns 100% of the equity of Talison Lithium Pty. Ltd., a company incorporated in Australia (“Talison”) that owns the Greenbushes mine"
  * [5] "Our bromine production operations in Arkansas are supported by an active brine rights leasing program."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "The global lithium market is highly competitive and growing very rapidly."
  * [2] "Competition in the global lithium market is increasingly based on index-based market pricing and differentiated via product quality, product diversity, reliability of supply and customer service."
  * [3] "Product performance and quality, price and contract terms are the primary factors in determining which qualified supplier is awarded a contract."
  * [8] "most of which are readily available from numerous independent suppliers and are purchased or provided under contracts at prices we believe are competitive."
  * [9] "It is characterized by aggressive expansion and entry from existing and new players, including automotive OEMs, commodity traders, junior miners, and large, well-capitalized diversified miners."

### MOS: MOSAIC CO (Materials, 10-K filed 2025-03-03)

> OVERVIEW The the Company Company is the world’s leading producer and marketer of concentrated phosphate and potash crop nutrients. Through our broad product offering, we are a single source supplier of phosphate- and potash-based crop nutrients and animal feed ingredients. We serve customers in ...

**Overall: none.** The company sells global commodities and competes primarily on price. It has some cost advantages from integration and scale, but returns are cyclical and recently below the cost of capital, so there is no durable moat.

**Returns: below.** ROIC was 2%, 14%, 26%, 7%, 3% over five years: far above the cost of capital only at the 2021-22 price peak, and below it in 2020, 2023 and 2024. Returns track the commodity cycle.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | 5% | 8% | 5% | 2% | 7% | 3% |
| 2021-12-31 | 42% | 20% | 13% | 7% | 14% | 16% | 8% |
| 2022-12-31 | 55% | 25% | 19% | 14% | 26% | 31% | 16% |
| 2023-12-31 | -28% | 10% | 9% | 7% | 7% | 10% | 5% |
| 2024-12-31 | -19% | 6% | 2% | 0% | 3% | 1% | 1% |

* **intangibles = 1.** Only a patented process for MicroEssentials and some feed brands are cited; the products compete in a price-driven commodity market, so the advantage is modest.
  * [9] "MicroEssentials® is a value-added ammoniated phosphate product that is enhanced through a patented process that creates very thin platelets of sulfur and other micronutrients, such as zinc, on the granulated product."
* **cost_advantage = 1.** Vertically integrated phosphate rock, scale, sulfur logistics, and a large low-cost Esterhazy mine support a cost position, but the claims are qualitative ('some competitors') and state-subsidized rivals may be cheaper. Returns are volatile.
  * [5] "We believe that our position as an integrated producer of phosphate rock provides us with a significant cost advantage over competitors that are non-integrated phosphate producers."
  * [6] "As the world’s second largest producer of concentrated phosphate, as well as the second largest miner of phosphate rock in the world and the largest in the U.S., we maintain an advantage over some competitors as the scale of operations effectively reduces production costs per unit."
  * [7] "We believe that our investments in sulfur logistical and melting assets continue to afford us a competitive advantage compared to other producers in cost and access to sulfur."
  * [10] "We believe the strategic location of our mines and chemical plants, in close proximity to our customers, and the benefit of our own distribution network, gives us an advantage over most of our competitors."
  * [11] "Esterhazy, the largest potash mine in the world, has completed full ramp up of capacity and production."
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "Because crop nutrients are global commodities available from numerous sources, crop nutrition companies compete primarily on the basis of delivered price."
  * [2] "Phosphate producers compete primarily based on price, as well as product quality, service and innovation."
  * [3] "Potash is a commodity available from several geographical regions around the world and, consequently, the market is highly competitive."
  * [4] "Foreign competitors may have access to less expensive raw materials, may not have to comply with as stringent regulatory requirements or are owned or subsidized by governments and, as a result, may have cost advantages over North American companies."

### PKG: PACKAGING CORP OF AMERICA (Materials, 10-K filed 2025-02-27)

> B USINESS the Company of America (“we,” “us,” “our,” “the Company,” or the “Company”) is the third largest producer of containerboard products and a leading producer of uncoated freesheet (UFS) paper in North America. We operate eight mills and 86 corrugated products plants and related facilities. ...

**Overall: none.** The text describes a fragmented, price-competitive, cyclical commodity industry with no stated durable advantage. The Company is third largest in containerboard, but the filing itself says no producer dominates. Returns are good, but they come from cycle and operating execution, not a documented moat.

**Returns: above.** ROIC was 12-20% over five years, above the 8-10% cost of capital and peaking in 2021-22. It has fallen to 14% as the cycle turned. This is a strong cyclical showing, but the text describes a commodity, price-driven industry.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | 11% | 7% | 9% | 12% | 15% | 6% |
| 2021-12-31 | 16% | 16% | 11% | 6% | 19% | 25% | 11% |
| 2022-12-31 | 10% | 17% | 12% | 8% | 20% | 28% | 13% |
| 2023-12-31 | -8% | 14% | 10% | 11% | 14% | 20% | 9% |
| 2024-12-31 | 7% | 13% | 10% | 6% | 14% | 19% | 9% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Our industries are highly competitive, with no single containerboard, corrugated packaging, or UFS paper producer having a dominant position."
  * [2] "Certain containerboard grades and UFS paper products cannot generally be differentiated by producer, which tends to intensify price competition."
  * [3] "Commodity grades of UFS paper are globally traded, with numerous worldwide manufacturers, and as a result, these products compete primarily on the basis of price."
  * [4] "We have little influence over the timing and extent of price changes of our products, which may be unpredictable and volatile."
  * [8] "Some of our competitors are larger than we are and may have greater financial and other resources, greater manufacturing economies of scale, greater energy self-sufficiency, or lower operating costs, compared to our company."
  * [9] "According to industry sources, corrugated products are produced by about 380 U.S. companies operating approximately 1,100 plants."

### ESS: ESSEX PROPERTY TRUST, INC. (Real Estate, 10-K filed 2025-02-21)

> OVERVIEW the Company (“the Company”), a Maryland corporation, is an S&P 500 company that operates as a self-administered and self-managed real estate investment trust (“REIT”). the Company owns all of its interest in its real estate and other investments directly or indirectly through the Company ...

**Overall: none.** Supply-constrained West Coast locations offer some protection, but the business faces many housing alternatives and larger competitors, and returns are only near the cost of capital. No durable moat is evident beyond modest location benefits.

**Returns: near.** ROIC was 6-8% in 2020-2023 and 11% in 2024, ROE 7-9% then 14%. For a REIT with a cost of capital of about 5-7%, returns were near the cost of capital, with a 2024 uptick that may reflect one-off gains.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | 33% | 38% | 54% | 6% | 9% | 4% |
| 2021-12-31 | -4% | 37% | 34% | 63% | 6% | 8% | 4% |
| 2022-12-31 | 12% | 37% | 25% | 61% | 7% | 7% | 3% |
| 2023-12-31 | 4% | 35% | 24% | 59% | 8% | 7% | 3% |
| 2024-12-31 | 6% | 40% | 42% | 60% | 11% | 14% | 6% |

* **efficient_scale = 1.** Supply constraints in coastal markets (land scarcity, permitting barriers, natural limits) limit new competing housing, which is a location-based barrier. However, the market is not limited to a few firms, and many competitors and housing alternatives exist, so the advantage is modest.
  * [1] "Constraints on new supply driven by: (i) low availability of developable land sites where competing housing could be economically built; (ii) political growth barriers, such as protected land, urban growth boundaries, and potential lengthy and expensive development permit processes; and (iii) natural limitations to development, such as mountains or waterways;"
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [2] "There are numerous housing alternatives that compete with the Company’s communities in attracting tenants."
  * [3] "Some competitors are larger and have greater resources than the Company."

### IRM: IRON MOUNTAIN INC (Real Estate, 10-K filed 2025-02-14)

> BUSINESS OVERVIEW the Company, a Delaware corporation ("IMI"), was founded in an underground facility near Hudson, New York in 1951 where it stored business records. Today, we are a global leader in information management services, and we are trusted by more than 240,000 customers in 61 countries, ...

**Overall: none.** Recurring contracts and sticky stored records give some customer inertia, but competition is on price among many rivals and returns are mediocre. Data center leasing is competitive. There is some stickiness in records storage, but not enough to establish a durable moat.

**Returns: near.** ROIC was 6-8% over five years, at or below the cost of capital. ROE is inflated by leverage, and FCF margin has turned negative. As a REIT, its cost of capital is lower, about 5-7%, so returns are near that level.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | 23% | 8% | 13% | 8% | 26% | 2% |
| 2021-12-31 | 8% | 19% | 10% | 3% | 7% | 45% | 3% |
| 2022-12-31 | 14% | 21% | 11% | 1% | 8% | 75% | 4% |
| 2023-12-31 | 7% | 17% | 3% | -4% | 6% | 43% | 1% |
| 2024-12-31 | 12% | 16% | 3% | -10% | 6% | n/a | 1% |

* **switching_costs = 1.** Multi-year storage contracts, high retention on a very large stored base of physical records, and long data center leases (10.6-year weighted average) make moving costly. The text does not quantify the advantage, though.
  * [1] "In our Records Management business, our contracted storage rental fee agreements generally range from one to five years in length."
  * [2] "As of December 31, 2024, we stored more than 730 million cubic feet of physical volume and we have consistently experienced strong customer retention levels."
  * [3] "In our Global Data Center Business, our lease durations vary by customer, with a weighted average lease expiration of 10.6 years as of December 31, 2024."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Competition for records and information customers is driven by factors such as pricing, reputation and reliability, the quality and security of storage solutions and the scope and scale of technology."
  * [7] "We also compete with numerous data center developers, owners and operators, many of whom own properties comparable to ours in several of the same metropolitan areas where our facilities are located."
  * [8] "We face competition from numerous storage and information management services providers globally, as well as storage and information management services that are managed and operated internally by organizations."

### O: REALTY INCOME CORP (Real Estate, 10-K filed 2025-02-25)

> Financial Statements the Company CORPORATION AND SUBSIDIARIES CONSOLIDATED BALANCE SHEETS (in thousands, except per share amounts) December 31, 2024 December 31, 2023 ASSETS Real estate held for investment, at cost: Land $ 17,320,520 $ 14,929,310 Buildings and improvements 40,974,535 34,657,094 ...

Not tagged: the located Item 1 'CONSOLIDATED BALANCE SHEETS' in the first 10,000 characters.

### CNP: CENTERPOINT ENERGY INC (Utilities, 10-K filed 2025-02-20)

> Business This combined Form 10-K is filed separately by three registrants: the Company., the Company Houston Electric, LLC and the Company Resources Corp. Information contained herein relating to any individual registrant is filed by such registrant solely on its own behalf. Each registrant makes ...

**Overall: narrow.** Regulated utility with monopoly electric T&D territories (efficient scale and regulatory certificates), but gas franchises are non-exclusive, competition from alternative fuels exists, and returns are only near the cost of capital. The franchise supports a narrow moat but not a wide one.

**Returns: near.** ROIC was about 4-5% over five years, which is near to slightly below the 5-7% cost of capital for utilities. ROE was 9-11% in recent years, near a utility cost of equity, with the 2020 loss being a one-off. Returns are regulated and capped.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -2% | 14% | -10% | -8% | 4% | -9% | -2% |
| 2021-12-31 | 13% | 16% | 18% | -38% | 5% | 17% | 4% |
| 2022-12-31 | 12% | 17% | 11% | -28% | 5% | 11% | 3% |
| 2023-12-31 | -7% | 20% | 11% | -6% | 5% | 9% | 2% |
| 2024-12-31 | -1% | 23% | 12% | -27% | 5% | 10% | 2% |

* **intangibles = 1.** Regulatory certificates (CCN, exclusive Indiana territory certificates) keep competitors out of the electric territories, but gas franchises are mostly non-exclusive and rates are regulated, which caps returns.
  * [2] "For another provider of transmission and distribution services to provide such services in Houston Electric’s territory, it would be required to obtain a CCN from the PUCT and, depending on the location of the facilities, may also be required to obtain franchises from one or more municipalities."
  * [4] "Indiana service territory certificates are exclusive."
* **efficient_scale = 1.** The electric T&D businesses are natural monopolies with no other T&D utility in the service areas, and entry requires regulatory approval, so a new entrant would not earn a return.
  * [1] "There are no other electric transmission and distribution utilities in Houston Electric’s service area."
  * [3] "There are no other electric transmission and distribution utilities in Indiana Electric’s service area."
  * [2] "For another provider of transmission and distribution services to provide such services in Houston Electric’s territory, it would be required to obtain a CCN from the PUCT and, depending on the location of the facilities, may also be required to obtain franchises from one or more municipalities."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [6] "In most cases, franchises to provide natural gas utility services are not exclusive."
  * [7] "the Company and CERC’s Natural Gas compete primarily with alternate energy sources such as electricity and other fuel sources."
  * [8] "Houston Electric does not have long-term contracts with any of its customers, but rather operates using a continuous billing cycle"
  * [9] "Houston Electric holds non-exclusive franchises from certain incorporated municipalities in its service territory."

### LNT: ALLIANT ENERGY CORP (Utilities, 10-K filed 2025-02-21)

> A. GENERAL the Company maintains its principal executive offices in Madison, Wisconsin. the Company operates as a regulated investor-owned public utility holding company, and its purpose-driven strategy is to serve its customers and build stronger communities. the Company primary focus is to ...

**Overall: narrow.** The moat is the regulated franchise: captive retail customers, no choice of supplier, and regulatory barriers to new generation. Returns are near the cost of capital, as expected for a regulated utility. Advantages are durable but returns are capped, and there are no other sources, so the moat is narrow.

**Returns: near.** ROIC of 4-5% and ROE of 10-11% across five years are in line with a regulated utility's cost of capital of about 5-7% (ROIC slightly below, ROE near 10%). These are capped regulated returns, not excess returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -6% | 22% | 18% | -25% | 5% | 11% | 4% |
| 2021-12-31 | 7% | 22% | 18% | -16% | 5% | 11% | 4% |
| 2022-12-31 | 15% | 22% | 16% | -24% | 5% | 11% | 4% |
| 2023-12-31 | -4% | 23% | 17% | 22% | 5% | 11% | 3% |
| 2024-12-31 | -1% | 22% | 17% | 29% | 4% | 10% | 3% |

* **efficient_scale = 1.** Regulated utility with captive retail electric and gas customers and regulatory approval needed for new generation. Entry by a rival is effectively barred in the service territories, though returns are capped by regulators and there is some competition from self-generation and municipalization.
  * [1] "Retail electric customers in Iowa and Wisconsin currently do not have the ability to choose their electric supplier, and IPL and WPL have obligations to serve all their retail electric customers."
  * [2] "Gas customers in Iowa and Wisconsin currently do not have the ability to choose their gas distributor, and IPL and WPL have obligations to serve all their gas customers."
  * [3] "IPL must obtain a certificate of public convenience, use and necessity (GCU Certificate) from the IUC in order to construct a new, or significantly alter (including fuel switching) an existing, EGU or energy storage facility located in Iowa with a nameplate generating capacity of 25 MW or more."
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [5] "IPL and WPL still face competition from self-generation by large industrial customers, customer- and third party-owned generation (e.g. solar panels), alternative energy sources, and petitions to municipalize (Iowa) as well as service territory expansions by municipal utilities through annexations (Wisconsin)."
  * [7] "In addition, the wholesale power market is competitive and IPL and WPL compete against independent power producers, other utilities and MISO market purchases to serve wholesale customers for their electric energy and capacity needs."

### PEG: PUBLIC SERVICE ENTERPRISE GROUP INC (Utilities, 10-K filed 2025-02-25)

> BUSINESS We were incorporated under the laws of the State of New Jersey in 1985 and our principal executive offices are located at 80 Park Plaza, Newark, New Jersey 07102. We are a public utility holding company that, acting through our wholly owned subsidiaries, is a predominantly regulated ...

**Overall: narrow.** The regulated franchise (PSE&G, the majority of earnings) provides an efficient-scale moat with stable but capped returns, so the franchise is the moat. The merchant nuclear business is exposed to market pricing, supported partly by the PTC. Regulatory risk and uneven returns limit it, so the moat is narrow.

**Returns: near.** ROIC of 6%, -2%, 4%, 9%, 5% and ROE of 12%, -4%, 7%, 18%, 11% are volatile, driven by the merchant nuclear business and the 2021 loss. For a regulated utility the cost of capital is about 5-7%, so returns are near that level, as expected for capped returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -5% | 24% | 20% | 2% | 6% | 12% | 4% |
| 2021-12-31 | 1% | -9% | -7% | -10% | -2% | -4% | -1% |
| 2022-12-31 | 1% | 14% | 11% | -14% | 4% | 7% | 2% |
| 2023-12-31 | 15% | 33% | 23% | 4% | 9% | 18% | 5% |
| 2024-12-31 | -8% | 23% | 17% | -12% | 5% | 11% | 3% |

* **efficient_scale = 2.** PSE&G holds a franchised, regulated territory with about 6.8 million people, with transmission and distribution earning approved returns on a rate base of $17.8 billion. This is a classic natural monopoly with capped returns, supported by the formula rate and revenue decoupling.
  * [1] "PSE&G—A New Jersey corporation, incorporated in 1924, which is a franchised public utility in New Jersey."
  * [2] "PSE&G distributes electricity and natural gas to end users in our respective franchised service territories."
  * [3] "We serve the most densely populated, commercialized and industrialized territory in New Jersey, including its six largest cities and approximately 300 suburban and rural communities."
  * [4] "Our transmission revenues are not impacted by sales volumes."
  * [5] "The Order provides for a $17.8 billion rate base, a 9.6% return on equity for PSE&G’s distribution business and a 55% equity component of its capitalization structure."
  * [6] "Our T&D business is not affected when customers choose alternate electric or gas suppliers since we earn our return on our net investment in rate base to provide T&D service, not by supplying the commodity."
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [9] "Typically, the bid price of the last unit dispatched by PJM establishes the energy market-clearing price."
  * [10] "Our competitors include but are not limited to merchant generators, utility generators, energy marketers, retailers, private equity firms, and other financial entities."
  * [11] "Changes previously ordered by FERC and implemented by PJM and other ISOs to eliminate contractual provisions that previously provided us a “right of first refusal” to construct new transmission projects in our service territory could result in third-party construction of transmission lines in our area in the future"
