# Moat tagger v2: set `sample`

Generated 2026-11-02 18:12 by `scripts/moat_pilot.py`, set `sample`, model `claude-sonnet-5-5`, prompts `claude-sonnet-5-5__c2d905646__high`. One call per company over Item 1 (business) and five years of financials, prompt v2: quotes checked word for word against the text, each moat source rated from the verified quotes only (`auto_researcher.llm.moat`).

* Companies: 50 (47 rated); chunks read 47 (Item 1 cut at 16 chunks for 0).
* Quotes returned: 421; kept 418; not found in the text 3 (0.7%); duplicates 0; over the cap 0; unparseable replies 0.
* Kept quotes by kind: intangibles 96, switching_costs 49, network_effects 12, cost_advantage 42, efficient_scale 26, against 193.
* Rating fixes (citations of missing or "against" quotes dropped, uncited ratings lowered to 0): 2.
* Model time: 7 min (8 s per company); API cost $2.20; served by another model 0.

## Ratings

0 = none or generic, 1 = specific but modest, 2 = strong and durable.

| Ticker | Sector | Overall | intang | switch | network | cost | eff.scale | Against quotes | Returns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MTCH | Communication Services | narrow | 1 | 0 | 1 | 0 | 0 | 5 | above |
| TTWO | Communication Services | ? | ? | ? | ? | ? | ? | 0 | ? |
| ABNB | Consumer Discretionary | narrow | 1 | 0 | 1 | 0 | 0 | 4 | above |
| CCL | Consumer Discretionary | none | 1 | 0 | 0 | 1 | 1 | 2 | below |
| CMG | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 4 | above |
| GM | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 5 | below |
| MAR | Consumer Discretionary | narrow | 1 | 1 | 0 | 0 | 0 | 3 | above |
| CAG | Consumer Staples | none | 0 | 0 | 0 | 0 | 0 | 5 | below |
| DG | Consumer Staples | narrow | 0 | 0 | 0 | 1 | 1 | 3 | near |
| MO | Consumer Staples | narrow | 2 | 0 | 0 | 0 | 0 | 3 | above |
| TSN | Consumer Staples | none | 0 | 0 | 0 | 0 | 0 | 5 | below |
| EOG | Energy | none | 0 | 0 | 0 | 1 | 0 | 4 | above |
| BBT | Financials | none | 0 | 0 | 0 | 0 | 0 | 5 | below |
| COF | Financials | none | 0 | 0 | 0 | 0 | 0 | 4 | near |
| EG | Financials | none | 1 | 0 | 0 | 0 | 0 | 6 | near |
| MCO | Financials | narrow | 1 | 1 | 0 | 0 | 0 | 3 | above |
| NTRS | Financials | none | 0 | 0 | 0 | 0 | 0 | 2 | near |
| PYPL | Financials | narrow | 1 | 0 | 1 | 0 | 0 | 3 | above |
| SCHW | Financials | narrow | 1 | 1 | 0 | 1 | 0 | 2 | near |
| ABBV | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 | above |
| DGX | Health Care | ? | ? | ? | ? | ? | ? | 0 | ? |
| GILD | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 | above |
| HUM | Health Care | none | 0 | 0 | 0 | 1 | 0 | 4 | near |
| IDXX | Health Care | narrow | 1 | 1 | 0 | 0 | 0 | 4 | above |
| MOH | Health Care | narrow | 0 | 0 | 0 | 0 | 1 | 6 | above |
| REGN | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 | above |
| ALLE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 4 | above |
| AME | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 4 | above |
| CHRW | Industrials | narrow | 0 | 0 | 1 | 1 | 0 | 5 | above |
| DE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 4 | above |
| LII | Industrials | none | 0 | 0 | 0 | 0 | 0 | 3 | above |
| PWR | Industrials | none | 0 | 1 | 0 | 0 | 0 | 4 | near |
| RSG | Industrials | narrow | 0 | 1 | 0 | 1 | 1 | 4 | near |
| UAL | Industrials | none | 0 | 0 | 1 | 0 | 0 | 5 | unclear |
| CDW | Information Technology | narrow | 0 | 0 | 0 | 1 | 0 | 5 | above |
| DELL | Information Technology | narrow | 0 | 1 | 0 | 1 | 0 | 5 | above |
| FICO | Information Technology | wide | 2 | 1 | 1 | 0 | 0 | 5 | above |
| NTAP | Information Technology | narrow | 1 | 0 | 0 | 0 | 0 | 6 | above |
| ORCL | Information Technology | narrow | 1 | 2 | 0 | 0 | 0 | 3 | above |
| PANW | Information Technology | narrow | 0 | 1 | 0 | 0 | 0 | 5 | above |
| VRSN | Information Technology | wide | 2 | 1 | 0 | 0 | 2 | 3 | above |
| ALB | Materials | none | 0 | 0 | 0 | 1 | 0 | 5 | below |
| MOS | Materials | none | 1 | 0 | 0 | 1 | 0 | 4 | below |
| PKG | Materials | none | 0 | 0 | 0 | 0 | 0 | 6 | above |
| ESS | Real Estate | none | 0 | 0 | 0 | 0 | 1 | 3 | near |
| IRM | Real Estate | none | 0 | 1 | 0 | 0 | 0 | 2 | below |
| O | Real Estate | ? | ? | ? | ? | ? | ? | 0 | ? |
| CNP | Utilities | none | 1 | 0 | 0 | 0 | 1 | 4 | below |
| LNT | Utilities | none | 1 | 0 | 0 | 0 | 1 | 3 | below |
| PEG | Utilities | narrow | 1 | 0 | 0 | 0 | 2 | 4 | below |

Not tagged, because the located Item 1 is another part of the 10-K: TTWO (starts mid-sentence); DGX ('Critical Accounting Policies' in the first 10,000 characters); O ('CONSOLIDATED BALANCE SHEETS' in the first 10,000 characters).

| Sector | none | narrow | wide |
| --- | ---: | ---: | ---: |
| Communication Services | 0 | 1 | 0 |
| Consumer Discretionary | 3 | 2 | 0 |
| Consumer Staples | 2 | 2 | 0 |
| Energy | 1 | 0 | 0 |
| Financials | 4 | 3 | 0 |
| Health Care | 1 | 5 | 0 |
| Industrials | 3 | 5 | 0 |
| Information Technology | 0 | 5 | 2 |
| Materials | 3 | 0 | 0 |
| Real Estate | 2 | 0 | 0 |
| Utilities | 2 | 1 | 0 |

## Against the hand check of v1

The reviewer judged the v1 tags; each verdict implies a target for the new tags ("too low" is met by any higher call, "too high" by any lower one, "agree" by the same call). 47 companies with the right Item 1 text in both runs. The reviewed v1 run is favoured by construction (the verdicts were written about its exact calls); a second run of the same v1 prompt shows how many targets a run misses by chance.

| Tags | Overall calls on target | Source ratings on target |
| --- | ---: | ---: |
| v1, the reviewed run | 40 (85%) | 215 of 235 (91%) |
| v1, run 2 (same prompt) | 36 (77%) | 207 of 235 (88%) |
| v2 (with financials) | 34 (72%) | 203 of 235 (86%) |

Companies whose overall call changed or misses its target:

| Ticker | v1 | Verdict on v1 | v2 | Returns (v2) | On target |
| --- | --- | --- | --- | --- | --- |
| CCL | narrow | high | none | below | yes |
| CMG | none | low | none | above | no |
| MCO | wide | agree | narrow | above | no |
| NTRS | none | low | none | near | no |
| IDXX | narrow | low | narrow | above | no |
| MOH | none | agree | narrow | above | no |
| AME | none | low | narrow | above | yes |
| CHRW | none | agree | narrow | above | no |
| LII | none | low | none | above | no |
| CDW | none | agree | narrow | above | no |
| DELL | none | agree | narrow | above | no |
| FICO | narrow | low | wide | above | yes |
| NTAP | none | agree | narrow | above | no |
| IRM | narrow | agree | none | below | no |
| CNP | narrow | agree | none | below | no |
| LNT | narrow | agree | none | below | no |

## Notes

* **What changed from v1.** The prompt adds the company's financial history and a step judging returns on capital before the overall call (`ONE_CALL_SYSTEM_V2`); sources are still rated from quotes only. Item 1 texts that start mid-sentence or hold MD&A or financial statement headings early are not tagged (`item1_problem`).
* **Tuned on the hand-checked set, then frozen.** Against the targets the 47 v1 verdicts imply, v2 (fixed rules: high returns plus a source rated 1 meant narrow, one 8-10% cost of capital) put 34 overall calls on target and moved weak resellers up and utilities down; v3 (returns corroborate the text, a lower cost of capital for utilities and REITs) put 39. A second run of the unchanged v1 prompt put 36, and 4 of 47 calls changed between the two identical v1 runs, so v3's lead is small. v3 was committed (bfcd03b) before the fresh set was tagged once.
* **The fresh set** is 50 companies in no earlier set, drawn the same way (sector-stratified, seed 1) from the 2025 10-Ks that pass the Item 1 check; its tags await the same hand check.
* **The financial history is point-in-time:** SEC XBRL company facts filed on or before the 10-K's filing date (the 10-K's own figures included), the latest value known then for each fiscal year (`data.sec_fundamentals.build_pit_table`). ROIC is operating income after a 21% tax over average equity + debt - cash; ratios on negative or near-zero capital are shown as n/a.

## Review sheet

For each company: what it does (the opening of Item 1, name hidden), the overall call, then each source rated above 0 with the model's reason and the quotes it cites, and the quotes against a moat. Sources rated 0 are listed by name only.

### MTCH: Match Group, Inc. (Communication Services, 10-K filed 2025-02-27)

> Who we are the Company, through its portfolio companies, is a leading provider of digital technologies designed to help people make meaningful connections. Our global portfolio of brands includes Tinder ® , Hinge ® , the Company ® , Meetic ® , OkCupid ® , PairsTM, Plenty Of Fish ® , Azar ® , BLK ® ...

**Overall: narrow.** Returns are consistently high, backed by strong brands and a modest user-community network effect (Tinder and Hinge scale). But the market is fragmented, users multi-home with no switching costs, and Apple and Google take a large share of revenue. Several modest sources plus above-cost returns support a narrow moat, not wide.

**Returns: above.** ROIC of 14-29% over five years, consistently above a 8-10% cost of capital, with operating margins of 16-31% and a dip in 2022. Revenue growth has slowed to 3%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 17% | 31% | 7% | 31% | 15% | n/a | 3% |
| 2021-12-31 | 25% | 29% | 9% | 28% | 29% | n/a | 7% |
| 2022-12-31 | 7% | 16% | 11% | 15% | 14% | n/a | 8% |
| 2023-12-31 | 6% | 27% | 19% | 25% | 25% | n/a | 15% |
| 2024-12-31 | 3% | 24% | 16% | 25% | 23% | n/a | 12% |

* **intangibles = 1.** Strong dating brands (Tinder, Hinge, Meetic, Pairs) with leading positions, and a patented Swipe feature, though patents are modest and brand loyalty is limited since users multi-home.
  * [1] "Brand is very important. Users generally associate strong brands with a higher likelihood of success and more tools to help the user connect safely and securely."
  * [3] "Tinder®was launched in 2012 and has since risen to scale and popularity faster than any other service in the online dating category."
  * [4] "Meetic is the most recognized dating app for singles over age 35 in Europe."
  * [5] "Tinder’s patented Swipe®technology has led to significant adoption, particularly among 18 to 25 year-old users"
* **network_effects = 1.** Text says successful brands depend on large active communities of users, implying a user-side network effect for matching, but communities are fragmented and users use multiple apps, so the advantage is modest.
  * [2] "Generally, successful brands depend on large, active communities of users, strong algorithmic filtering technology, and awareness of successful usage among similar users."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The industry for social connection apps is competitive and has no single, dominant brand globally."
  * [7] "As a result, the market for social connection apps is fragmented, and no single service has been able to effectively serve all of those seeking social connections."
  * [8] "A large portion of customers use multiple services over a given period of time, either concurrently or sequentially, making our broad portfolio of brands a competitive advantage."
  * [9] "We pay Apple and Google a meaningful share of the revenue we receive from in-app transactions."
  * [10] "Both Apple and Google believe they have broad discretion to unilaterally change the terms and conditions governing their respective app stores"
* Fixes: intangibles: dropped citations [11].

### TTWO: TAKE TWO INTERACTIVE SOFTWARE INC (Communication Services, 10-K filed 2025-05-20)

> for additional discussion. Impairments During the fiscal year ended March 31, 2025, we recognized Goodwill impairment charges of $3,545.2, representing a partial impairment related to one of our reporting units, and we recognized impairment charges of $137.0 for acquisition-related Developed Game ...

Not tagged: the located Item 1 starts mid-sentence.

### ABNB: Airbnb, Inc. (Consumer Discretionary, 10-K filed 2025-02-13)

> Overview We operate a global marketplace, where hosts offer guests stays and experiences on our platform. We are a community based on connection and belonging—a community that was born in 2007 when two hosts welcomed three guests to their San Francisco home, and has since grown to over 5 million ...

**Overall: narrow.** Returns are very high and consistent since 2022. The text supports a network effect from the two-sided marketplace and a brand that lowers marketing needs. Competition is intense, hosts and guests can multi-home, and local regulation is a risk, so the advantages are modest. That supports a narrow moat, not a wide one.

**Returns: above.** ROIC was 59% in 2024, far above the cost of capital. ROE was 32-70% in 2022-2024 and operating margin was 15-23%. FCF margins held around 40% for four years. 2020-2021 were hit by the pandemic, and the history is short.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -30% | <-100% | <-100% | -23% | n/a | n/a | -44% |
| 2021-12-31 | 77% | 7% | -6% | 38% | n/a | -9% | -3% |
| 2022-12-31 | 40% | 21% | 23% | 41% | n/a | 37% | 13% |
| 2023-12-31 | 18% | 15% | 48% | 39% | n/a | 70% | 26% |
| 2024-12-31 | 12% | 23% | 24% | 41% | 59% | 32% | 13% |

* **intangibles = 1.** The brand is said to reduce reliance on performance marketing, a direct if unquantified statement of brand pull. Patents and trademarks are only generic claims.
  * [2] "the strength of the the Company brand and our communications strategy allows us to be less reliant on performance marketing."
* **network_effects = 1.** A two-sided marketplace with over 5 million hosts and 2 billion guest arrivals means more listings attract more guests and vice versa. The text states this only loosely, and competitors like OTAs also have large networks, so the advantage is substantial but not certain.
  * [1] "Our technology platform powers our two-sided marketplace and enables our global network of hosts and guests."
  * [3] "has since grown to over 5 million hosts who have welcomed over 2 billion guest arrivals in almost every country and region across the globe."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "We operate in a highly competitive environment."
  * [5] "We compete to attract and retain hosts to and on our platform to list their homes and experiences, as hosts have a range of options for doing so."
  * [6] "We compete to attract and retain guests to and on our platform, as guests have a range of options to find and book accommodations and experiences."
  * [7] "For example, New York City passed regulations in 2023, which resulted in a de facto ban of short-term rental activities."

### CCL: CARNIVAL CORP (Consumer Discretionary, 10-K filed 2025-01-27)

> Business . A. Overview I. Summary the Company was incorporated in Panama in 1974 and the Company was incorporated in England and Wales in 2000. the Company and the Company operate a dual listed company (“DLC”), whereby the businesses of the Company and the Company are combined through a number of ...

**Overall: none.** The Company is the largest cruise operator in a concentrated industry with brands and scale, but returns are only now reaching the cost of capital after the pandemic, and cruising competes with land vacations and cyclical demand. Advantages are modest and not yet visible in returns, so no moat.

**Returns: below.** ROIC was deeply negative 2020-2022 (pandemic), 4% in 2023 and 8% in 2024, at or below the cost of capital. Margins have recovered but history is volatile and ROE is boosted by leverage.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-11-30 | -73% | <-100% | <-100% | <-100% | -20% | -45% | -21% |
| 2021-11-30 | -66% | <-100% | <-100% | <-100% | -16% | -58% | -18% |
| 2022-11-30 | >100% | -36% | -50% | -54% | -10% | -63% | -12% |
| 2023-11-30 | 77% | 9% | -0% | 5% | 4% | -1% | -0% |
| 2024-11-30 | 16% | 14% | 8% | 5% | 8% | 24% | 4% |

* **intangibles = 1.** Brands with sizable share and heritage (Carnival, Princess, AIDA, P&O UK) exist, but the text does not show that customers pay a price premium or that brands block competitors; modest evidence.
  * [2] "the Company Cruise Line is “The World’s Most Popular Cruise Line®”"
  * [3] "AIDA is the most recognized brand in the German cruise market."
  * [4] "P&O Cruises (UK) is Britain’s largest cruise line and its heritage can be traced back over 185 years."
* **cost_advantage = 1.** Scale as the largest cruise company, leveraged in procurement, plus owned private destinations and berth access, give some cost advantage, though not quantified relative to peers.
  * [5] "We approach our spend strategically and look for suppliers who demonstrate the ability to help us leverage our scale in terms of cost, quality, service, innovation and sustainability."
  * [6] "In addition, to secure preferential berth access to third-party ports, we enter into berthing agreements and commitments."
  * [7] "In addition to our cruise operations, we own Holland America Princess Alaska Tours, the leading tour company in Alaska and the Canadian Yukon, which complements our Alaska cruise operations."
* **efficient_scale = 1.** Four players hold about 80% of capacity, an oligopoly with high capital needs for ships, but capacity is growing and no cap on rivals exists; the text shows concentration, not a limit on profitable entry.
  * [1] "we, along with our principal cruise competitors Royal Caribbean Group, Norwegian Cruise Line Holdings, Ltd. and MSC Cruises, represented approximately 80% of the cruise industry capacity."
* Rated 0: switching_costs, network_effects.
* Against a moat:
  * [9] "We compete with land-based vacation alternatives throughout the world, such as hotels, resorts (including all-inclusive resorts), theme parks, organized tours, casinos, vacation ownership properties, and other internet-based alternative lodging sites."
  * [10] "Our travel agents’ relationships are generally not exclusive and travel agents generally receive a base commission, plus the potential of additional commissions"

### CMG: CHIPOTLE MEXICAN GRILL INC (Consumer Discretionary, 10-K filed 2025-02-05)

> General the Company., a Delaware corporation, together with its subsidiaries (“the Company,” “we,” “us,” or “our”) owns and operates the Company restaurants, which feature a relevant menu of burritos, burrito bowls (a burrito without the tortilla), quesadillas, tacos, and salads. We strive to ...

**Overall: none.** Returns are very high, but the text shows no source of advantage. Competition is intense, rivals claim similar quality, and customers switch easily. High returns with no mechanism in the text are not a moat on their own, so I judge there is none. Brand strength may exist but is not evidenced here.

**Returns: above.** ROIC rose from 18% in 2020 to 56% in 2024, far above the 8-10% cost of capital. Operating margin expanded from 5% to 17% with steady growth of 14-26% a year. The returns have been consistently high and improving over the five years.

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
  * [4] "Competition from food delivery services, which offer meals from a wide variety of restaurants, also has increased in recent years and is expected to continue to increase."

### GM: General Motors Co (Consumer Discretionary, 10-K filed 2025-01-28)

> the Company Company (sometimes referred to as we, our, us, ourselves, the Company, the Company or the Company) was incorporated as a Delaware corporation in 2009. We design, build and sell trucks, crossovers, cars and automobile parts and provide software-enabled services and subscriptions ...

**Overall: none.** A mass-market, cyclical automaker competing on price, design and incentives, with falling market share in China and South America and no stated durable advantage. ROIC stays below the cost of capital, so there is no moat.

**Returns: below.** ROIC of 4-6% over five years is below the 8-10% cost of capital. Operating margins of 5-7% are thin. ROE of 9-17% is boosted by leverage, including the finance arm.

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
  * [3] "Market leadership in individual countries in which we compete varies widely."
  * [4] "While no single piece of intellectual property is individually material to our business as a whole, our intellectual property is important to our operations and continued technological development."
  * [5] "Certain fleet transactions, particularly sales to daily rental car companies, are generally less profitable than retail sales to end customers."

### MAR: MARRIOTT INTERNATIONAL INC /MD/ (Consumer Discretionary, 10-K filed 2025-02-11)

> Overview We are a worldwide operator, franchisor, and licensor of hotel, residential, timeshare, and other lodging properties under more than 30 brand names at different price and service points. Consistent with our focus on management, franchising, and licensing, we own or lease very few of our ...

**Overall: narrow.** The asset-light franchise and management model, a large brand portfolio and loyalty program, and long contracts support high returns on capital (ROIC about 27%). Competition from Hilton, Hyatt, IHG, Airbnb and online travel agencies, and owner termination rights, limit durability, so the moat is narrow.

**Returns: above.** ROIC was 27-29% in 2022-2024, well above the cost of capital, after the 2020 pandemic collapse (1%). Margins are healthy, with an asset-light model. The returns are steady since the recovery but the history is short and the 2020 drop shows cyclicality.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -50% | 1% | -3% | 14% | 1% | -47% | -1% |
| 2021-12-31 | 31% | 13% | 8% | 7% | 14% | >100% | 4% |
| 2022-12-31 | 50% | 17% | 11% | 10% | 27% | >100% | 9% |
| 2023-12-31 | 14% | 16% | 13% | 11% | 29% | n/a | 12% |
| 2024-12-31 | 6% | 15% | 9% | 8% | 27% | n/a | 9% |

* **intangibles = 1.** Large brand portfolio with 17% U.S. share and a loyalty program driving most room nights, which attracts owners and guests. Brand strength is claimed, though many strong rivals exist, so the advantage is solid but not overwhelming.
  * [3] "In 2024, 72 percent of our U.S. hotel room nights and 65 percent of our global hotel room nights were booked by Loyalty Program members."
  * [4] "We believe that our Loyalty Program generates substantial repeat business that might otherwise go to competing properties."
  * [5] "Based on lodging industry data, we have an approximately 17 percent share of the U.S. hotel market and a four percent share of the hotel market outside the U.S. (based on number of rooms)."
  * [6] "We believe that our brands are attractive to hotel owners seeking a management company, franchise, or other licensing affiliation because of the benefits of our Loyalty Program, centralized reservation systems, marketing programs, and other offerings, and our emphasis on guest service."
* **switching_costs = 1.** Long management (20-30 yr) and franchise (10-20 yr) agreements lock in owners, and hotel conversions are costly. Owners can terminate for poor performance, which limits the lock-in.
  * [1] "Such agreements are generally for initial periods of 20 to 30 years, with options for us to renew for up to 10 or more additional years."
  * [2] "Such agreements are generally for periods of 10 to 20 years."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Many of our management agreements also permit the hotel owners to terminate the agreement if we do not meet certain performance metrics, financial returns fail to meet defined levels for a period of time, and we have not cured those deficiencies."
  * [8] "we compete against many other companies with strong brands and guest appeal, including Hilton, IHG Hotels & Resorts, Hyatt, Wyndham Hotels & Resorts, Accor, Choice Hotels, Best Western Hotels & Resorts, and others."
  * [9] "We encounter strong competition in the short-term lodging market from regional, national, and international chains that operate lodging properties or franchise their brands, lodging properties that are not affiliated with a chain, and online platforms, including Airbnb and Vrbo"

### CAG: CONAGRA BRANDS INC. (Consumer Staples, 10-K filed 2025-07-10)

> General Development of Business the Company (the “Company”, “the Company”, “we”, “us”, or “our”), headquartered in Chicago, is one of North America’s leading branded food companies. We combine a 100-year history of making quality food with agility and a relentless focus on collaboration and ...

**Overall: none.** Branded food portfolio with intense competition, a heavy Walmart customer concentration, commodity inputs and larger rivals. No source is directly evidenced, and ROIC has been below the cost of capital in most recent years, so there is no moat.

**Returns: below.** ROIC fell from 10% to 4-5% in fiscal 2023-2024 and recovered only to 7% in 2025, at or below the 8-10% cost of capital. Margins have been volatile and revenue is declining.

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
  * [2] "Our food items compete with widely advertised, well-known, branded food, as well as private branded and customized food items."
  * [3] "Some of our competitors are larger and have greater resources than we have."
  * [4] "Our largest customer, Walmart, Inc. and its affiliates, accounted for approximately 29% of consolidated net sales for fiscal 2025 and 28% for fiscal 2024 and 2023."
  * [5] "We use many different raw materials, most of which are commodities, to make and package our products."

### DG: DOLLAR GENERAL CORP (Consumer Staples, 10-K filed 2025-03-21)

> ​ General ​ We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025, with the greatest concentration of stores in the southern, southwestern, midwestern and eastern United States. Our first stores ...

**Overall: narrow.** The company has scale-based purchasing and cost advantages and a locational niche in small towns. But the market is intensely competitive, larger rivals have more resources, and returns and margins have fallen steadily to about the cost of capital. A narrow moat is just supportable given the long record of high returns and the density of its store network, but the erosion is a concern.

**Returns: near.** ROIC was well above the cost of capital from 2021 to 2023 (30%, 26%, 24%) but has fallen sharply to 15% and then 10%. Operating margin dropped from 11% to 4%. The trend shows erosion, and the latest return is only at the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-29 | 22% | 11% | 8% | 8% | 30% | 40% | 11% |
| 2022-01-28 | 1% | 9% | 7% | 5% | 26% | 37% | 9% |
| 2023-02-03 | 11% | 9% | 6% | 1% | 24% | 41% | 9% |
| 2024-02-02 | 2% | 6% | 4% | 2% | 15% | 27% | 6% |
| 2025-01-31 | 5% | 4% | 3% | 4% | 10% | 16% | 4% |

* **cost_advantage = 1.** The company cites low-cost operations and purchasing scale, and it is the largest discount retailer by store count. But it admits that competitors such as Walmart have greater resources and may get better supplier terms, so the advantage is modest and uncertain.
  * [1] "Our ability to offer everyday low prices on quality merchandise is supported by our low-cost operating approach and our strategy to maintain a limited number of items per merchandise category, which we believe helps us maintain strong purchasing power."
  * [2] "Purchasing large volumes of merchandise within our focused assortment in each merchandise category allows us to keep our average product costs low, contributing to our ability to offer competitive everyday prices to our customers."
  * [6] "We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025"
* **efficient_scale = 1.** Small towns with limited alternatives may support only one small-box store, which offers some local protection. But the claim is soft and competitors operate widely nearby.
  * [3] "We believe our ability to effectively deliver both value and convenience allows us to succeed in small markets with limited shopping alternatives, as well as in larger and more competitive markets."
  * [4] "approximately 80% of our stores are located in towns of 20,000 or fewer people"
* Rated 0: intangibles, switching_costs, network_effects.
* Against a moat:
  * [7] "We operate in the basic discount consumer goods market, which is highly competitive with respect to price, customers, store location, merchandise quality, assortment and presentation, service offerings, in-stock consistency, customer service, promotional activity, employees, and market share."
  * [8] "Certain of our competitors have greater financial, distribution, marketing and other resources than we do and may be able to secure better arrangements from suppliers than we can."
  * [9] "Competition is intense and we believe it will continue to be so, with certain competitors reducing their store locations while others move into or increase their presence in our geographic and product markets"

### MO: ALTRIA GROUP, INC. (Consumer Staples, 10-K filed 2025-02-26)

> General Development of Business When used in this Annual Report on Form 10-K (“Form 10-K”), the terms “the Company,” “we,” “us” and “our” refer to either (i) the Company and its consolidated subsidiaries or (ii) the Company only and not its consolidated subsidiaries, as appropriate in the context. ...

**Overall: narrow.** Brand strength (Marlboro, Copenhagen) with long-standing leadership and loyalty shows up in very high, steady returns and margins despite declining volumes. The mechanism is mainly the intangible brand, helped by marketing restrictions that favor incumbents. Secular decline in smoking, regulatory risk and litigation limit durability, so the moat is narrow rather than wide.

**Returns: above.** ROIC is very high and rising, from 29% to 46-48% over five years, against a cost of capital of 8-10%. Operating margins are 42-47% and stable, and FCF margins are 31-37%, even though revenue is shrinking by 1-4% a year.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 4% | 42% | 17% | 31% | 29% | 97% | 9% |
| 2021-12-31 | -1% | 44% | 10% | 32% | 37% | n/a | 6% |
| 2022-12-31 | -4% | 47% | 23% | 32% | 46% | n/a | 15% |
| 2023-12-31 | -2% | 47% | 33% | 37% | 48% | n/a | 22% |
| 2024-12-31 | -2% | 47% | 47% | 36% | 46% | n/a | 31% |

* **intangibles = 2.** Marlboro has been the top-selling US cigarette brand for 50 years, and Copenhagen/Skoal lead moist smokeless. Brand loyalty is stated as a key competitive factor, and this supports pricing power. Regulatory limits on advertising and the FDA authorization process also protect incumbents. Volume is declining, though.
  * [1] "PM USA is the largest cigarette company in the United States and substantially all cigarettes are manufactured and sold to customers in the United States."
  * [2] "has been the largest-selling cigarette brand"
  * [3] "USSTC is the leading producer and marketer of MST products."
  * [4] "The market for tobacco products is highly competitive, characterized by brand recognition and loyalty, with product quality, taste, price, product innovation, marketing, packaging and distribution constituting the significant methods of competition."
  * [5] "NJOY’s commercialized product portfolio of tobacco and menthol e-vapor products is fully covered by marketing granted orders (“MGO”) from the U.S. Food and Drug Administration (“FDA”)."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Total smokeable products segment’s cigarettes shipment volume in the United States was 68.6 billion units in 2024, a decrease of 10.2% from 2023."
  * [7] "we recognize that the decreasing social acceptance of tobacco usage may impact our ability to attract and retain talent"
  * [8] "Middleton contracts with a third-party importer to supply substantially all of its cigars"

### TSN: TYSON FOODS, INC. (Consumer Staples, 10-K filed 2025-11-10)

> GENERAL the Company and its subsidiaries (collectively, the “Company,” “we,” “us,” “our,” “the Company” or “the Company”) (NYSE: the Company) is a world-class food company and recognized leader in protein. Founded in 1935 by John W. the Company, it has grown under four generations of family ...

**Overall: none.** The business is commodity-based with short-term customer contracts, price-based competition and a heavy dependence on Walmart. The text shows no source of advantage that is directly stated, and recent returns are below the cost of capital, with only the peak years above it. There is no moat.

**Returns: below.** ROIC was 13-14% in 2021-22 but fell to -1% in 2023 and 3-4% in 2024-25, well below the cost of capital. Margins and returns are very volatile, typical of a commodity cycle.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-10-02 | 9% | 9% | 6% | 6% | 14% | 18% | 9% |
| 2022-10-01 | 13% | 8% | 6% | 2% | 13% | 17% | 9% |
| 2023-09-30 | -1% | -1% | -1% | -0% | -1% | -3% | -2% |
| 2024-09-28 | 1% | 3% | 2% | 3% | 4% | 4% | 2% |
| 2025-09-27 | 2% | 2% | 1% | 2% | 3% | 3% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "Our agreements with customers are generally short-term, primarily due to the nature of our products, industry practices and fluctuations in supply, demand and price for such products."
  * [4] "The principal competitive elements are advertising, consumer and trade promotions, price, product safety and quality, brand identification, innovation, breadth and depth of product offerings, availability of products, customer service and credit terms."
  * [5] "Walmart Inc. accounted for approximately"
  * [6] "The primary raw materials used in our prepared foods operations are commodity-based raw materials, including beef, pork, chicken, turkey, flour, vegetables, cheese, eggs, seasonings and other cooking ingredients."
  * [7] "buy live cattle on the open spot market"

### EOG: EOG RESOURCES INC (Energy, 10-K filed 2025-02-27)

> Business General the Company., a Delaware corporation organized in 1985, together with its subsidiaries (collectively, the Company), explores for, develops, produces and markets crude oil, natural gas liquids (NGLs) and natural gas primarily in major producing basins in the United States of America ...

**Overall: none.** Returns are high in the recent upcycle but swing with commodity prices; the 2020 loss shows this. Products are sold at unregulated market prices to a few refiner buyers. The only source is a modest cost advantage from acreage quality and infrastructure, without quantified proof. Cyclical high returns do not make a durable moat.

**Returns: above.** ROIC was 21-34% in 2021-2024 (24% in 2024), well above the cost of capital, but 2020 was negative (-2%). Returns follow commodity prices and are cyclical, so they cannot be taken as lasting.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -37% | -5% | -5% | 45% | -2% | -3% | -2% |
| 2021-12-31 | 69% | 33% | 25% | 47% | 21% | 22% | 13% |
| 2022-12-31 | 38% | 39% | 30% | 43% | 34% | 33% | 19% |
| 2023-12-31 | -6% | 40% | 31% | 47% | 30% | 29% | 18% |
| 2024-12-31 | -2% | 34% | 27% | 51% | 24% | 22% | 14% |

* **cost_advantage = 1.** The company states a low-cost goal and has a large, stacked-pay Delaware Basin position and owned infrastructure that lowers costs. However, no quantified cost comparison with peers is given, and the claim is partly aspirational, so the advantage looks modest.
  * [1] "the Company is focused on being among the highest return and lowest cost producers, committed to strong environmental performance and playing a significant role in the long-term future of energy."
  * [2] "The Delaware Basin consists of approximately 4,800 feet of liquids-rich stacked pay potential offering the Company multiple co-development opportunities throughout its 395,000 net acre position."
  * [3] "In addition, key gathering, processing and transportation infrastructure was added in order to lower operating costs and increase price realizations."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [4] "Conversely, sales of crude oil and condensate and NGLs by the Company are made at unregulated market prices."
  * [5] "In each case, the price received was based on market prices at that specific sales point or based on the price index applicable for that location."
  * [6] "Certain of the Company's competitors have financial and other resources substantially greater than those the Company possesses and have established strategic long-term positions or strong governmental relationships in countries or areas in which the Company may seek new or expanded entry."
  * [7] "During 2024, three purchasers each accounted for more than 10% of the Company's total crude oil and condensate, NGLs and natural gas revenues and gathering, processing and marketing revenues."

### BBT: TRUIST FINANCIAL CORP (Financials, 10-K filed 2025-02-25)

> the Company is a purpose-driven financial services company committed to inspiring and building better lives and communities. Headquartered in Charlotte, North Carolina, the Company has leading market share in many of the high-growth markets in the U.S. and offers a wide range of products and ...

**Overall: none.** The text describes an intensely competitive, heavily regulated industry with no direct moat source, and returns on equity are below the cost of capital and volatile. Scale and deposit share in some states are not shown to be a durable advantage.

**Returns: below.** ROE was 6-9% in most years, -2% in 2023 and 7% in 2024. ROA was about 1%. Returns are at or below the cost of equity of roughly 8-10% and are volatile.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | n/a | n/a | n/a | n/a | n/a | 6% | 1% |
| 2021-12-31 | n/a | n/a | n/a | n/a | n/a | 9% | 1% |
| 2022-12-31 | n/a | n/a | n/a | n/a | n/a | 9% | 1% |
| 2023-12-31 | n/a | n/a | n/a | n/a | n/a | -2% | -0% |
| 2024-12-31 | n/a | n/a | n/a | n/a | n/a | 7% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "The financial services industry is intensely competitive and constantly evolving."
  * [4] "Many of our competitors have substantial positions nationally or in the markets in which we operate."
  * [5] "Some also have greater scale, financial and operational resources, investment capacity, product and service offerings, and brand recognition."
  * [6] "Competition affects every aspect of our business, including product and service offerings, rates, pricing and fees, credit limits, and client service."
  * [7] "At the same time, non-banking entities, including financial technology companies, have increased competition by providing financial products and services directly to customers and indirectly through partnerships."

### COF: CAPITAL ONE FINANCIAL CORP (Financials, 10-K filed 2025-02-20)

> OVERVIEW General the Company, a Delaware corporation established in 1994 and headquartered in McLean, Virginia, is a diversified financial services holding company with banking and non-banking subsidiaries. the Company and its subsidiaries (the “Company” or “the Company One”) offer a broad array of ...

**Overall: none.** The text shows scale but no mechanism that keeps competitors out. Competition is described as intense and based on price and rewards, and recent returns are near the cost of capital. No moat.

**Returns: near.** ROE was volatile at 5%, 20%, 13%, 9% and 8%, with ROA of only 1-3%. The 2021 spike came from a reserve release and one-offs. Recent ROE of 8-9% is near or below the cost of equity. ROIC is not reported.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -0% | n/a | 10% | 56% | n/a | 5% | 1% |
| 2021-12-31 | 7% | n/a | 41% | 38% | n/a | 20% | 3% |
| 2022-12-31 | 13% | n/a | 21% | 38% | n/a | 13% | 2% |
| 2023-12-31 | 7% | n/a | 13% | 53% | n/a | 9% | 1% |
| 2024-12-31 | 6% | n/a | 12% | 43% | n/a | 8% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "In general, customers are attracted to credit card issuers largely on the basis of price, credit limit, reward programs, customer experience and other product features."
  * [4] "Each of our business segments operates in a highly competitive environment, and we face competition in all aspects of our business from numerous bank and non-bank providers of financial services."
  * [5] "Our businesses generally compete on the basis of the quality and range of their products and services, transaction execution, innovation and price."
  * [6] "On May 21, 2024, our credit card program agreement with Walmart terminated (“Walmart Program Termination”)."

### EG: EVEREST GROUP, LTD. (Financials, 10-K filed 2025-02-27)

> The Company. the Company is a Bermuda-based reinsurance and insurance organization. As part of the Standard & Poor’s (“S&P”) 500 Index, we are a leading financial services institution focused on diversifying our portfolio and geographic presence. Through our direct and indirect subsidiaries ...

**Overall: none.** The only source of advantage is financial strength ratings and licensing, and these are shared by many competitors. The industry is highly competitive and mature, with price-based competition and alternative capital. Returns are volatile and, averaged over the cycle, near the cost of capital. Together this does not support a moat.

**Returns: near.** ROE ranged from 3% to 23% (3%, 14%, 6%, 23%, 10%) and ROA from 1% to 6%. This is volatile and mostly near or below the cost of equity, with only 2021 and 2023 clearly above it. ROIC is not reported.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 17% | n/a | 3% | 30% | n/a | 3% | 1% |
| 2021-12-31 | 24% | n/a | 12% | 32% | n/a | 14% | 4% |
| 2022-12-31 | 2% | n/a | 5% | 31% | n/a | 6% | 2% |
| 2023-12-31 | 21% | n/a | 17% | 31% | n/a | 23% | 6% |
| 2024-12-31 | 18% | n/a | 8% | 29% | n/a | 10% | 3% |

* **intangibles = 1.** A+ financial strength ratings and broad licensing help win business, but many competitors hold similar ratings and licences, so the advantage is modest and not clearly durable.
  * [1] "Group’s active operating subsidiaries are each rated A+ (“Superior”) by A.M. Best Company (“A.M. Best”)"
  * [2] "Strong financial ratings are particularly important for reinsurance and insurance companies given that customers rely on a company to pay covered losses well into the future."
  * [3] "Such licensing enables U.S. domestic ceding company clients to take credit for uncollateralized reinsurance receivables from the Company Re in their statutory financial statements."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "The global reinsurance market is highly competitive and mature."
  * [5] "Competition for clients might be based on pricing, capacity, coverage terms, conditions or other factors."
  * [6] "The global insurance market is highly competitive."
  * [7] "In addition, we also compete with new companies and existing companies that move into the insurance industry."
  * [8] "The broker with the largest share of the company’s business, Marsh and McLennan, accounted for approximately 21.9% of gross written premiums."
  * [9] "In addition, competitors might include investment companies, mutual companies, insurance companies, alternative risk providers (such as captives, catastrophe bonds and pools) and others"

### MCO: MOODYS CORP /DE/ (Financials, 10-K filed 2025-02-14)

> Background As used in this report, except where the context indicates otherwise, the terms “Moody’s” or the “Company” refer to Moody’s Corporation, a Delaware corporation, and its subsidiaries. The Company’s executive offices are located at 7 World Trade Center at 250 Greenwich Street, New York, NY ...

**Overall: narrow.** Returns are consistently well above the cost of capital, and the text shows an NRSRO-registered ratings franchise that issuers and investors rely on, plus workflow-embedded analytics. These support a moat, but the text gives no market share figures and names competing CRAs and in-house alternatives. No source reaches 2 on the text alone, so the moat is narrow.

**Returns: above.** ROIC was 37%, 32%, 18%, 20% and 27% over five years, always well above a 8-10% cost of capital. Operating margins of 34-46% are high, with a dip in 2022 when revenue fell 12%, followed by a recovery.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 11% | 44% | 33% | 38% | 37% | >100% | 16% |
| 2021-12-31 | 16% | 46% | 36% | 30% | 32% | 95% | 16% |
| 2022-12-31 | -12% | 34% | 25% | 22% | 18% | 49% | 9% |
| 2023-12-31 | 8% | 36% | 27% | 32% | 20% | 54% | 11% |
| 2024-12-31 | 20% | 41% | 29% | 36% | 27% | 60% | 14% |

* **intangibles = 1.** MIS holds an NRSRO registration and a long-established rating franchise whose opinions are used by investors worldwide, which issuers need for market access. Regulation and reputation limit new entrants, and MA owns the world's largest company database. The text gives no market share figures, so the rating is 1.
  * [1] "MIS is registered with the SEC as an NRSRO and is subject to the SEC's oversight and examination authority."
  * [2] "Moody’s opinions on credit are used by institutional investors throughout the world, making an issuer’s debt potentially more attractive to a wide range of buyers."
  * [3] "Among ratings advisors, Moody’s has a strong position and is well-recognized for the depth and breadth of its analytical capabilities."
  * [7] "a data business powered by the world’s largest database on companies and credit (Data & Information)"
* **switching_costs = 1.** MA's tools are embedded in customer workflows, and regulated customers pass compliance requirements to it by contract, which makes replacement harder. The text gives no retention figures or contract lengths, so the advantage looks moderate.
  * [4] "MA's integrated and technology-enabled solutions provide unique capabilities and insights that are embedded in customer workflows."
  * [5] "MA also derives a significant amount of its sales from banks and other financial services providers who are subject to regulatory oversight and who are required to conduct due diligence and pass through certain regulatory requirements to key suppliers such as MA by contract."
  * [6] "However, annual fee arrangements with frequent debt issuers, annual fees from debt monitoring, commercial paper and medium-term note programs, bank deposit ratings, insurance company financial strength ratings, mutual fund ratings, and other areas partially mitigate MIS’s dependence on the volume or number of new debt securities issued in the global fixed-income markets."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "MIS competes with other CRAs and with investment banks and brokerage firms that offer credit opinions and research."
  * [9] "Many users of MIS’s ratings also have in-house credit research capabilities."
  * [10] "MA competes broadly in the financial information and enterprise risk software industries against various diversified competitors."

### NTRS: NORTHERN TRUST CORP (Financials, 10-K filed 2025-02-24)

> the Company the Company (the “Corporation”) is a leading provider of wealth management, asset servicing, asset management and banking solutions to corporations, institutions, families and individuals. The Corporation is a financial holding company conducting business through various U.S. and ...

**Overall: none.** The text shows no mechanism that clearly keeps competitors out. It describes intense competition from many financial and fintech firms and relies on scale and brand claims. ROE of 10-16% is only near the cost of equity for a bank, and heavy regulation adds cost. High asset servicing scale may carry some advantage, but the text does not demonstrate it, so no moat is supported.

**Returns: near.** ROIC is not reported. ROE ranged from 10% to 16% and ROA was about 1%. That is around or modestly above a cost of equity of 9-10%, and it is only moderately steady. Net margins were 16-24%. 2024 revenue growth of 22% looks unusual and may reflect one-off items.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 0% | n/a | 20% | 29% | n/a | 11% | 1% |
| 2021-12-31 | 6% | n/a | 24% | 19% | n/a | 13% | 1% |
| 2022-12-31 | 5% | n/a | 20% | 33% | n/a | 11% | 1% |
| 2023-12-31 | 0% | n/a | 16% | 37% | n/a | 10% | 1% |
| 2024-12-31 | 22% | n/a | 24% | -7% | n/a | 16% | 1% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "the Company faces intense competition in all aspects and areas of its business."
  * [2] "Our competitors include a broad range of financial institutions and service companies, including other custodial banks, investment counseling firms, deposit-taking institutions, asset management firms, benefits consultants, trust companies, investment banking firms, insurance companies, and various financial technology companies, including software providers and data services firms."

### PYPL: PayPal Holdings, Inc. (Financials, 10-K filed 2025-02-04)

> OVERVIEW At the Company, our mission is to revolutionize commerce globally. Our products are designed to enable digital payments and simplify commerce experiences for consumers and merchants to make selling, shopping, and sending and receiving money simple, personalized, and secure, online or ...

**Overall: narrow.** Returns are consistently above the cost of capital, supported by a real two-sided network, trusted brands and global licenses. These advantages are tempered by intense competition, price sensitivity and slowing growth, so the moat is narrow rather than wide.

**Returns: above.** ROIC was 30% in 2020-21, fell to 17% in 2022 and has been 18-19% since, well above a 8-10% cost of capital. Operating margins are steady at 14-17%, and ROE is 12-23%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 21% | 15% | 20% | 25% | 30% | 23% | 7% |
| 2021-12-31 | 18% | 17% | 16% | 19% | 30% | 20% | 6% |
| 2022-12-31 | 8% | 14% | 9% | 19% | 17% | 12% | 3% |
| 2023-12-31 | 8% | 17% | 14% | 14% | 18% | 21% | 5% |
| 2024-12-31 | 7% | 17% | 13% | 21% | 19% | 20% | 5% |

* **intangibles = 1.** Well-known brands (the Company, Venmo) and money transmitter licenses across many jurisdictions are cited, with Venmo a leading U.S. P2P app. The licenses are a modest barrier and the brand claims are fairly generic.
  * [4] "we believe that our regulatory licenses, which enable us to operate in markets around the world, are a distinct advantage and help support business growth."
  * [5] "Our Venmo digital wallet in the United States (“U.S.”) is a leading mobile application used to move money between our customers."
* **network_effects = 1.** A two-sided network of 434M accounts connecting consumers and merchants, with P2P that acquires new users, is a direct network effect. Competition from cards, Apple Pay and others, and slowing account growth (2%), make it durable but not dominant.
  * [1] "We operate a global, two-sided network at scale that connects consumers and merchants with 434 million active accounts across approximately 200 markets as of December 31, 2024."
  * [2] "Our relationship on both sides of a transaction enables us to offer unique product experiences designed to remove friction, drive sales, and enhance shopping experiences."
  * [3] "P2P is an important source of customer engagement and also serves as a customer acquisition channel that facilitates organic growth by enabling potential users to establish active accounts with the Company or Venmo at the time they make or receive a P2P payment."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The global payments industry is highly competitive, dynamic, and innovative, and subject to regulatory scrutiny and oversight."
  * [8] "Many of the areas in which we compete evolve rapidly with innovative and disruptive technologies, shifting user preferences and needs, price sensitivity of consumers and merchants, and frequent introductions of new products and services."
  * [9] "Competition also may intensify as new competitors emerge, businesses combine or enter into new partnerships, and established companies in other segments expand to become competitive with various aspects of our business."

### SCHW: SCHWAB CHARLES CORP (Financials, 10-K filed 2025-02-26)

> General Corporate Overview The Charles the Company (CSC) is a savings and loan holding company. CSC engages, through its subsidiaries (collectively referred to as the Company or the Company), in wealth management, securities brokerage, banking, asset management, custody, and financial advisory ...

**Overall: narrow.** The Company has scale-based cost advantages and some stickiness of client assets, which together support a modest moat. Returns are decent but not clearly far above the cost of capital, and competition is broad. Narrow rather than wide.

**Returns: near.** ROIC is not reported. ROE ran 8-15% (13% in 2023 and 2024), roughly at or modestly above the cost of equity, with ROA about 1%. Net margins of 27-35% are healthy and fairly steady, but ROE was depressed by acquisition effects and large equity.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 9% | n/a | 28% | 53% | n/a | 8% | 1% |
| 2021-12-31 | 58% | n/a | 32% | 6% | n/a | 10% | 1% |
| 2022-12-31 | 12% | n/a | 35% | 5% | n/a | 15% | 1% |
| 2023-12-31 | -9% | n/a | 27% | >100% | n/a | 13% | 1% |
| 2024-12-31 | 4% | n/a | 30% | 10% | n/a | 13% | 1% |

* **intangibles = 1.** Brand and trust are claimed as an advantage, but the claim is generic and unquantified. Regulation is a burden shared by all competitors, not a barrier unique to the Company.
  * [3] "In an industry dependent on trust, the Company reputation and brand across multiple constituents enable us to attract clients and employees while credibly introducing new products to the market."
* **switching_costs = 1.** Custody of $10 trillion in assets and RIA workflow platforms imply some stickiness. The text does not directly state switching costs, only that relationships deepen.
  * [4] "Providing bank, wealth, and asset management services to broker-dealer clients helps serve a wider array of needs, thereby deepening relationships, enhancing the stability of client assets, and enabling diversified revenue streams."
  * [5] "The Advisor Services website is the core platform for RIAs to conduct daily business activities online with the Company, including viewing and managing client account information and accessing news and market information."
* **cost_advantage = 1.** Scale and shared infrastructure are stated as a cost advantage that allows low pricing across a large client base. The Company is one of the largest US firms, and the claim is direct but not quantified.
  * [1] "As one of the largest investment services firms in the U.S., we are able to spread operating costs and amortize new investments over a large base of clients, and harness the resources to evolve capabilities to meet client needs."
  * [2] "Coupled with scale, our operating efficiency and sharing of infrastructure across different businesses creates a cost advantage that enables us to competitively price products and services while profitably serving clients of various sizes across multiple channels."
* Rated 0: network_effects, efficient_scale.
* Against a moat:
  * [7] "Within Investor Services, our competition in serving individual investors spans brokerage, wealth management, and asset management firms, as well as banks, trust companies, financial technology companies, and retirement service providers."
  * [8] "In the Advisor Services arena, we compete with institutional custodians, wirehouses, regional and independent broker-dealers, banks, and trust companies."

### ABBV: AbbVie Inc. (Health Care, 10-K filed 2025-02-14)

> Overview the Company or "the company" refer to the Company, or the Company and its consolidated subsidiaries, as the context requires. the Company is a global, diversified research-based biopharmaceutical company positioned for success with a comprehensive product portfolio that has leadership ...

**Overall: narrow.** Returns have been above the cost of capital, backed by patent and regulatory exclusivity on a diversified portfolio (Skyrizi, Rinvoq, Botox, aesthetics). The advantage is patent-based and finite: Humira has lost exclusivity, Imbruvica and others face government price-setting, and the Skyrizi and Rinvoq patents expire in 2033. Falling ROIC supports a narrow rather than wide moat.

**Returns: above.** ROIC was 16%, 16%, 19%, 16% and 12% over 2020-2024, above an 8-10% cost of capital but declining as Humira biosimilars arrive. Margins fell in 2023-24, partly because of acquired intangible amortization and in-process R&D charges. ROE is distorted by leverage and buybacks.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 38% | 25% | 10% | 37% | 16% | n/a | 4% |
| 2021-12-31 | 23% | 32% | 21% | 39% | 16% | 81% | 8% |
| 2022-12-31 | 3% | 31% | 20% | 42% | 19% | 72% | 8% |
| 2023-12-31 | -6% | 23% | 9% | 41% | 16% | 35% | 4% |
| 2024-12-31 | 4% | 16% | 8% | 32% | 12% | 63% | 3% |

* **intangibles = 1.** Patent and regulatory exclusivity on key products (Skyrizi and Rinvoq patents to 2033, patents extending into the mid 2040s, 12-year biologic exclusivity) plus market-leading aesthetics positions give a real but finite advantage. Humira already faces biosimilars and the IRA is eroding pricing, so it is not durable beyond patent lives.
  * [1] "The United States composition of matter patents covering risankizumab and upadacitinib are expected to expire in 2033."
  * [2] "These patents and applications, including various patents that expire during the period 2025 to the mid 2040s, in aggregate are believed to be of material importance in the operation of the Company business."
  * [3] "The law provides a pathway for approval of biosimilars following the expiration of 12 years of regulatory exclusivity for the innovator biologic and a potential additional 180 day-extension term for conducting pediatric studies."
  * [4] "which hold market-leading positions in the U.S. and in key markets around the world."
  * [5] "The more complex nature of biologics and biosimilar products has led to close regulatory scrutiny over follow-on biosimilar products, which can reduce the effect of biosimilars on sales of the innovator biologic as compared to the sales erosion caused by generic versions of small molecule pharmaceutical products."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Humira faces direct biosimilar competition globally, and the Company will continue to face competitive pressure from these biologics and from orally administered products."
  * [8] "In addition, the substitution of generic and biosimilar pharmaceutical products for branded pharmaceutical products creates competitive pressures on the Company products that do not have patent protection."
  * [9] "In 2024, three wholesale distributors (McKesson Corporation, Cardinal Health, Inc. and Cencora, Inc.) accounted for substantially all of the Company pharmaceutical product sales in the United States."
  * [10] "In August 2023, the U.S. Department of Health and Human Services (HHS), through the Centers for Medicare & Medicaid Services (the CMS), selected Imbruvica as one of the first 10 medicines subject to government-set prices beginning in 2026."
  * [11] "In addition, in the past few years, a number of other companies have started to develop, have successfully developed and/or are currently marketing products that are being positioned as competitors to Botox."

### DGX: QUEST DIAGNOSTICS INC (Health Care, 10-K filed 2025-02-20)

> ." Critical Accounting Policies The preparation of financial statements in conformity with accounting principles generally accepted in the United States requires us to make estimates and assumptions and select accounting policies that affect our reported financial results and the disclosure of ...

Not tagged: the located Item 1 'Critical Accounting Policies' in the first 10,000 characters.

### GILD: GILEAD SCIENCES, INC. (Health Care, 10-K filed 2025-02-28)

> the Company (including its consolidated subsidiaries, referred to as “the Company,” the “company,” “we,” “our” or “us”) is a biopharmaceutical company that has pursued and achieved breakthroughs in medicine for more than three decades, with the goal of creating a healthier world for all people. We ...

**Overall: narrow.** The moat rests on patent-protected drugs (Biktarvy to 2033) in HIV, hepatitis and oncology, which supports high returns in most years. Patents expire, payers press on prices, wholesalers are concentrated, and competition is intense, so the advantage lasts about a decade. Returns are above the cost of capital in most years, so narrow.

**Returns: above.** ROIC was 14-18% in 2021-2023, above the cost of capital, but fell to 8% in 2020 and 3% in 2024. Those two years were likely distorted by large acquired in-process R&D charges, so the picture is mixed and uneven.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 10% | 16% | 0% | 30% | 8% | 1% | 0% |
| 2021-12-31 | 11% | 36% | 23% | 40% | 18% | 32% | 9% |
| 2022-12-31 | -0% | 27% | 17% | 31% | 14% | 22% | 7% |
| 2023-12-31 | -1% | 28% | 21% | 27% | 15% | 26% | 9% |
| 2024-12-31 | 6% | 6% | 2% | 36% | 3% | 2% | 1% |

* **intangibles = 1.** Patent-protected key drugs (Biktarvy through 2033, Epclusa, Sunlenca to 2037) and regulatory exclusivities give solid but finite protection. Several products face expiries in 2027-2033 and generic settlements, so the advantage is real but time-limited.
  * [1] "Patents and other proprietary rights are very important to our business."
  * [2] "Biktarvy | 2033 | 2033"
  * [4] "We acquired exclusive rights to these patents in the agreements we have with these parties."
  * [5] "As a result of these regulations, product development and product approval processes are very expensive and time consuming, which has a significant impact on our capital expenditures and results of operations."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "We operate in a highly competitive environment."
  * [8] "During the year ended December 31, 2024, approximately 91% of our product sales in the U.S. and approximately 65% of our total worldwide revenues were from three large wholesalers: Cardinal Health, Inc., Cencora, Inc. and McKesson Corporation, and their specialty distributor affiliates."
  * [9] "pricing pressures from private insurers and government payers as our products mature, which often result in a reduction of the net product prices"
  * [10] "new branded or generic products introduced into major markets, which may impact our ability to maintain pricing and market share."
  * [11] "Complera/Eviplera | 2025 | 2026"

### HUM: HUMANA INC (Health Care, 10-K filed 2025-02-20)

> General Headquartered in Louisville, Kentucky, the Company. and its subsidiaries, referred to throughout this document as “we,” “us,” “our,” the “Company” or “the Company,” is committed to putting health first – for our teammates, our customers, and our company. Through our the Company insurance ...

**Overall: none.** The company is a large Medicare Advantage insurer with scale-based provider discounts, but it operates in a highly competitive industry where rivals are often larger. Customers re-enroll annually and revenue depends 85% on government payment rates. Returns have fallen to about the cost of capital, so no durable moat is supported.

**Returns: near.** ROIC fell from 27% in 2020 to 12-14% in 2021-2023 and 8% in 2024. Operating margin dropped from 6% to 2%, and ROE fell to 7%. Returns are declining toward the cost of capital and are not stable.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 19% | 6% | 4% | 6% | 27% | 26% | 11% |
| 2021-12-31 | 8% | 4% | 4% | 1% | 12% | 20% | 7% |
| 2022-12-31 | 12% | 4% | 3% | 4% | 13% | 18% | 6% |
| 2023-12-31 | 15% | 4% | 2% | 3% | 14% | 16% | 6% |
| 2024-12-31 | 11% | 2% | 1% | 2% | 8% | 7% | 3% |

* **cost_advantage = 1.** Scale gives some provider discounts, but competitors with larger membership bases are said to exist, so the advantage is modest and not unique.
  * [2] "Our membership base and the ability to influence where our members seek care generally enable us to obtain contractual discounts with providers."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [5] "The health benefits industry is highly competitive."
  * [6] "Many of our competitors have a larger membership base and/or greater financial resources than our health plans in the markets in which we compete."
  * [7] "During 2024, 85% of our total premiums and services revenue were derived from contracts with the federal government, including 14% derived from our individual Medicare Advantage contracts in Florida"
  * [8] "Our individual Medicare Advantage products covered under Medicare Advantage contracts with CMS are renewed generally for a calendar year term unless CMS notifies us of its decision not to renew by May 1 of the calendar year in which the contract would end"

### IDXX: IDEXX LABORATORIES INC /DE (Health Care, 10-K filed 2025-02-21)

> COMPANY OVERVIEW the Company was incorporated in Delaware in 1983. We develop, manufacture, and distribute products and provide services primarily for the companion animal veterinary, livestock and poultry, dairy and water testing industries. We also provide human medical point-of-care and ...

**Overall: narrow.** Consistently very high returns are backed by a recurring-consumables installed base integrated with software, and by regulatory approvals and a long exclusive slide supply deal. The company admits that no patent matters much and that large competitors, Mars and Zoetis, offer similar products. The advantages are real but modest, so a narrow moat fits better than a wide one.

**Returns: above.** ROIC has stayed between 49% and 58% over five years, far above the cost of capital. Operating margins of 26-30% have been steady and ROE and ROA are high. ROIC fell to 49% in 2024.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 12% | 26% | 21% | 20% | 56% | >100% | 28% |
| 2021-12-31 | 19% | 29% | 23% | 20% | 58% | >100% | 31% |
| 2022-12-31 | 5% | 27% | 20% | 12% | 53% | >100% | 26% |
| 2023-12-31 | 9% | 30% | 23% | 21% | 58% | 81% | 28% |
| 2024-12-31 | 6% | 29% | 23% | 21% | 49% | 58% | 27% |

* **intangibles = 1.** Exclusive Catalyst slide supply arrangement through 2044 and EPA/FDA/USDA approvals for water, dairy and livestock tests give some protection, but the company says no single patent is material and competitors have entered despite its patents.
  * [5] "The agreements also prohibit Ortho from promoting and selling these chemistry slides in the veterinary sector, excluding the EU, other than to the Company."
  * [6] "Our water tests are used by government laboratories, water utilities, and private certified laboratories to test drinking water in compliance with regulatory standards, including U.S. Environmental Protection Agency (“EPA”) standards."
  * [7] "Before a test can be used as part of a water quality monitoring program in the U.S. that is regulated by the EPA, the test method must first be approved by the EPA."
  * [8] "All test methods used for drug residue testing require validation by the FDA and acceptance by the NCIMS."
* **switching_costs = 1.** A large installed base of analyzers drives recurring consumables revenue, and instruments are integrated with IVLS, VetConnect PLUS and practice management software, which makes switching disruptive for clinics. The text gives no retention data or contract lengths.
  * [1] "Sales of consumables to customers who use our chemistry analyzers provide the majority of our instrument consumables revenues from our installed base of the Company VetLab instruments."
  * [2] "Our large practice management systems installed base provides access to veterinary channel transaction activity, enabling a syndicated data offering."
  * [3] "We sell IVLS as an integral component for our in-clinic analyzer suite."
  * [4] "Regardless of modality utilized, veterinarians are provided with clinically relevant data which is integrated within our information management technologies."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [12] "Several of our direct and potential competitors have substantially greater financial and managerial resources than us, as well as greater experience in manufacturing, marketing, research and development, and obtaining regulatory approvals than we do."
  * [13] "In addition, we face robust competition as other companies have been successful in bringing competitive products and services to market, despite the protections afforded by these patent and license rights."
  * [14] "Our major competitors in most geographic locations in North America are Mars, Incorporated brands Antech Diagnostics and Heska; and Zoetis Inc. (including its wholly-owned subsidiary Abaxis, Inc.)."
  * [15] "we do not believe that any single patent, copyright, trademark, trade secret, or license is material to our business as a whole, except for licenses related to our Catalyst instruments and consumables"

### MOH: MOLINA HEALTHCARE, INC. (Health Care, 10-K filed 2025-02-11)

> OVERVIEW ABOUT the Company the Company, a FORTUNE 500 company, provides managed healthcare services under the Medicaid and Medicare programs, and through the state insurance marketplaces (the “Marketplace”). the Company was founded in 1980 as a provider organization serving low-income families in ...

**Overall: narrow.** Returns are consistently good, and licensing and procurement barriers give a modest efficient-scale edge. But contracts are short, competitively re-bid and terminable at will, competition is fierce from larger national plans, and margins are thin and exposed to government rate-setting. There is no strong source of advantage, so the moat is narrow.

**Returns: above.** ROIC is very high (63% to over 100%), but it is computed on thin equity and debt net of cash, and insurers are better judged on ROE and ROA. ROE of 27-33% and ROA of 6-8% are solid and fairly steady. Operating margins are thin at 4-6%, and FCF margin fell to 1% in 2024.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 15% | 6% | 3% | 9% | n/a | 33% | 8% |
| 2021-12-31 | 43% | 4% | 2% | 7% | n/a | 28% | 6% |
| 2022-12-31 | 15% | 4% | 2% | 2% | >100% | 28% | 6% |
| 2023-12-31 | 7% | 5% | 3% | 5% | 93% | 30% | 8% |
| 2024-12-31 | 19% | 4% | 3% | 1% | 63% | 27% | 8% |

* **efficient_scale = 1.** State licensing and bidding procedures create only partial barriers to entry. Several states award multiple contracts and large national plans are entering, so the market does not support only one or a few firms.
  * [1] "However, the licensing requirements and bidding and contracting procedures in some states may present partial barriers to entry into our industry."
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [3] "Such contracts are subject to risk of loss in states that issue RFPs open to competitive bidding by other health plans."
  * [4] "Our state Medicaid contracts typically have terms of three to five years, contain renewal options exercisable by the state Medicaid agency, and allow either the state or the health plan to terminate the contract with or without cause."
  * [5] "There is increasing competition driven by renewed interest from large national health plans."
  * [6] "The Medicare market is highly competitive across the country, with large competitors, such as CVS Health Corporation, Humana Inc., and UnitedHealth Group Inc."
  * [7] "Competition can vary considerably from state to state, and some of our competitors have larger membership bases and/or greater financial resources than our health plans in the markets in which we compete."
  * [9] "The competition is fierce, as our markets are attractive."

### REGN: REGENERON PHARMACEUTICALS, INC. (Health Care, 10-K filed 2025-02-05)

> This Annual Report on Form 10-K contains forward-looking statements that involve risks and uncertainties relating to future events and the future performance of the Company (where applicable, together with its subsidiaries, "the Company," "Company," "we," "us," and "our"), and actual events or ...

**Overall: narrow.** Returns remain above the cost of capital, backed by patent and regulatory exclusivity on Dupixent and other products plus a differentiated antibody discovery platform. Against that, EYLEA faces biosimilar entry, patents are finite, a few wholesalers dominate sales, and competition is intense. The moat is narrow, resting on exclusivity of individual drugs and pipeline productivity rather than durable structural advantages.

**Returns: above.** ROIC was 28%, 49%, 19%, 14% and 12% over 2020-2024, above the 8-10% cost of capital but falling steadily as EYLEA faced biosimilars and margins compressed. Operating margin fell from 56% to 28%. Returns are still above the cost of capital but the trend is down.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 30% | 42% | 41% | 24% | 28% | 32% | 22% |
| 2021-12-31 | 89% | 56% | 50% | 41% | 49% | 54% | 38% |
| 2022-12-31 | -24% | 39% | 36% | 36% | 19% | 21% | 16% |
| 2023-12-31 | 8% | 31% | 30% | 30% | 14% | 16% | 13% |
| 2024-12-31 | 8% | 28% | 31% | 26% | 12% | 16% | 12% |

* **intangibles = 1.** Product-specific patents and regulatory approvals protect key drugs (Dupixent patents run to the 2030s), and the VelocImmune platform and genetics database add a modest edge. But EYLEA faces biosimilars and key platform patents expire by 2032. Drug-specific exclusivity is finite and uncertain, so the advantage is real but not wide.
  * [1] "Our remaining issued patents covering these technologies will expire between 2025 and 2032."
  * [2] "Each of these facilities is approved by the FDA and certain other regulatory agencies to manufacture our bulk drug materials and products."
  * [4] "In the European Economic Area ("EEA") (which is comprised of 27 Member States of the EU plus Norway, Iceland, and Liechtenstein), medicinal products can only be commercialized after a related Marketing Authorization has been granted."
  * [5] "RGC undertakes genetic sequencing of these samples to create a unique resource of de-identified genetic data and associated phenotype data for research."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "We face substantial competition from pharmaceutical and biotechnology companies."
  * [7] "Others may independently develop similar products or processes to those developed by us, duplicate any of our products or processes or, if patents are issued to us, design around any products and processes covered by our patents."
  * [8] "We had sales to two customers (Besse Medical, a subsidiary of Cencora, Inc., and McKesson Corporation) that each accounted for more than 10% of total gross product revenue for the year ended December 31, 2024."
  * [9] "Competition among products approved for sale is based on efficacy, safety, reliability, ease of administration, dosing frequency, availability, price, patent and other intellectual property position, and other factors."
  * [10] "There is no certainty that our existing patents or others, if obtained, will provide us protection from competition or provide commercial benefit."

### ALLE: Allegion plc (Industrials, 10-K filed 2025-02-18)

> BUSINESS Overview the Company ("the Company," "we," "us" or "the Company") is a leading global provider of security products and solutions that keep people and assets safe and secure in the places they live, learn, work and connect. We pioneer safety and security to create a safer and more ...

**Overall: narrow.** Returns are consistently well above the cost of capital, supported by century-old brands and specification-driven lock-in that make the business sticky. The market is fragmented and competitive, with rivals such as Assa Abloy and dormakaba, and no source is rated 2, so the moat is narrow rather than wide.

**Returns: above.** ROIC has stayed at 18-23% over five years, well above the 8-10% cost of capital. Operating margin has risen from 15% to 21%, and FCF margins are steady at 12-16%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -5% | 15% | 12% | 16% | 18% | 39% | 10% |
| 2021-12-31 | 5% | 18% | 17% | 15% | 23% | 61% | 16% |
| 2022-12-31 | 14% | 18% | 14% | 12% | 20% | 54% | 13% |
| 2023-12-31 | 12% | 19% | 15% | 14% | 20% | 48% | 13% |
| 2024-12-31 | 3% | 21% | 16% | 15% | 21% | 42% | 14% |

* **intangibles = 1.** Century-old brands (Schlage, Von Duprin, LCN) with specification relationships are described as the most valuable assets, but there is no pricing-power evidence, and the text states no dependence on particular patents. The advantage looks modest but real.
  * [1] "We consider our CISA, Interflex, LCN, Schlage, SimonsVoss, Von Duprin and other associated trademarks to be among our most valuable assets, and we have registered these trademarks in a number of countries."
  * [2] "Several of our brands were established more than 100 years ago, and many originally created their categories:"
* **switching_costs = 1.** Products are specified into door openings by architects and consultants and must fit exactly with existing hardware, and there is a recurring service base. This implies moderate lock-in, though it is not stated as switching costs directly.
  * [3] "Each door must fit exactly within its frame, be prepared precisely for its hinges, synchronize with its specific lockset and corresponding latch and align with a specific key to secure the door."
  * [4] "Our field sales professionals are assisted by specification writers who work with architects, engineers and consultants to help design door openings and security systems to meet end-users’ functional, aesthetic and regulatory requirements."
  * [5] "Our Stanley Access Technologies business offers extensive planned inspection, maintenance and repair services for its automatic entrance solutions throughout the U.S. and Canada."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The security products markets are highly competitive and fragmented throughout the world, with a number of large multi-national companies and thousands of smaller regional and local companies."
  * [7] "We compete based on the breadth, innovation and quality of our products and solutions, pricing, our ability to custom-configure solutions to meet individual end-user requirements and our global supply chain."
  * [8] "Although certain proprietary intellectual property rights are important to our success, we do not believe we are materially dependent on any particular patent or license, or any particular group of patents or licenses."
  * [9] "Our 10 largest customers represented approximately 27% of our total Net revenues in 2024."

### AME: AMETEK INC/ (Industrials, 10-K filed 2025-02-20)

> General Development of Business the Company (“the Company” or the “Company”) is incorporated in Delaware. Its predecessor was originally incorporated in Delaware in 1930 under the name American Machine and Metals, Inc. the Company is a leading global manufacturer of electronic instruments and ...

**Overall: narrow.** Consistent ROIC of 12-13% and operating margins of 24-26% show that its niche positions, differentiated products and aerospace relationships earn solid returns. The text says the markets are highly competitive and that no patent or trademark is essential, so the individual sources are modest. That supports a narrow moat, not a wide one.

**Returns: above.** ROIC of 11-13% is consistently above a 8-10% cost of capital, but only modestly. Operating margins are high and rising (23% to 26%), and ROE is steady at 15-16%. The returns are good but not exceptional.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -12% | 23% | 19% | 27% | 11% | 16% | 9% |
| 2021-12-31 | 22% | 24% | 18% | 19% | 13% | 15% | 9% |
| 2022-12-31 | 11% | 24% | 19% | 16% | 13% | 16% | 10% |
| 2023-12-31 | 7% | 26% | 20% | 24% | 13% | 16% | 10% |
| 2024-12-31 | 5% | 26% | 20% | 25% | 12% | 15% | 9% |

* **intangibles = 1.** Niche market positions and a long aerospace reputation are cited, but without figures, and the company says no patent or trademark is essential.
  * [1] "the Company maintains significant market share in a number of targeted niche markets through its ability to produce and deliver high-quality, differentiated products at competitive prices."
  * [2] "Among its more significant competitive advantages is its 70-year-plus reputation as an established aerospace supplier."
* **switching_costs = 1.** Aerospace products are designed to customer specifications and sold through long-standing OEM relationships, which suggests some qualification and integration stickiness. The text gives no contract terms or quantified evidence.
  * [4] "the Company aerospace products are designed to customer specifications and manufactured to stringent operational and reliability requirements."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "In general, the Company markets are highly competitive with competition based on technology, performance, quality, service and price."
  * [7] "In process and analytical instruments, numerous companies compete in each market on the basis of product quality, performance and innovation."
  * [8] "the Company, however, does not consider any single patent or trademark, or any group of them, essential either to its business as a whole or to either one of its reportable segments."
  * [9] "EMG’s businesses compete with a number of companies in each of its markets."
* Fixes: switching_costs: dropped citations [3].

### CHRW: C. H. ROBINSON WORLDWIDE, INC. (Industrials, 10-K filed 2025-02-14)

> Overview the Company (“the Company,” “the company,” “we,” “us,” or “our”) is one of the largest global logistics providers in the world, with consolidated total revenues of $17.7 billion in 2024. We deliver logistics like no one else. Companies around the world look to us to reimagine supply ...

**Overall: narrow.** Returns are consistently above the cost of capital because the business is asset-light, and the text shows modest scale and network advantages (the largest platform in a fragmented market). Against this, short contracts, no volume commitments, price competition and low barriers point to a narrow rather than wide moat.

**Returns: above.** ROIC was 21-29% in 2020-2022, dropped to 14% in 2023 and recovered to 18% in 2024, so it stayed above a 8-10% cost of capital throughout. Operating margins are thin (3-5%) and cyclical, but asset-light capital use keeps returns high.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 6% | 4% | 3% | 3% | 21% | 29% | 10% |
| 2021-12-31 | 43% | 5% | 4% | 0% | 27% | 43% | 14% |
| 2022-12-31 | 7% | 5% | 4% | 6% | 29% | 56% | 14% |
| 2023-12-31 | -29% | 3% | 2% | 4% | 14% | 23% | 6% |
| 2024-12-31 | 1% | 4% | 3% | 3% | 18% | 30% | 9% |

* **network_effects = 1.** A platform connecting 83,000 shippers and 450,000 carriers gives some two-sided liquidity benefit, but the market is fragmented and carriers and shippers multi-home, so the effect is modest.
  * [1] "our proprietary technology connects 83,000 customers and 450,000 carriers"
* **cost_advantage = 1.** Scale in purchasing and data is claimed as an advantage, and high returns suggest some benefit, but there is no quantified cost gap versus competitors in a fragmented industry.
  * [2] "Our customers leverage our significant capacity, broad procurement options, global data insights, and substantial shipment volumes for better efficiency, service, and marketplace advantages;"
* Rated 0: intangibles, switching_costs, efficient_scale.
* Against a moat:
  * [5] "The transportation services industry is highly competitive and fragmented."
  * [6] "Most of our contractual rate commitments are for one year or less and allow for renegotiation."
  * [7] "As is typical in the transportation industry, most of these contracts do not include specific volume commitments."
  * [8] "We often compete with respect to price, scope of services, or a combination thereof"
  * [9] "Customers communicate their freight needs, typically on an order-by-order basis"

### DE: DEERE & CO (Industrials, 10-K filed 2025-12-18)

> BUSINESS. This Annual Report on Form 10-K contains forward-looking statements that are subject to risks and uncertainties. All statements other than statements of historical fact included in this Annual Report on Form 10-K are forward-looking statements. Forward-looking statements provide our ...

**Overall: narrow.** Returns are consistently well above the cost of capital, and the text supports modest advantages from brand, dealer network, proprietary parts and an installed base that is increasingly tied to its technology. Competition is intense and growing, with many global rivals and new tech entrants, and no source is rated 2. That supports a narrow moat, not a wide one.

**Returns: above.** ROIC ran 13% to 34% over 2021-2024, well above a cost of capital of 8-10%, and ROE was 21-48%. Margins stayed high (operating margin 17-21%) even as revenue fell 16% and 12% in the last two years. ROIC for 2025 is not reported, but ROE of 21% is still strong. The financing arm distorts the figures somewhat.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-10-31 | 24% | 18% | 14% | 16% | 13% | 38% | 7% |
| 2022-10-30 | 19% | 18% | 14% | 7% | 18% | 37% | 8% |
| 2023-10-29 | 16% | 21% | 17% | 12% | 34% | 48% | 10% |
| 2024-10-27 | -16% | 17% | 14% | 15% | 23% | 32% | 7% |
| 2025-11-02 | -12% | n/a | 11% | 13% | n/a | 21% | 5% |

* **intangibles = 1.** Brand recognition is cited as a competitive factor, and the dealer network of about 2,050 North American locations supports distribution. Both are stated only in general terms, with no pricing power or share figures, so the advantage looks modest.
  * [1] "John the Company brand recognition is a competitive factor in North America and many other parts of the world."
  * [4] "Through the U.S. and Canada, we market products to approximately 2,050 independent dealer locations."
* **switching_costs = 1.** Proprietary parts for the installed base and the connected Operations Center give some lock-in with owners. The text does not quantify it, so it looks modest.
  * [2] "We supply parts, many of which are proprietary, to support the current product lines as well as products we have sold in the past."
  * [3] "This connection provides real-time alerts and information about equipment location, utilization, performance, and maintenance to improve productivity and efficiency, as well as to monitor agronomic job execution."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The agricultural equipment industry continues to change and is becoming even more competitive through the emergence and global expansion of many competitors."
  * [7] "As technology increasingly enables enhanced productivity in agriculture, the industry is also attracting non-traditional competitors, including technology-focused companies and start-up ventures."
  * [8] "The principal competitive factors in all markets include product performance, technology features, innovation, quality, distribution, sustainability, financing, customer service, and value."
  * [9] "The equipment operations are the financial services operations’ major source of business, although many retail purchasers of our products finance their purchases outside our organization through a variety of sources, including commercial banks and finance and leasing companies."

### LII: LENNOX INTERNATIONAL INC (Industrials, 10-K filed 2025-02-11)

> Business References in this Annual Report on Form 10-K to “we,” “our,” “us,” “the Company,” “the Company” or the “Company” refer to the Company. and its subsidiaries, unless the context requires otherwise. The Company We are a global leader in energy-efficient climate-control solutions. We design, ...

**Overall: none.** Returns are exceptionally high, but the text gives no mechanism that stops competitors from taking the business. It describes a competitive oligopoly in which price, availability and performance matter, and patents are not material. The company-owned distribution network (Company Stores and direct-to-dealer sales) is a plausible but unquantified advantage. High returns without a documented source are not a moat by themselves, so the call is none.

**Returns: above.** ROIC is very high and steady at 42-55% over five years, well above the cost of capital, with operating margins rising from 13% to 19%. ROE above 100% reflects leverage and buybacks.

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

### PWR: QUANTA SERVICES, INC. (Industrials, 10-K filed 2025-02-20)

> Business OVERVIEW the Company (together with its subsidiaries, “the Company,” “we,” “us” or “our”) is a leading provider of comprehensive infrastructure solutions for the electric and gas utility, renewable energy, technology, communications, pipeline and energy industries in the United States, ...

**Overall: none.** The business is a labor-intensive contractor with price-based bidding, low barriers to entry in some markets, and customers that can use in-house crews. Preferred-vendor status gives modest stickiness, but returns are near the cost of capital and recent improvement is not enough to show a durable moat.

**Returns: near.** ROIC was 9%, 7%, 8%, 10% and 11% over 2020-2024, around the cost of capital and only recently at the top of the range. Operating margins are steady at 5-6%, which is thin.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -8% | 5% | 4% | 8% | 9% | 11% | 5% |
| 2021-12-31 | 16% | 5% | 4% | 2% | 7% | 10% | 5% |
| 2022-12-31 | 32% | 5% | 3% | 4% | 8% | 9% | 4% |
| 2023-12-31 | 22% | 5% | 4% | 5% | 10% | 13% | 5% |
| 2024-12-31 | 13% | 6% | 4% | 6% | 11% | 13% | 5% |

* **switching_costs = 1.** Preferred-provider status and alliance agreements with preferential bidding give some stickiness, but contracts are mostly MSAs and bid-based work, so customers can still switch fairly easily.
  * [1] "We are a preferred service provider for many of our customers, which generally means we have met minimum standards for a specific category of service, maintained a high level of performance and agreed to certain payment terms and negotiated rates."
  * [2] "Strategic alliance agreements also generally state an intention to work together over a period of time and/or on specific types of projects, and many provide us with preferential bidding procedures."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "In addition, there are relatively few barriers to entry into some of the industries in which we operate and, as a result, organizations that have adequate financial resources and access to technical expertise may become a competitor."
  * [5] "A significant portion of our revenues is currently derived from unit price or fixed price agreements, and price is often an important factor in the award of such agreements."
  * [6] "The industries and geographic markets in which we operate are highly competitive, and several of our competitors are large companies that have significant financial, technical and marketing resources."
  * [7] "The in-house service organizations of our existing or prospective customers employ personnel who perform some of the same types of services we provide."

### RSG: REPUBLIC SERVICES, INC. (Industrials, 10-K filed 2025-02-14)

> BUSINESS Overview Republic is one of the largest providers of environmental services in the United States, as measured by revenue. We operate across the United States and Canada through 367 collection operations, 248 transfer stations, 75 recycling centers, 208 active landfills, 2 treatment, ...

**Overall: narrow.** The company has local density, vertical integration and permitted landfill assets that give a modest cost advantage and some barriers to entry. ROIC is only slightly above the cost of capital and improved gradually. Together this supports a narrow moat rather than a wide one.

**Returns: near.** ROIC rose from 8% to 11% over 2020-2024, around or modestly above the 8-10% cost of capital. Operating margins were steady at 14-17%, and free cash flow margins held at 11%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -1% | 14% | 8% | 11% | 8% | 12% | 4% |
| 2021-12-31 | 11% | 16% | 10% | 11% | 9% | 15% | 5% |
| 2022-12-31 | 17% | 15% | 10% | 11% | 10% | 16% | 6% |
| 2023-12-31 | 10% | 16% | 10% | 11% | 10% | 17% | 6% |
| 2024-12-31 | 7% | 17% | 11% | 11% | 11% | 19% | 6% |

* **switching_costs = 1.** Long-term municipal contracts and exclusive franchises lock in volumes for a period. They are renewable and competitively bid, so switching costs are modest.
  * [2] "We seek to obtain long-term contracts for collecting recyclable, solid waste and industrial waste material under residential collection contracts with municipalities, exclusive franchise agreements, small-container and large-container contracts and environmental solutions service contracts."
* **cost_advantage = 1.** Vertical integration and route density, with a large fleet and owned disposal assets, give a local cost advantage. The text states this directly but offers no figures comparing costs with competitors.
  * [3] "Our goal is to develop the best vertically integrated market position to enable us to build density and improve returns."
  * [4] "We realize synergies from consolidating businesses into our existing operations, whether through acquisitions or public-private partnerships, which allows us to reduce capital expenditures and expenses associated with truck routing, personnel, fleet maintenance, inventories and back-office administration."
  * [5] "Based on an industry trade publication, we operate the third largest vocational fleet in the United States."
* **efficient_scale = 1.** Landfills and transfer stations are scarce permitted assets, and permitting takes years with uncertain outcomes. This limits new entrants in local markets, but the text only implies it.
  * [1] "Development projects, while generally less capital intensive than acquisitions, typically require extensive permitting efforts that can take years to complete with no assurance of success."
* Rated 0: intangibles, network_effects.
* Against a moat:
  * [6] "We believe the total addressable North American environmental services market in which we operate generates approximately $165 billion of annual revenue"
  * [7] "Environmental solutions remains fragmented, which provides consolidation opportunities to drive scale."
  * [8] "We believe our ability to successfully complete these acquisitions is enhanced by the challenges facing many privately-held companies, including increasing competition in the environmental services industry"
  * [9] "We operate throughout North America, but the physical collection and recycling or disposal of material is very much a local business, and the dynamics and opportunities differ in each of the markets we serve."

### UAL: United Airlines Holdings, Inc. (Industrials, 10-K filed 2025-02-27)

> Overview the Company (together with its consolidated subsidiaries, "the Company" or the "Company") is a holding company and its wholly-owned subsidiary is the Company (together with its consolidated subsidiaries, "United"). United's shared purpose is "Connecting People. Uniting the World." United ...

**Overall: none.** The airline industry is highly competitive and open, with price matching and low-cost rivals. The route network and hub scale give only a modest network benefit. Returns are only recently above the cost of capital after heavy losses, which is not consistent evidence of a durable moat.

**Returns: unclear.** ROIC was deeply negative in 2020-2021, 7% in 2022, then 11% and 13% in 2023-2024. Recent returns are above the cost of capital but the history is volatile and cyclical, with a short track record of acceptable returns.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -65% | -41% | -46% | -38% | -23% | -81% | -13% |
| 2021-12-31 | 60% | -4% | -8% | -0% | -4% | -36% | -3% |
| 2022-12-31 | 82% | 5% | 2% | 3% | 7% | 12% | 1% |
| 2023-12-31 | 19% | 8% | 5% | -0% | 11% | 32% | 4% |
| 2024-12-31 | 6% | 9% | 6% | 7% | 13% | 29% | 4% |

* **network_effects = 1.** Hub-and-spoke gives frequency and connectivity benefits to a broad network, which is a modest scale-of-network advantage rather than a true user network effect; competitors have similar hub systems.
  * [2] "The hub and spoke system allows us to transport passengers between a large number of destinations with substantially more frequent service than if each route were served directly."
* Rated 0: intangibles, switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "The domestic airline industry is highly competitive and dynamic."
  * [6] "Currently, any U.S. carrier deemed fit by the U.S. Department of Transportation (the "DOT") is largely free to operate scheduled passenger service between any two points within the United States."
  * [7] "In order to remain competitive and maintain passenger traffic levels, we often find it necessary to match competitors' discounted fares."
  * [8] "Carriers with lower costs may offer lower fares to passengers, which could have a potential negative impact on the Company's revenues."
  * [9] "International competition has increased and may continue to increase in the future as a result of airline mergers and acquisitions, JBAs, alliances, restructurings, liberalization of aviation bilateral agreements and new or increased service by competitors."

### CDW: CDW Corp (Information Technology, 10-K filed 2025-02-21)

> Our Company the Company (together with its subsidiaries, the “Company,” “the Company” or “we”), a Fortune 500 company and member of the S&P 500 Index, is a leading multi-brand provider of information technology (“IT”) solutions to business, government, education and healthcare customers in the ...

**Overall: narrow.** Returns are high and steady, but the text shows only a modest scale/purchasing advantage in a fragmented, competitive, low-margin reselling market with commodity-like products and easily available supply. Dependence on large vendors adds to the weakness. Steady returns and some scale benefits support a narrow moat at most, with little else in the text to explain them.

**Returns: above.** ROIC of 18-23% over five years is consistently above the cost of capital, with stable operating margins of 6-8%. ROE is inflated by leverage. Returns are steady even as revenue fell in 2023 and 2024.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | 6% | 4% | 6% | 23% | 70% | 9% |
| 2021-12-31 | 13% | 7% | 5% | 3% | 20% | 99% | 9% |
| 2022-12-31 | 14% | 7% | 5% | 5% | 19% | 97% | 8% |
| 2023-12-31 | -10% | 8% | 5% | 7% | 19% | 61% | 8% |
| 2024-12-31 | -2% | 8% | 5% | 5% | 18% | 49% | 8% |

* **cost_advantage = 1.** Scale and vendor-tier pricing give some purchasing benefits, but the text offers no comparison with rivals and large distributors and OEMs sell directly, so the advantage looks modest.
  * [6] "These certifications also provide us with access to favorable pricing, tools and resources, including vendor incentive programs, which we use to provide additional value to our customers."
  * [7] "Our scale enables us to have a national and international footprint, as well as invest in resources to meet specific customer end-market needs."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "The market for technology products, solutions and services is highly competitive and subject to economic conditions and rapid technological changes."
  * [2] "These are highly fragmented markets served by thousands of IT resellers and solutions providers."
  * [3] "While innovation can help our business as it creates new offerings for us to sell, it can also disrupt our business model and create new and stronger competitors."
  * [4] "We believe competitive sources of supply are available in substantially all of the product categories that we offer."
  * [5] "In 2024, we generated $2.0 billion of Net sales from each of our three largest vendor partners."

### DELL: Dell Technologies Inc. (Information Technology, 10-K filed 2025-03-25)

> Company Overview the Company is a leader in the global technology industry focused on providing broad and innovative technology solutions for the data and artificial intelligence (“AI”) era. These solutions range from client devices and peripherals to infrastructure solutions across servers, ...

**Overall: narrow.** The text describes a commodity-like hardware business with intense price competition, new entrants and cloud providers buying directly from ODMs. The mechanisms it does cite are modest: scale in supply chain and direct sales, plus multiyear contracts. Reported ROIC is high and has been above the cost of capital for four years, which together with modest scale and switching-cost evidence supports a narrow moat at best. Margins are thin, so the advantage looks fragile rather than wide.

**Returns: above.** ROIC jumped from 7% in FY2021 to 16-28% in later years, comfortably above the 8-10% cost of capital. Margins are thin (4-7% operating) and FCF margin is volatile, and the ROIC calculation is flattered by very small or negative equity from buybacks and the VMware spin-off. Returns are above the cost of capital but with thin margins and modest growth.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-01-29 | 2% | 4% | 4% | 11% | 7% | 61% | 3% |
| 2022-01-28 | 17% | 5% | 5% | 7% | 16% | n/a | 5% |
| 2023-02-03 | 1% | 6% | 2% | 1% | 28% | n/a | 3% |
| 2024-02-02 | -14% | 6% | 4% | 7% | 25% | n/a | 4% |
| 2025-01-31 | 8% | 7% | 5% | 2% | 27% | n/a | 6% |

* **switching_costs = 1.** Multiyear as-a-Service and subscription agreements create some recurring revenue, but this is a small part of a hardware-heavy business and no integration lock-in is described.
  * [1] "Additionally, these offerings typically result in multiyear agreements which generate recurring revenue streams over the term of the arrangement."
* **cost_advantage = 1.** The company cites a large-scale supply chain and a direct sales model, but gives no figures or comparison with competitors showing a structural cost edge. These are mostly claims of scale.
  * [3] "Our world-class supply chain operates at a significant scale with the ability to remain agile in a variety of environments."
  * [4] "We believe our direct sales channel is a significant competitive advantage and emphasizes direct communication with customers, allowing us to refine our products and marketing programs while providing insight to better navigate through supply chain challenges and complexity."
* Rated 0: intangibles, network_effects, efficient_scale.
* Against a moat:
  * [5] "We face ongoing product and price competition in all areas of our business from both branded and generic competitors, including companies that specialize in one or more of our product or service lines."
  * [6] "We also face competition from non-traditional IT companies, including large Infrastructure-as-a-Service providers, that often buy their infrastructure directly from original design manufacturers."
  * [7] "We believe that new businesses will continue to enter these markets and develop technologies that, if successfully commercialized, may compete with our products and services."
  * [8] "Although we use our patented inventions and license some of them to others, we are not substantially dependent on any single patent or group of related patents."
  * [9] "In some cases, where multiple sources of supply are not available, we rely on a single source or a limited number of sources of supply if we believe it is advantageous to do so because of performance, quality, support, delivery, capacity, or price considerations."

### FICO: FAIR ISAAC CORP (Information Technology, 10-K filed 2025-11-07)

> Business GENERAL the Company (NYSE: the Company) (together with its consolidated subsidiaries, the “Company,” which may also be referred to in this report as “we,” “us,” “our,” and “the Company”) is a global analytics software leader. We were founded in 1956 on the premise that data, used ...

**Overall: wide.** The score standard, embedded in lending decisions and backed by consistently very high returns and margins, supports a durable intangible advantage. Software adds modest switching costs and a data network benefit. Dependence on the bureaus as distributors, the VantageScore rival, and regulatory scrutiny keep the moat from being clearly 20+ years, but it is strong; a wide rating is arguable, though the evidence leaves one source at 2 and returns well above the cost of capital.

**Returns: above.** ROIC is far above the 8-10% cost of capital in every year (41% rising to 64%), with operating margins expanding from 38% to 46% and steady free cash flow margins above 30%.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-09-30 | 2% | 38% | 30% | 32% | 41% | n/a | 25% |
| 2022-09-30 | 5% | 39% | 27% | 37% | 46% | n/a | 25% |
| 2023-09-30 | 10% | 42% | 28% | 31% | 52% | n/a | 28% |
| 2024-09-30 | 13% | 43% | 30% | 36% | 54% | n/a | 31% |
| 2025-09-30 | 16% | 46% | 33% | 39% | 64% | n/a | 36% |

* **intangibles = 2.** The credit score is the de facto U.S. standard, used in most credit decisions and by nearly all major lenders, and embedded in underwriting systems and regulatory/GSE practice. This is a strong brand/standard-based intangible, though distribution through bureaus and VantageScore pose some risk.
  * [1] "the Company® Score, which is the standard measure of consumer credit risk in the U.S. It is used in most U.S. credit decisions, by nearly all major banks, credit card issuers, mortgage lenders, and auto loan originators."
  * [2] "We also serve consumers through online services that enable people to access and understand their the Company Scores — the standard measure of consumer credit risk in the United States (“U.S.”)"
  * [6] "End users of our products include three-quarters of each of the largest 100 financial institutions in the U.S. and the largest 100 banks in the world."
  * [7] "Both scores maintain the same score to risk relationship as standard the Company® Scores, enhancing their compatibility with existing credit underwriting systems and models."
* **switching_costs = 1.** Software is embedded in mission-critical decision systems under multi-year subscriptions with minimums, and scores are built into lenders' underwriting models, making change costly.
  * [3] "We typically sell our software as multi-year subscriptions, with payments based on usage metrics such as the number of accounts, transactions or decisioning use cases deployed, often subject to contracted minimum payments."
  * [4] "Our predictive models are frequently used in mission-critical transactional systems and drive decisions and actions in near real time."
  * [7] "Both scores maintain the same score to risk relationship as standard the Company® Scores, enhancing their compatibility with existing credit underwriting systems and models."
* **network_effects = 1.** The Falcon fraud network pools data from over 10,000 institutions to improve models, a data-scale benefit, but it is a modest, indirect network effect.
  * [5] "Our models are continually improved using a proprietary, global data set of transaction data contributed by more than 10,000 institutions that participate in the the Company® Falcon® Intelligence Network."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "The market for our solutions is intensely competitive and is constantly changing."
  * [9] "However, many of our competitors are larger than the Company, have more development, sales and marketing resources than the Company, and some have larger shares of our target geographic or product markets."
  * [10] "During fiscal 2025, 2024 and 2023, revenues generated from our agreements with Experian, TransUnion and Equifax collectively accounted for 51%, 45% and 41% of our total revenues, respectively."
  * [11] "Primary competitors among outside suppliers of scoring models are the three major consumer reporting agencies in the U.S. and Canada, which are also our partners in offering our scoring solutions, and VantageScore (a joint venture entity established by the three major U.S. consumer reporting agencies), which is selling a credit scoring product competitive with our products."
  * [12] "The patents we hold may not be upheld as valid and may not prevent the development of competitive products."

### NTAP: NetApp, Inc. (Information Technology, 10-K filed 2025-06-09)

> Business Overview the Company (the Company, we, us, or the Company) helps customers make their data infrastructure more seamless, more dynamic, and higher performing. We were incorporated in 1992, are headquartered in San Jose, California, and provide a full range of enterprise-class software, ...

**Overall: narrow.** Returns are consistently very high, and the text shows one modest source: ONTAP's installed base and the unique native embedding in the hyperscale clouds. Competition is intense, on price as well, there are no long-term contracts, and two distributors account for 45% of revenue. The evidence supports a narrow moat at most, not a wide one.

**Returns: above.** ROIC is about 67% in the last two years, with ROE above 100% in most years (inflated by buybacks and low equity). Operating margins are steady at 16-20%, and revenue growth is flat to low. Returns are clearly well above cost of capital, though high leverage on a small equity base flatters the ratios.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-04-29 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2022-04-29 | n/a | 18% | 15% | 16% | n/a | >100% | 9% |
| 2023-04-28 | 1% | 16% | 20% | 14% | n/a | >100% | 13% |
| 2024-04-26 | -1% | 19% | 16% | 24% | 67% | 86% | 10% |
| 2025-04-25 | 5% | 20% | 18% | 20% | 67% | >100% | 11% |

* **intangibles = 1.** Unique position embedded natively in the three major clouds is a partnership-based differentiator, but the company itself says it is not dependent on any individual IP right, and partnerships can be terminated on short notice.
  * [1] "As the only enterprise-grade storage service natively embedded in the world’s largest clouds, we power data across Amazon AWS, Microsoft Azure, and Google Cloud."
  * [4] "we believe that our business is not materially dependent on any particular patent, trademark, copyright, license or other individual intellectual property right"
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "We face ongoing product and price competition in all areas of our business, including from both branded- and generic-product competitors."
  * [6] "Our markets are highly competitive, fragmented, and characterized by rapidly changing technology."
  * [7] "We typically do not enter into binding long-term purchase commitments with our customers, resellers, and distributors, meaning there is no guarantee that we will continue to receive large, recurring orders from them."
  * [8] "A substantial portion of our products is sold on the basis of standard purchase orders that are cancelable prior to shipment without penalty."
  * [9] "Sales to customers Arrow Electronics, Inc. and TD Synnex Corporation accounted for 21% and 24% of our net revenues, respectively, in fiscal 2025."
  * [10] "When industry supply is constrained or the supply chain is disrupted, our suppliers may allocate volumes away from us and to our competitors, who depend on many of the same suppliers as we do."

### ORCL: ORACLE CORP (Information Technology, 10-K filed 2025-06-18)

> Business the Company provides products and services that address enterprise information technology (IT) needs. Our products and services include enterprise applications and infrastructure offerings that are delivered worldwide through a variety of flexible and interoperable IT deployment models. ...

**Overall: narrow.** The entrenched database and applications installed base, with near-universal support renewals and mission-critical workloads, gives high switching costs. These are reflected in consistently high margins and returns. The filing itself cites low barriers to entry in many segments and intense competition from hyperscalers, and heavy capex in the new cloud business adds uncertainty, so the moat is narrow rather than wide.

**Returns: above.** Operating margins of 26-38% and ROIC above 100% in 2025 (small equity base after buybacks) and ROE above 85% point to consistently high returns. Margins are fairly steady, though FCF margin dropped to -1% in 2025 as data center capex rose.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-05-31 | 4% | 38% | 34% | 34% | n/a | >100% | 11% |
| 2022-05-31 | 5% | 26% | 16% | 12% | n/a | n/a | 6% |
| 2023-05-31 | 18% | 26% | 17% | 17% | n/a | n/a | 7% |
| 2024-05-31 | 6% | 29% | 20% | 22% | n/a | >100% | 8% |
| 2025-05-31 | 8% | 31% | 22% | -1% | >100% | 85% | 8% |

* **intangibles = 1.** Database and Java are described as the most popular in the world, an established installed-base position, but no patents or regulatory protection are cited and the text says patents are not material.
  * [5] "the Company Database is the world’s most popular enterprise database and is designed to enable reliable and secure storage, retrieval and manipulation of all forms of data."
  * [6] "Java is the world’s most popular programming language among professional developers"
* **switching_costs = 2.** Customers run core operations on the Company's database and applications, and substantially all support customers renew, which indicates high switching costs for mission-critical systems. The text does not quantify the lock-in.
  * [1] "Substantially all license support customers renew their support contracts with us upon expiration in order to continue to benefit from technical support services and the periodic issuance of unspecified updates and enhancements, which current license support customers are entitled to receive."
  * [3] "Using the Company technologies, our customers build, deploy, run, manage and support their internal and external products, services and business operations"
  * [8] "Substantially all of our customers opt to purchase license support contracts when they purchase these licenses to run within the the Company Cloud or other cloud-based and on-premise IT environments."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [9] "In addition, due to the low barriers to entry in many of our market segments, new technologies and new and growing competitors frequently emerge to challenge our offerings."
  * [10] "Our enterprise cloud, license and hardware offerings compete directly with certain offerings from some of the largest and most competitive companies in the world"
  * [11] "Key competitive factors in each of the segments in which we currently compete and may compete in the future include: total cost of ownership, performance, scalability, reliability, security, functionality, efficiency, ease of use, speed to production and quality of technical support."

### PANW: Palo Alto Networks Inc (Information Technology, 10-K filed 2025-08-29)

> General the Company. is a global cybersecurity provider and our vision is a world where each day is safer and more secure than the one before. We were incorporated in 2005 and are headquartered in Santa Clara, California. Our mission is to be the cybersecurity partner of choice for enterprises, ...

**Overall: narrow.** Customers face some friction from the installed firewall base, subscription renewals and platform consolidation, and recent returns are above the cost of capital. But the text shows only modest switching costs, with intense competition from large and well-funded vendors, a fragmented field, and a reliance on three distributors. The advantage looks real but narrow, and the returns record is short.

**Returns: above.** ROIC was negative in FY2021-22, then rose to 14%, 17% and 21% in FY2023-25 as operating margin grew to 13%; FCF margins are consistently high at 33-39%. Returns are now above the cost of capital but the track record is only three years, and the trend is recent.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-07-31 | 25% | -7% | -12% | 33% | -15% | -57% | -5% |
| 2022-07-31 | 29% | -3% | -5% | 33% | -8% | -63% | -2% |
| 2023-07-31 | 25% | 6% | 6% | 38% | 14% | 45% | 3% |
| 2024-07-31 | 16% | 9% | 32% | 39% | 17% | 75% | 15% |
| 2025-07-31 | 15% | 13% | 12% | 38% | 21% | 17% | 5% |

* **switching_costs = 1.** Renewing support subscriptions, a common PAN-OS across the firewall installed base, and existing Panorama deployments imply moderate integration and switching friction, though the text does not state costs directly.
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

**Overall: wide.** The company holds an exclusive franchise for .com and .net, the dominant domains, under long-running, presumptively renewing agreements, which yields extremely high and steady margins and returns. Pricing is regulated (7% annual caps in four of six years) and renewal depends on ICANN/DOC, with some substitution risk from social media and platforms. The franchise is nearly unreplicable and has lasted decades, which supports a wide moat.

**Returns: above.** Operating margins are 65-68% and stable, FCF margins 54-57%, and ROA 36-50%. ROIC is not reported but is clearly very high given the margins and asset-light model. Returns are consistently far above the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 3% | 65% | 64% | 54% | n/a | n/a | 45% |
| 2021-12-31 | 5% | 65% | 59% | 57% | n/a | n/a | 42% |
| 2022-12-31 | 7% | 66% | 47% | 56% | n/a | n/a | 36% |
| 2023-12-31 | 5% | 67% | 55% | 54% | n/a | n/a | 47% |
| 2024-12-31 | 4% | 68% | 50% | 56% | n/a | n/a | 50% |

* **intangibles = 2.** The company holds the exclusive, government-sanctioned right to operate .com and .net, with presumptive renewal and auto-renewing agreements with ICANN and the DOC. This is a regulatory/exclusive licence that no competitor can replicate for these domains, though renewal is not guaranteed and pricing is capped.
  * [1] "Although the .com Registry Agreement contains a “presumptive” right of renewal, ICANN could terminate or refuse to renew the Registry Agreement in certain prescribed circumstances."
  * [2] "On November 30, 2024, the Cooperative Agreement was automatically renewed on the same terms for a successive six-year term and will automatically renew on November 30, 2030, unless the DOC provides written notice of non-renewal within 120 days prior to the end of the then-current term."
  * [3] "The RZMA is subject to an automatic eight-year renewal, unless earlier modified or terminated."
  * [4] "The .com Registry Agreement permits an increase to the Maximum Price (as defined in the .com Registry Agreement) of .com domain name registrations by up to 7% over the previous year in each of the final four years of each six-year period."
* **switching_costs = 1.** Registrants of .com/.net names cannot move their domain to another registry without changing the name, and the company is the sole authoritative directory. But the text does not frame this as switching costs directly.
  * [6] "We operate the authoritative directory, for all .com, .net, and .name domain names"
* **efficient_scale = 2.** The company is the sole operator of the .com and .net registries and runs two of only thirteen root servers, a market that supports a single provider per TLD.
  * [5] "We operate two of the world’s thirteen root servers."
  * [6] "We operate the authoritative directory, for all .com, .net, and .name domain names"
* Rated 0: network_effects, cost_advantage.
* Against a moat:
  * [7] "Under our agreements with ICANN, we are subject to certain restrictions in the operation of .com, .net and .name on pricing, bundling, marketing, methods of distribution, introduction of new registry services, and use of registrars, that do not apply to ccTLDs and other gTLDs and therefore may create a competitive disadvantage."
  * [8] "Demand for domain names could be negatively impacted to the extent end-users establish their online identities using social media (such as Facebook, Instagram or Tiktok) or transact business on e-commerce platforms (such as Amazon, Etsy and Taobao) instead of registering domain names."
  * [9] "Some of the software and protocols used in our business are in the public domain or are otherwise available to our competitors, and some are based on open standards set by organizations such as the Internet Engineering Task Force."

### ALB: ALBEMARLE CORP (Materials, 10-K filed 2025-02-12)

> Business. the Company was incorporated in Virginia in 1993. Our principal executive offices are located at 4250 Congress Street, Suite 900, Charlotte, North Carolina 28209. Unless the context otherwise indicates, the terms “the Company,” “we,” “us,” “our” or “the Company” mean the Company and its ...

**Overall: none.** The company holds some owned lithium and bromine resources, which could offer a cost position, but lithium is a competitive, index-priced commodity with aggressive new entry and Specialties and Ketjen compete on price and performance. Returns are volatile and mostly near or below the cost of capital, with a sharply negative 2024. There is no evidence of a durable moat.

**Returns: below.** ROIC was 6%, 8%, 22%, 2% and -11% over 2020-2024. It spiked in the 2022 lithium price peak and turned deeply negative in 2024. Margins and returns are highly volatile and mostly below the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -13% | 16% | 12% | -2% | 6% | 9% | 4% |
| 2021-12-31 | 6% | 24% | 4% | -18% | 8% | 2% | 1% |
| 2022-12-31 | >100% | 34% | 37% | 9% | 22% | 38% | 20% |
| 2023-12-31 | 31% | 3% | 16% | -9% | 2% | 18% | 9% |
| 2024-12-31 | -44% | -33% | -22% | -18% | -11% | -12% | -7% |

* **cost_advantage = 1.** Ownership stakes in the Greenbushes mine and brine resources (Atacama, Silver Peak, Arkansas) suggest a resource-based cost position, but the text gives no cost comparison with competitors, and lithium is increasingly priced on market indices. Modest and uncertain.
  * [1] "We obtain lithium: (a) by purchasing lithium concentrate from our 49%-owned joint venture, Windfield Holdings Pty. Ltd. (“Windfield”), which directly owns 100% of the equity of Talison Lithium Pty. Ltd., a company incorporated in Australia (“Talison”) that owns the Greenbushes mine"
  * [2] "Our bromine production operations in Arkansas are supported by an active brine rights leasing program."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [5] "The global lithium market is highly competitive and growing very rapidly."
  * [6] "Competition in the global lithium market is increasingly based on index-based market pricing and differentiated via product quality, product diversity, reliability of supply and customer service."
  * [7] "Product performance and quality, price and contract terms are the primary factors in determining which qualified supplier is awarded a contract."
  * [8] "It is characterized by aggressive expansion and entry from existing and new players, including automotive OEMs, commodity traders, junior miners, and large, well-capitalized diversified miners."
  * [9] "The cost of raw materials is generally based on market prices, although we may use contracts with price caps or other tools, as appropriate, to mitigate price volatility."

### MOS: MOSAIC CO (Materials, 10-K filed 2025-03-03)

> OVERVIEW The the Company Company is the world’s leading producer and marketer of concentrated phosphate and potash crop nutrients. Through our broad product offering, we are a single source supplier of phosphate- and potash-based crop nutrients and animal feed ingredients. We serve customers in ...

**Overall: none.** The company sells commodity crop nutrients competing mainly on price. It has some cost advantages from integration and scale, but returns are only above the cost of capital at the cycle peak and have fallen back below it. No durable moat is shown.

**Returns: below.** ROIC was 2% in 2020, 14% and 26% in the 2021-22 price spike, then 7% in 2023 and 3% in 2024. Returns are cyclical and mostly below the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | 5% | 8% | 5% | 2% | 7% | 3% |
| 2021-12-31 | 42% | 20% | 13% | 7% | 14% | 16% | 8% |
| 2022-12-31 | 55% | 25% | 19% | 14% | 26% | 31% | 16% |
| 2023-12-31 | -28% | 10% | 9% | 7% | 7% | 10% | 5% |
| 2024-12-31 | -19% | 6% | 2% | 0% | 3% | 1% | 1% |

* **intangibles = 1.** Patented MicroEssentials is a modest differentiator in an otherwise commodity business; brand claims are generic.
  * [10] "MicroEssentials® is a value-added ammoniated phosphate product that is enhanced through a patented process that creates very thin platelets of sulfur and other micronutrients, such as zinc, on the granulated product."
* **cost_advantage = 1.** Integrated rock supply, scale, sulfur logistics and the world's largest potash mine give a cost position, but it is stated as belief, and foreign subsidized competitors may have lower costs. Advantage looks moderate.
  * [4] "We believe that we are a low-cost integrated producer of phosphate-based crop nutrients, due in part to our scale, vertical integration and strategic network of production and distribution facilities."
  * [5] "We believe that our position as an integrated producer of phosphate rock provides us with a significant cost advantage over competitors that are non-integrated phosphate producers."
  * [6] "We believe that our investments in sulfur logistical and melting assets continue to afford us a competitive advantage compared to other producers in cost and access to sulfur."
  * [7] "Esterhazy, the largest potash mine in the world, has completed full ramp up of capacity and production."
  * [11] "We are the world’s second largest miner of phosphate rock (excluding China) and currently operate four mines in North America with a combined annual capacity of 17.2 million tonnes."
* Rated 0: switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "Because crop nutrients are global commodities available from numerous sources, crop nutrition companies compete primarily on the basis of delivered price."
  * [2] "Potash is a commodity available from several geographical regions around the world and, consequently, the market is highly competitive."
  * [3] "Foreign competitors may have access to less expensive raw materials, may not have to comply with as stringent regulatory requirements or are owned or subsidized by governments and, as a result, may have cost advantages over North American companies."
  * [13] "Phosphate producers compete primarily based on price, as well as product quality, service and innovation."

### PKG: PACKAGING CORP OF AMERICA (Materials, 10-K filed 2025-02-27)

> B USINESS the Company of America (“we,” “us,” “our,” “the Company,” or the “Company”) is the third largest producer of containerboard products and a leading producer of uncoated freesheet (UFS) paper in North America. We operate eight mills and 86 corrugated products plants and related facilities. ...

**Overall: none.** Returns are above the cost of capital, but the text identifies no direct moat source. Containerboard and UFS are largely undifferentiated and compete on price, and the company says no producer is dominant. Returns look cyclical rather than protected, so there is no moat.

**Returns: above.** ROIC was 12-20% over 2020-2024 (14% in the last two years), above a 8-10% cost of capital. However, this reflects cyclical, commodity-driven margins (operating margin 11-17%), with a decline from the 2022 peak.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -4% | 11% | 7% | 9% | 12% | 15% | 6% |
| 2021-12-31 | 16% | 16% | 11% | 6% | 19% | 25% | 11% |
| 2022-12-31 | 10% | 17% | 12% | 8% | 20% | 28% | 13% |
| 2023-12-31 | -8% | 14% | 10% | 11% | 14% | 20% | 9% |
| 2024-12-31 | 7% | 13% | 10% | 6% | 14% | 19% | 9% |

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Certain containerboard grades and UFS paper products cannot generally be differentiated by producer, which tends to intensify price competition."
  * [2] "Our industries are highly competitive, with no single containerboard, corrugated packaging, or UFS paper producer having a dominant position."
  * [3] "Commodity grades of UFS paper are globally traded, with numerous worldwide manufacturers, and as a result, these products compete primarily on the basis of price."
  * [4] "We have little influence over the timing and extent of price changes of our products, which may be unpredictable and volatile."
  * [5] "According to industry sources, corrugated products are produced by about 380 U.S. companies operating approximately 1,100 plants."
  * [9] "Increasing shifts to these alternatives have had, and are likely to continue to have, an adverse effect on traditional print media and paper usage and lower demand for communication papers."

### ESS: ESSEX PROPERTY TRUST, INC. (Real Estate, 10-K filed 2025-02-21)

> OVERVIEW the Company (“the Company”), a Maryland corporation, is an S&P 500 company that operates as a self-administered and self-managed real estate investment trust (“REIT”). the Company owns all of its interest in its real estate and other investments directly or indirectly through the Company ...

**Overall: none.** The only moat-like feature is the supply-constrained West Coast markets, which benefits all owners and is not a company-specific advantage. Competition is broad, with many housing alternatives and larger rivals. Returns were near or below the cost of capital for four of five years. The 2024 improvement is not enough to establish a moat.

**Returns: near.** ROIC of 6-8% from 2020-2023 was below or at the cost of capital, rising to 11% in 2024. ROE was mostly 7-9% before jumping to 14% in 2024. Margins are fairly stable, but returns are only near the cost of capital and the 2024 jump is recent and may reflect one-offs.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | 2% | 33% | 38% | 54% | 6% | 9% | 4% |
| 2021-12-31 | -4% | 37% | 34% | 63% | 6% | 8% | 4% |
| 2022-12-31 | 12% | 37% | 25% | 61% | 7% | 7% | 3% |
| 2023-12-31 | 4% | 35% | 24% | 59% | 8% | 7% | 3% |
| 2024-12-31 | 6% | 40% | 42% | 60% | 11% | 14% | 6% |

* **efficient_scale = 1.** The Company targets markets with supply constraints (limited land, growth barriers, permitting), which protects existing assets from new competing housing. But this is a market characteristic that all owners share, not a firm-specific advantage, and many alternatives exist.
  * [1] "Constraints on new supply driven by: (i) low availability of developable land sites where competing housing could be economically built; (ii) political growth barriers, such as protected land, urban growth boundaries, and potential lengthy and expensive development permit processes; and (iii) natural limitations to development, such as mountains or waterways;"
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [2] "There are numerous housing alternatives that compete with the Company’s communities in attracting tenants."
  * [3] "Some competitors are larger and have greater resources than the Company."
  * [4] "If the demand for the Company’s communities is reduced or if competitors develop and/or acquire competing housing, rental rates and occupancy may drop which may have a material adverse effect on the Company’s financial condition and results of operations."

### IRM: IRON MOUNTAIN INC (Real Estate, 10-K filed 2025-02-14)

> BUSINESS OVERVIEW the Company, a Delaware corporation ("IMI"), was founded in an underground facility near Hudson, New York in 1951 where it stored business records. Today, we are a global leader in information management services, and we are trusted by more than 240,000 customers in 61 countries, ...

**Overall: none.** Customer stickiness from contracts and physical storage is real, but returns on invested capital sit below the cost of capital and are declining. Competition includes price and numerous data center operators. Returns do not support a moat despite the retention characteristics.

**Returns: below.** ROIC was 6-8% over five years, at or below the 8-10% cost of capital, and it has fallen to 6%. Operating margins have declined from 23% to 16%, and free cash flow has turned negative amid heavy growth investment and restructuring.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -3% | 23% | 8% | 13% | 8% | 26% | 2% |
| 2021-12-31 | 8% | 19% | 10% | 3% | 7% | 45% | 3% |
| 2022-12-31 | 14% | 21% | 11% | 1% | 8% | 75% | 4% |
| 2023-12-31 | 7% | 17% | 3% | -4% | 6% | 43% | 1% |
| 2024-12-31 | 12% | 16% | 3% | -10% | 6% | n/a | 1% |

* **switching_costs = 1.** Multi-year contracts, high retention, and long data center leases (10.6-year weighted average) show real stickiness. Physically moving stored records is costly. The advantage is moderate, and the text does not quantify it.
  * [1] "In our Records Management business, our contracted storage rental fee agreements generally range from one to five years in length."
  * [2] "we stored more than 730 million cubic feet of physical volume and we have consistently experienced strong customer retention levels"
  * [3] "In our Global Data Center Business, our lease durations vary by customer, with a weighted average lease expiration of 10.6 years as of December 31, 2024."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "Competition for records and information customers is driven by factors such as pricing, reputation and reliability, the quality and security of storage solutions and the scope and scale of technology."
  * [7] "We also compete with numerous data center developers, owners and operators, many of whom own properties comparable to ours in several of the same metropolitan areas where our facilities are located."

### O: REALTY INCOME CORP (Real Estate, 10-K filed 2025-02-25)

> Financial Statements the Company CORPORATION AND SUBSIDIARIES CONSOLIDATED BALANCE SHEETS (in thousands, except per share amounts) December 31, 2024 December 31, 2023 ASSETS Real estate held for investment, at cost: Land $ 17,320,520 $ 14,929,310 Buildings and improvements 40,974,535 34,657,094 ...

Not tagged: the located Item 1 'CONSOLIDATED BALANCE SHEETS' in the first 10,000 characters.

### CNP: CENTERPOINT ENERGY INC (Utilities, 10-K filed 2025-02-20)

> Business This combined Form 10-K is filed separately by three registrants: the Company., the Company Houston Electric, LLC and the Company Resources Corp. Information contained herein relating to any individual registrant is filed by such registrant solely on its own behalf. Each registrant makes ...

**Overall: none.** The monopoly territories and regulatory certificates are real barriers to entry, but regulators cap returns. ROIC of about 5% is below the cost of capital, which counts against a moat. The gas business also has non-exclusive franchises and competes with alternative fuels. The protected positions earn only regulated returns, so there is no moat.

**Returns: below.** ROIC has been about 4-5% over five years, well below an 8-10% cost of capital. ROE of 9-11% is modest and net margins swung in 2020. Operating margins are rising, and returns are stable, but low. A regulated utility's returns are capped by its regulators.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -2% | 14% | -10% | -8% | 4% | -9% | -2% |
| 2021-12-31 | 13% | 16% | 18% | -38% | 5% | 17% | 4% |
| 2022-12-31 | 12% | 17% | 11% | -28% | 5% | 11% | 3% |
| 2023-12-31 | -7% | 20% | 11% | -6% | 5% | 9% | 2% |
| 2024-12-31 | -1% | 23% | 12% | -27% | 5% | 10% | 2% |

* **intangibles = 1.** Regulatory certificates (CCN, exclusive Indiana territory certificates) and long-term franchises keep competitors out of the electric territories. Gas franchises are mostly non-exclusive, and regulators set rates.
  * [2] "For another provider of transmission and distribution services to provide such services in Houston Electric’s territory, it would be required to obtain a CCN from the PUCT and, depending on the location of the facilities, may also be required to obtain franchises from one or more municipalities."
  * [3] "Indiana service territory certificates are exclusive."
  * [5] "The terms of the franchises, with various expiration dates, typically range from 30 to 40 years."
* **efficient_scale = 1.** The electric T&D businesses are monopolies in their territories, with no other T&D utility present and regulatory approval required for entry. This is a natural-monopoly, regulated franchise. The gas business is less protected because its franchises are non-exclusive and it competes with alternative fuels.
  * [1] "There are no other electric transmission and distribution utilities in Houston Electric’s service area."
  * [4] "There are no other electric transmission and distribution utilities in Indiana Electric’s service area."
  * [2] "For another provider of transmission and distribution services to provide such services in Houston Electric’s territory, it would be required to obtain a CCN from the PUCT and, depending on the location of the facilities, may also be required to obtain franchises from one or more municipalities."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [6] "In most cases, franchises to provide natural gas utility services are not exclusive."
  * [7] "the Company and CERC’s Natural Gas compete primarily with alternate energy sources such as electricity and other fuel sources."
  * [8] "Houston Electric does not have long-term contracts with any of its customers, but rather operates using a continuous billing cycle, with meter readings being conducted and invoices being distributed to REPs each business day."
  * [9] "The PUCT and certain municipalities have the authority to set the rates and terms of service provided by Houston Electric under cost-of-service rate regulation."

### LNT: ALLIANT ENERGY CORP (Utilities, 10-K filed 2025-02-21)

> A. GENERAL the Company maintains its principal executive offices in Madison, Wisconsin. the Company operates as a regulated investor-owned public utility holding company, and its purpose-driven strategy is to serve its customers and build stronger communities. the Company primary focus is to ...

**Overall: none.** The company holds regulated monopoly territories with obligations to serve, which give protection through efficient scale and regulatory approvals. Regulators cap returns, and ROIC has stayed around 5%, below the cost of capital, so the franchise has not produced excess returns. ROE near 10-11% is acceptable for a utility. The protection looks real but modest, and the returns do not support a moat.

**Returns: below.** ROIC is 4-5% in all five years, below an 8-10% cost of capital. ROE is 10-11% and steady, and margins are stable. FCF margins are negative until 2023 because of heavy capital spending.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -6% | 22% | 18% | -25% | 5% | 11% | 4% |
| 2021-12-31 | 7% | 22% | 18% | -16% | 5% | 11% | 4% |
| 2022-12-31 | 15% | 22% | 16% | -24% | 5% | 11% | 4% |
| 2023-12-31 | -4% | 23% | 17% | 22% | 5% | 11% | 3% |
| 2024-12-31 | -1% | 22% | 17% | 29% | 4% | 10% | 3% |

* **intangibles = 1.** Regulatory certificates and permits are required to build generation, and Iowa law supports IPL's gas service. These give some protection but are common to regulated utilities, and returns are capped by regulators.
  * [3] "IPL must obtain a certificate of public convenience, use and necessity (GCU Certificate) from the IUC in order to construct a new, or significantly alter (including fuel switching) an existing, EGU or energy storage facility located in Iowa with a nameplate generating capacity of 25 MW or more."
  * [4] "In Iowa, counties and cities are prohibited from regulating the sale of natural gas and propane, which supports IPL’s ability to provide gas utility service to retail customers and industries."
* **efficient_scale = 1.** Retail electric and gas customers cannot choose another supplier, and the utilities have service obligations, so the territories are effectively regulated monopolies. Franchises are non-exclusive and municipalization and self-generation remain threats.
  * [1] "Retail electric customers in Iowa and Wisconsin currently do not have the ability to choose their electric supplier, and IPL and WPL have obligations to serve all their retail electric customers."
  * [2] "Gas customers in Iowa and Wisconsin currently do not have the ability to choose their gas distributor, and IPL and WPL have obligations to serve all their gas customers."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [5] "Although electric service in Iowa and Wisconsin is regulated, IPL and WPL still face competition from self-generation by large industrial customers, customer- and third party-owned generation (e.g. solar panels), alternative energy sources, and petitions to municipalize (Iowa) as well as service territory expansions by municipal utilities through annexations (Wisconsin)."
  * [6] "IPL provides utility services to incorporated communities as directed by the IUC and utilizes non-exclusive franchises, which cover the use of public right-of-ways for utility facilities in incorporated communities for a maximum term of 25 years."
  * [7] "If the utility’s actual return on common equity is outside of this range, future rates could be adjusted."

### PEG: PUBLIC SERVICE ENTERPRISE GROUP INC (Utilities, 10-K filed 2025-02-25)

> BUSINESS We were incorporated under the laws of the State of New Jersey in 1985 and our principal executive offices are located at 80 Park Plaza, Newark, New Jersey 07102. We are a public utility holding company that, acting through our wholly owned subsidiaries, is a predominantly regulated ...

**Overall: narrow.** PSE&G's franchised regulated monopoly is a real efficient-scale source, and the regulated rate base offers stable, allowed returns. But ROIC is near or below the cost of capital, and the nuclear business is a merchant exposed to market prices. The ROIC figures include the mark-to-market and one-off swings of the nuclear and legacy business. The regulated utility franchise supports a narrow moat, offsetting the weak returns.

**Returns: below.** ROIC was 6%, -2%, 4%, 9% and 5% over 2020-2024, mostly below the 8-10% cost of capital. ROE ranged from -4% to 18%. Margins and returns are volatile, partly because of nuclear and legacy items. Utility returns are capped by regulators and sit near the cost of capital.

| Fiscal year end | Revenue growth | Operating margin | Net margin | FCF margin | ROIC | ROE | ROA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-12-31 | -5% | 24% | 20% | 2% | 6% | 12% | 4% |
| 2021-12-31 | 1% | -9% | -7% | -10% | -2% | -4% | -1% |
| 2022-12-31 | 1% | 14% | 11% | -14% | 4% | 7% | 2% |
| 2023-12-31 | 15% | 33% | 23% | 4% | 9% | 18% | 5% |
| 2024-12-31 | -8% | 23% | 17% | -12% | 5% | 11% | 3% |

* **intangibles = 1.** Regulatory approvals set allowed returns on a large rate base, and the NRC licenses cover the nuclear plants. These are real, but the returns are capped by regulators and the nuclear business is exposed to market prices.
  * [4] "Our current approved transmission rates provide for a base ROE of 9.90% and a 50 basis point adder for our membership in PJM as an RTO."
  * [5] "The Order provides for a $17.8 billion rate base, a 9.6% return on equity for PSE&G’s distribution business and a 55% equity component of its capitalization structure."
* **efficient_scale = 2.** PSE&G holds franchised T&D territories covering 74% of New Jersey's population, in a natural monopoly with regulated returns. A competitor could not profitably duplicate the wires. Transmission formula rates and decoupling make earnings predictable.
  * [1] "PSE&G distributes electricity and natural gas to end users in our respective franchised service territories."
  * [2] "PSE&G—A New Jersey corporation, incorporated in 1924, which is a franchised public utility in New Jersey."
  * [3] "We serve the most densely populated, commercialized and industrialized territory in New Jersey, including its six largest cities and approximately 300 suburban and rural communities."
  * [6] "over 90% of our Electric and Gas Distribution margin will only vary based upon the number of customers."
  * [7] "Our transmission revenues are not impacted by sales volumes."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [8] "Typically, the bid price of the last unit dispatched by PJM establishes the energy market-clearing price."
  * [9] "Various market participants compete with us and one another in transacting in the wholesale energy markets and entering into bilateral contracts."
  * [10] "It is uncertain whether the OSA will be renewed."
  * [11] "Changes previously ordered by FERC and implemented by PJM and other ISOs to eliminate contractual provisions that previously provided us a “right of first refusal” to construct new transmission projects in our service territory could result in third-party construction of transmission lines in our area in the future"
