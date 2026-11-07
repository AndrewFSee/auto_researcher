# Moat tagger pilot

Generated 2026-11-02 17:22 by `scripts/moat_pilot.py`, set `sample`, model `claude-sonnet-5-5`, prompts `claude-sonnet-5-5__cbc3f896c__high`. Two passes over each 10-K's Item 1 (business): extract quotes per chunk and check them word for word against the text, then rate each moat source from the verified quotes only (`auto_researcher.llm.moat`).

* Companies: 50 (50 rated); chunks read 50 (Item 1 cut at 16 chunks for 0).
* Quotes returned: 488; kept 486; not found in the text 1 (0.2%); duplicates 1; over the cap 0; unparseable replies 0.
* Kept quotes by kind: intangibles 110, switching_costs 55, network_effects 15, cost_advantage 56, efficient_scale 33, against 217.
* Rating fixes (citations of missing or "against" quotes dropped, uncited ratings lowered to 0): 1.
* Model time: 6 min (7 s per company); API cost $2.43; served by another model 0.

## Ratings

0 = none or generic, 1 = specific but modest, 2 = strong and durable.

| Ticker | Sector | Overall | intang | switch | network | cost | eff.scale | Against quotes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTCH | Communication Services | narrow | 1 | 0 | 1 | 0 | 0 | 6 |
| TTWO | Communication Services | none | 1 | 0 | 0 | 0 | 0 | 5 |
| ABNB | Consumer Discretionary | narrow | 1 | 0 | 1 | 0 | 0 | 4 |
| CCL | Consumer Discretionary | narrow | 1 | 0 | 0 | 1 | 1 | 2 |
| CMG | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 4 |
| GM | Consumer Discretionary | none | 0 | 0 | 0 | 0 | 0 | 5 |
| MAR | Consumer Discretionary | narrow | 1 | 1 | 1 | 0 | 0 | 2 |
| CAG | Consumer Staples | none | 1 | 0 | 0 | 0 | 0 | 6 |
| DG | Consumer Staples | narrow | 0 | 0 | 0 | 1 | 1 | 3 |
| MO | Consumer Staples | narrow | 2 | 0 | 0 | 0 | 0 | 4 |
| TSN | Consumer Staples | none | 1 | 0 | 0 | 0 | 0 | 5 |
| EOG | Energy | none | 0 | 0 | 0 | 1 | 0 | 4 |
| BBT | Financials | none | 0 | 0 | 0 | 0 | 0 | 5 |
| COF | Financials | none | 0 | 0 | 0 | 0 | 0 | 4 |
| EG | Financials | none | 1 | 0 | 0 | 0 | 0 | 5 |
| MCO | Financials | wide | 2 | 1 | 0 | 0 | 0 | 3 |
| NTRS | Financials | none | 0 | 0 | 0 | 0 | 0 | 2 |
| PYPL | Financials | narrow | 1 | 0 | 1 | 0 | 0 | 4 |
| SCHW | Financials | narrow | 1 | 1 | 0 | 2 | 0 | 2 |
| ABBV | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 |
| DGX | Health Care | none | 0 | 0 | 0 | 0 | 0 | 3 |
| GILD | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 5 |
| HUM | Health Care | none | 0 | 0 | 0 | 1 | 0 | 5 |
| IDXX | Health Care | narrow | 1 | 1 | 0 | 0 | 0 | 5 |
| MOH | Health Care | none | 1 | 0 | 0 | 0 | 1 | 7 |
| REGN | Health Care | narrow | 1 | 0 | 0 | 0 | 0 | 7 |
| ALLE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 4 |
| AME | Industrials | none | 1 | 1 | 0 | 0 | 0 | 5 |
| CHRW | Industrials | none | 0 | 0 | 1 | 1 | 0 | 6 |
| DE | Industrials | narrow | 1 | 1 | 0 | 0 | 0 | 4 |
| LII | Industrials | none | 0 | 0 | 0 | 0 | 0 | 3 |
| PWR | Industrials | none | 0 | 1 | 0 | 0 | 0 | 5 |
| RSG | Industrials | narrow | 0 | 1 | 0 | 1 | 1 | 2 |
| UAL | Industrials | none | 0 | 0 | 1 | 0 | 0 | 5 |
| CDW | Information Technology | none | 0 | 0 | 0 | 1 | 0 | 6 |
| DELL | Information Technology | none | 0 | 0 | 0 | 1 | 0 | 5 |
| FICO | Information Technology | narrow | 2 | 1 | 1 | 0 | 0 | 5 |
| NTAP | Information Technology | none | 1 | 1 | 0 | 0 | 0 | 7 |
| ORCL | Information Technology | narrow | 1 | 2 | 0 | 0 | 0 | 5 |
| PANW | Information Technology | narrow | 0 | 1 | 0 | 0 | 0 | 5 |
| VRSN | Information Technology | wide | 2 | 0 | 0 | 0 | 2 | 3 |
| ALB | Materials | none | 0 | 0 | 0 | 1 | 0 | 6 |
| MOS | Materials | none | 0 | 0 | 0 | 1 | 0 | 4 |
| PKG | Materials | none | 0 | 0 | 0 | 0 | 0 | 7 |
| ESS | Real Estate | none | 0 | 0 | 0 | 0 | 1 | 3 |
| IRM | Real Estate | narrow | 1 | 1 | 0 | 0 | 0 | 3 |
| O | Real Estate | none | 0 | 0 | 0 | 1 | 0 | 1 |
| CNP | Utilities | narrow | 1 | 0 | 0 | 0 | 2 | 4 |
| LNT | Utilities | narrow | 1 | 0 | 0 | 0 | 2 | 3 |
| PEG | Utilities | narrow | 1 | 0 | 0 | 0 | 2 | 4 |

| Sector | none | narrow | wide |
| --- | ---: | ---: | ---: |
| Communication Services | 1 | 1 | 0 |
| Consumer Discretionary | 2 | 3 | 0 |
| Consumer Staples | 2 | 2 | 0 |
| Energy | 1 | 0 | 0 |
| Financials | 4 | 2 | 1 |
| Health Care | 3 | 4 | 0 |
| Industrials | 5 | 3 | 0 |
| Information Technology | 3 | 3 | 1 |
| Materials | 3 | 0 | 0 |
| Real Estate | 2 | 1 | 0 |
| Utilities | 0 | 3 | 0 |

## Notes

* **How the model was chosen (on the 5 `dev` companies only).** `qwen3:8b` (local) in three prompt rounds quoted well (2-3% of quotes not in the text) but rated badly: credit for network effects its own reasons denied, a cost advantage from 'sustainable technologies', a strong brand from 'a well-known portfolio of brands'; thinking did not fix it. Claude Haiku 5.5 and Sonnet 5.5 then ran the one-call prompt on the same 5; Sonnet weighed the evidence against a moat and found RTX's installed-base switching costs, Haiku rated more coarsely. The 50 were tagged once, with Sonnet 5.5 and the prompt fixed beforehand.
* **Deviation: the quote check was loosened after the run, before the hand review.** The first check rejected 21 of 488 quotes (4.3%), nearly all real: the text extracted from 10-K HTML has spaces around '®' and dashes ('Tinder ®', 'PSE&G —A') and one table row was quoted with bars. The check now ignores white space, trademark signs and table bars; it rejects 1 quote, a genuine paraphrase (MTCH: 'Pairs is a leading provider of online dating services in Japan'; the text says 'Pairs is a dating platform ...'). The model's replies were not rerun; only the citations it dropped changed (14 fixes before, 1 after).
* **Cost** $2.43 for the 50 (Claude API, sentiq's workspace: auto_researcher's own key has no credit).
* **Name hiding is partial.** Name, short name, self-defined aliases, tickers and web addresses are replaced, but brands and details (Tinder, F-35, United's hubs) identify many companies, and Claude knows how they fared after these 10-Ks (filed in 2025, before its training cutoff). The tags are fit for a hand check and for describing companies; a return test needs filings after the model's cutoff or a check that hiding more of the name does not change the tags.
* **Only Item 1** (business) is read, from one 10-K per company; moat evidence in MD&A or risk factors is missed, and 'against' quotes come from the competition sections Item 1 usually has.

## Hand check

50 companies reviewed on the review page (verdicts in `data/moat/verdicts.json`). Tickers with the wrong text: DGX (MD&A and the financial statements), TTWO (MD&A: impairments, critical accounting policies, liquidity), O (the financial statements, with the business description only at the end).

| Companies | Overall agreed | Too high | Too low | Source ratings agreed | Too high | Too low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all (50) | 41 (82%) | 1 | 8 | 228 of 250 (91%) | 1 | 21 |
| right text (47) | 40 (85%) | 1 | 6 | 215 of 235 (91%) | 1 | 19 |

| Tag | Reviewed (right text) | Agreed |
| --- | ---: | ---: |
| none | 23 | 19 |
| narrow | 22 | 19 |
| wide | 2 | 2 |

Disputed overall calls ("too low" = the moat should be higher):

* **TTWO**: tagged none, too low (wrong text). Should be higher. The tagger gave intangibles a 1 and still called it no moat. GTA, NBA 2K and Rockstar's track record are durable IP.
* **CCL**: tagged narrow, too high. Should be lower. Capacity is an oligopoly, but cruising has rarely earned returns above the cost of capital, and customers switch freely.
* **CMG**: tagged none, too low. Should be higher. Brand and unit economics are well above restaurant peers.
* **NTRS**: tagged none, too low. Should be higher. Custody switching costs and scale are the classic trust-bank moat; the tagger held the filing's silence against it.
* **DGX**: tagged none, too low (wrong text). Should be higher. Data issue: the input was MD&A, not Item 1. Re-run with the right section.
* **IDXX**: tagged narrow, too low. Should be higher. A razor-and-blade installed base in vet diagnostics with very high returns on capital.
* **AME**: tagged none, too low. Should be higher. Two modest sources (niche intangibles and qualification stickiness) and consistently strong margins fit narrow, not none.
* **LII**: tagged none, too low. Should be higher. Direct distribution and an oligopolistic residential HVAC market.
* **FICO**: tagged narrow, too low. Should be higher. It is the entrenched standard with extreme pricing power; VantageScore acceptance is the main risk.

Disputed source ratings:

* CMG: intangibles 0 too low; cost_advantage 0 too low
* MAR: intangibles 1 too low
* BBT: switching_costs 0 too low; cost_advantage 0 too low
* COF: network_effects 0 too low; cost_advantage 0 too low
* MCO: network_effects 0 too low; efficient_scale 0 too low
* NTRS: switching_costs 0 too low; cost_advantage 0 too low
* DGX: cost_advantage 0 too low; efficient_scale 0 too low
* IDXX: switching_costs 1 too low
* DE: intangibles 1 too low
* LII: intangibles 0 too low
* RSG: efficient_scale 1 too low
* UAL: network_effects 1 too high
* PANW: intangibles 0 too low
* VRSN: intangibles 2 too low
* PKG: switching_costs 0 too low
* IRM: switching_costs 1 too low

### What the hand check shows

* **One rater, with hindsight.** The verdicts are the user's, made with the company names shown and knowing how the companies fared since; they measure whether the tags look right to an informed reader, not a ground truth. Several are marked low confidence (CHRW, EOG, O).
* **The tagger errs low.** Of the overall calls the reviewer disputed on the right text, all but one say the moat should be higher, as do 21 of the 22 disputed source ratings, mostly 0s. The reasons given lean on returns: IDXX's returns on capital, AME's margins, CMG's unit economics, and the one call disputed as too high, CCL, because cruising rarely earns its cost of capital. Item 1 has no numbers, so the tagger cannot see returns; NTRS was held to a filing that never describes its custody moat.
* **A source rated 1 does not force a moat, and that is mostly right.** 18 companies were called none despite a source rated 1; the reviewer agreed on 16 of them.
* **Wrong text from the section offsets.** In a scan of all 480 10-Ks filed in 2025 that have an Item 1 span, 11 start like MD&A or financial statements in their first 3,000 characters (CL, ZTS, PLD, BALL, DGX, BG, CDNS, LH, O, AFL, MCHP); TTWO's span is wrong too but starts mid-sentence, so the scan misses it. Others run on into Item 1A (REGN, NTAP), which adds risk factors but keeps Item 1.

## Review sheet

For each company: what it does (the opening of Item 1, name hidden), the overall call, then each source rated above 0 with the model's reason and the quotes it cites, and the quotes against a moat. Sources rated 0 are listed by name only.

### MTCH: Match Group, Inc. (Communication Services, 10-K filed 2025-02-27)

> Who we are the Company, through its portfolio companies, is a leading provider of digital technologies designed to help people make meaningful connections. Our global portfolio of brands includes Tinder ® , Hinge ® , the Company ® , Meetic ® , OkCupid ® , PairsTM, Plenty Of Fish ® , Azar ® , BLK ® ...

**Overall: narrow.** The Company owns leading dating brands with some network-driven scale, but the industry is fragmented, users switch freely and use several apps, the company depends on Apple and Google, and large social platforms compete. That gives a modest, narrow brand and network advantage at best.

* **intangibles = 1.** Strong dating brands (Tinder, Hinge, Meetic, Pairs) with scale and recognition, plus a patented Swipe feature, but the market is fragmented and users multi-home, so the brand advantage is modest.
  * [1] "Brand is very important. Users generally associate strong brands with a higher likelihood of success and more tools to help the user connect safely and securely."
  * [3] "Tinder® was launched in 2012 and has since risen to scale and popularity faster than any other service in the online dating category."
  * [4] "Meetic is the most recognized dating app for singles over age 35 in Europe."
  * [5] "Tinder’s patented Swipe® technology has led to significant adoption, particularly among 18 to 25 year-old users"
* **network_effects = 1.** The text says successful brands depend on large, active user communities, which points to a local network effect in dating. It is stated only generally, and the market is fragmented with multi-homing, so the advantage is limited.
  * [2] "Generally, successful brands depend on large, active communities of users, strong algorithmic filtering technology, and awareness of successful usage among similar users."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The industry for social connection apps is competitive and has no single, dominant brand globally."
  * [8] "As a result, the market for social connection apps is fragmented, and no single service has been able to effectively serve all of those seeking social connections."
  * [9] "A large portion of customers use multiple services over a given period of time, either concurrently or sequentially, making our broad portfolio of brands a competitive advantage."
  * [10] "We pay Apple and Google a meaningful share of the revenue we receive from in-app transactions."
  * [11] "Our brands compete with each other and with third-party businesses on brand characteristics, service features, and business model"
  * [12] "Many of our brands enable users to establish a profile and review other users’ profiles without charge."
* Fixes: intangibles: dropped citations [6].

### TTWO: TAKE TWO INTERACTIVE SOFTWARE INC (Communication Services, 10-K filed 2025-05-20)

> for additional discussion. Impairments During the fiscal year ended March 31, 2025, we recognized Goodwill impairment charges of $3,545.2, representing a partial impairment related to one of our reporting units, and we recognized impairment charges of $137.0 for acquisition-related Developed Game ...

**Overall: none.** The Company has hit franchises, notably GTA, that give it some intangible strength. Against that, revenue is hit-driven, a few platform and retail customers hold bargaining power (Apple, Google, Sony, Microsoft), and there are multibillion-dollar goodwill impairments and operating losses. The text shows no durable, returns-protecting moat.

* **intangibles = 1.** Franchises such as Grand Theft Auto, NBA 2K and Civilization drive revenue, and GTA is a significant share of sales. The text gives little on why rivals cannot copy them, and large impairments show uncertainty in the portfolio.
  * [5] "Sales of Grand Theft Auto products generated 12.6% of our net revenue for the fiscal year ended March 31, 2025."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Apple and Google generally have the discretion to set the amounts of their platform fees and change their platforms’ terms of service and other policies with respect to us or other developers at their sole discretion, and those changes may be unfavorable to us."
  * [2] "Our five largest customers accounted for 81.0%, 79.8% and 79.6% of net revenue during the fiscal years ended March 31, 2025, 2024 and 2023, respectively."
  * [3] "Generally, a significant portion of our revenue has been derived from a few popular franchises, particularly around new releases within those franchises, some of which have annual or biennial releases."
  * [4] "The impairment charges are a result of a reduction in the forecasted performance of certain games due to industry conditions and changes in our strategies in response to those conditions."
  * [6] "There has been increased consolidation in our industry, as larger, better capitalized competitors will be in a stronger position to withstand prolonged periods of economic downturn and sustain their business through the financial volatility."

### ABNB: Airbnb, Inc. (Consumer Discretionary, 10-K filed 2025-02-13)

> Overview We operate a global marketplace, where hosts offer guests stays and experiences on our platform. We are a community based on connection and belonging—a community that was born in 2007 when two hosts welcomed three guests to their San Francisco home, and has since grown to over 5 million ...

**Overall: narrow.** The Company has a brand and a two-sided marketplace with a large supply of unique listings, which give it a network effect and some brand pull. However, hosts and guests can easily choose among many rivals, there are no switching costs or cost advantages, and regulators can restrict short-term rentals, as in New York City. The moat is narrow.

* **intangibles = 1.** The brand is said to reduce reliance on performance marketing, which suggests customers seek it out directly. No figures or pricing power are given, and the general claims about patents and trademarks do not count.
  * [3] "the strength of the the Company brand and our communications strategy allows us to be less reliant on performance marketing."
* **network_effects = 1.** The two-sided marketplace links 5 million hosts with guests, and listing supply and booking volume draw each side to the other. This is a real network effect, though hosts multi-home and rivals like Booking and Expedia are large.
  * [1] "Our technology platform powers our two-sided marketplace and enables our global network of hosts and guests."
  * [2] "has since grown to over 5 million hosts who have welcomed over 2 billion guest arrivals in almost every country and region across the globe"
  * [4] "We compete for hosts based on many factors including the volume of bookings generated by guests"
  * [5] "We compete for guests based on many factors, including unique inventory and availability of listings"
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "We operate in a highly competitive environment."
  * [7] "as hosts have a range of options for doing so"
  * [8] "as guests have a range of options to find and book accommodations and experiences"
  * [9] "For example, New York City passed regulations in 2023, which resulted in a de facto ban of short-term rental activities."

### CCL: CARNIVAL CORP (Consumer Discretionary, 10-K filed 2025-01-27)

> Business . A. Overview I. Summary the Company was incorporated in Panama in 1974 and the Company was incorporated in England and Wales in 2000. the Company and the Company operate a dual listed company (“DLC”), whereby the businesses of the Company and the Company are combined through a number of ...

**Overall: narrow.** The Company is the largest cruise operator in a concentrated industry, with well-known brands, scale purchasing and owned destinations. But customers switch easily, it faces competition from land-based vacations, and the text does not show durable pricing power. This points to a narrow moat at best.

* **intangibles = 1.** Several well-known cruise brands with long heritage and leading positions in specific markets (Germany, UK, Alaska tours) support pricing and demand, but cruise brands are not strongly protected and the text offers limited proof of pricing power.
  * [2] "the Company Cruise Line is “The World’s Most Popular Cruise Line®” and has provided multi-generational family entertainment at exceptional value to its guests for over 50 years."
  * [3] "AIDA is the most recognized brand in the German cruise market."
  * [4] "P&O Cruises (UK) is Britain’s largest cruise line and its heritage can be traced back over 185 years."
  * [9] "In addition to our cruise operations, we own Holland America Princess Alaska Tours, the leading tour company in Alaska and the Canadian Yukon, which complements our Alaska cruise operations."
* **cost_advantage = 1.** Scale purchasing and owned or preferential port access offer some cost and access advantages, but the quantified advantage over rivals is not shown.
  * [5] "We approach our spend strategically and look for suppliers who demonstrate the ability to help us leverage our scale in terms of cost, quality, service, innovation and sustainability."
  * [6] "In addition, to secure preferential berth access to third-party ports, we enter into berthing agreements and commitments."
  * [8] "We also have a Cruise Support segment that includes our portfolio of leading port destinations and exclusive islands as well as other services, all of which are operated for the benefit of our cruise brands."
* **efficient_scale = 1.** Four companies hold about 80% of industry capacity, an oligopoly, but new ship capacity keeps being added and the company also competes with land-based vacations, so the market clearly supports new entrants and growth.
  * [1] "we, along with our principal cruise competitors Royal Caribbean Group, Norwegian Cruise Line Holdings, Ltd. and MSC Cruises, represented approximately 80% of the cruise industry capacity."
* Rated 0: switching_costs, network_effects.
* Against a moat:
  * [10] "We compete with land-based vacation alternatives throughout the world, such as hotels, resorts (including all-inclusive resorts), theme parks, organized tours, casinos, vacation ownership properties, and other internet-based alternative lodging sites."
  * [11] "Our travel agents’ relationships are generally not exclusive and travel agents generally receive a base commission, plus the potential of additional commissions"

### CMG: CHIPOTLE MEXICAN GRILL INC (Consumer Discretionary, 10-K filed 2025-02-05)

> General the Company., a Delaware corporation, together with its subsidiaries (“the Company,” “we,” “us,” or “our”) owns and operates the Company restaurants, which feature a relevant menu of burritos, burrito bowls (a burrito without the tortilla), quesadillas, tacos, and salads. We strive to ...

**Overall: none.** The text describes a highly competitive industry with low barriers to entry and no switching costs. No quote states a durable advantage, so no moat can be shown from this text.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "The fast-casual, quick-service, and casual dining segments of the restaurant industry are highly competitive with respect to, among other things, taste, price, food quality and presentation, service, location, convenience, brand reputation, and cleanliness and ambience of each restaurant."
  * [2] "Among our main competitors are restaurant formats that claim to serve higher quality ingredients without artificial flavors, colors and preservatives, and that serve food quickly and at a reasonable price."
  * [3] "Certain key ingredients are purchased from a small number of suppliers."
  * [4] "Competition from food delivery services, which offer meals from a wide variety of restaurants, also has increased in recent years and is expected to continue to increase."

### GM: General Motors Co (Consumer Discretionary, 10-K filed 2025-01-28)

> the Company Company (sometimes referred to as we, our, us, ourselves, the Company, the Company or the Company) was incorporated as a Delaware corporation in 2009. We design, build and sell trucks, crossovers, cars and automobile parts and provide software-enabled services and subscriptions ...

**Overall: none.** A large automaker with a strong US truck franchise, but it competes on price, relies on incentives, and faces many competitors with cyclical markets and low switching costs. No mechanism is stated directly enough to support a moat.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The principal factors that determine consumer vehicle preferences in the markets in which we operate include overall vehicle design, price, quality, available options, safety, reliability, fuel economy or range and functionality."
  * [7] "The level of incentives is dependent upon the level of competition in the markets in which we operate and the level of demand for our products."
  * [8] "While no single piece of intellectual property is individually material to our business as a whole, our intellectual property is important to our operations and continued technological development."
  * [9] "Certain fleet transactions, particularly sales to daily rental car companies, are generally less profitable than retail sales to end customers."
  * [10] "Market leadership in individual countries in which we compete varies widely."

### MAR: MARRIOTT INTERNATIONAL INC /MD/ (Consumer Discretionary, 10-K filed 2025-02-11)

> Overview We are a worldwide operator, franchisor, and licensor of hotel, residential, timeshare, and other lodging properties under more than 30 brand names at different price and service points. Consistent with our focus on management, franchising, and licensing, we own or lease very few of our ...

**Overall: narrow.** Asset-light franchisor with large scale, a strong loyalty program and long contracts gives a narrow moat, but competition from Hilton, Hyatt, IHG and Airbnb is intense and owners can terminate for underperformance.

* **intangibles = 1.** Brands across luxury to midscale tiers and loyalty-driven repeat business are meaningful, but the text mostly asserts brand strength and competitors have comparable brands.
  * [1] "Although we believe that our strong brand recognition assists us in attracting and retaining guests and hotel owners, we compete against many other companies with strong brands and guest appeal"
  * [7] "We believe that our Loyalty Program generates substantial repeat business that might otherwise go to competing properties."
* **switching_costs = 1.** Long-term management (20-30 years) and franchise (10-20 years) agreements lock in owners, though owners can terminate for poor performance.
  * [3] "Such agreements are generally for initial periods of 20 to 30 years, with options for us to renew for up to 10 or more additional years."
  * [4] "Such agreements are generally for periods of 10 to 20 years."
* **network_effects = 1.** A loyalty program booking 65-72% of room nights attracts owners and guests, a mild two-sided effect, but it is a loyalty program, which is not a pure network effect.
  * [2] "In 2024, 72 percent of our U.S. hotel room nights and 65 percent of our global hotel room nights were booked by Loyalty Program members."
  * [6] "We believe that our brands are attractive to hotel owners seeking a management company, franchise, or other licensing affiliation because of the benefits of our Loyalty Program, centralized reservation systems, marketing programs, and other offerings, and our emphasis on guest service."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [8] "Many of our management agreements also permit the hotel owners to terminate the agreement if we do not meet certain performance metrics, financial returns fail to meet defined levels for a period of time, and we have not cured those deficiencies."
  * [9] "We encounter strong competition in the short-term lodging market from regional, national, and international chains that operate lodging properties or franchise their brands, lodging properties that are not affiliated with a chain, and online platforms, including Airbnb and Vrbo"

### CAG: CONAGRA BRANDS INC. (Consumer Staples, 10-K filed 2025-07-10)

> General Development of Business the Company (the “Company”, “the Company”, “we”, “us”, or “our”), headquartered in Chicago, is one of North America’s leading branded food companies. We combine a 100-year history of making quality food with agility and a relentless focus on collaboration and ...

**Overall: none.** Branded packaged food with some brand recognition, but intense competition, private label, commodity inputs, larger rivals, and heavy Walmart concentration. No clearly stated durable advantage, though brands give some modest edge.

* **intangibles = 1.** Owns well-known branded food portfolio and claims brand loyalty, but no figures on market share or pricing power; competition from private label and larger rivals makes the brand advantage modest.
  * [6] "We compete primarily on the basis of quality, product innovation, value, customer service, brand recognition, and brand loyalty."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "We experience intense competition for sales of our food items in our major markets."
  * [2] "Some of our competitors are larger and have greater resources than we have."
  * [3] "Our largest customer, Walmart, Inc. and its affiliates, accounted for approximately 29% of consolidated net sales for fiscal 2025 and 28% for fiscal 2024 and 2023."
  * [4] "Our food items compete with widely advertised, well-known, branded food, as well as private branded and customized food items."
  * [5] "We use many different raw materials, most of which are commodities, to make and package our products."
  * [7] "we believe such raw materials to be in adequate supply and generally available from numerous sources."

### DG: DOLLAR GENERAL CORP (Consumer Staples, 10-K filed 2025-03-21)

> ​ General ​ We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025, with the greatest concentration of stores in the southern, southwestern, midwestern and eastern United States. Our first stores ...

**Overall: narrow.** The Company has real scale and density advantages in rural small-box retail that support good returns, but the market is highly competitive on price with larger rivals and no switching costs, brand lock-in or network effects. Moat is modest.

* **cost_advantage = 1.** Scale as the largest small-box discounter and a low-cost, limited-assortment model support purchasing power and low costs, but competitors like Walmart have greater resources and the advantage is not clearly structural.
  * [1] "Our ability to offer everyday low prices on quality merchandise is supported by our low-cost operating approach and our strategy to maintain a limited number of items per merchandise category, which we believe helps us maintain strong purchasing power."
  * [2] "Purchasing large volumes of merchandise within our focused assortment in each merchandise category allows us to keep our average product costs low, contributing to our ability to offer competitive everyday prices to our customers."
  * [9] "We are the largest discount retailer in the United States by number of stores, with 20,662 stores located in 48 U.S. states and Mexico as of February 28, 2025"
* **efficient_scale = 1.** Many stores are in small towns with limited shopping alternatives, where a second store may not be profitable, but this is only loosely stated and competition is intense.
  * [3] "We believe our ability to effectively deliver both value and convenience allows us to succeed in small markets with limited shopping alternatives, as well as in larger and more competitive markets."
  * [4] "approximately 80% of our stores are located in towns of 20,000 or fewer people."
* Rated 0: intangibles, switching_costs, network_effects.
* Against a moat:
  * [6] "We operate in the basic discount consumer goods market, which is highly competitive with respect to price, customers, store location, merchandise quality, assortment and presentation, service offerings, in-stock consistency, customer service, promotional activity, employees, and market share."
  * [7] "Certain of our competitors have greater financial, distribution, marketing and other resources than we do and may be able to secure better arrangements from suppliers than we can."
  * [8] "Competition is intense and we believe it will continue to be so, with certain competitors reducing their store locations while others move into or increase their presence in our geographic and product markets"

### MO: ALTRIA GROUP, INC. (Consumer Staples, 10-K filed 2025-02-26)

> General Development of Business When used in this Annual Report on Form 10-K (“Form 10-K”), the terms “the Company,” “we,” “us” and “our” refer to either (i) the Company and its consolidated subsidiaries or (ii) the Company only and not its consolidated subsidiaries, as appropriate in the context. ...

**Overall: narrow.** The moat rests on durable brands, with Marlboro at number one for 50 years, and on a regulatory environment that is hard for new entrants to navigate. The market is highly competitive and cigarette volumes are declining about 10% a year. Pricing power and brand loyalty should persist for a decade or more, but the shrinking category limits durability, so the moat is narrow.

* **intangibles = 2.** Marlboro has been the top-selling US cigarette brand for 50 years, and the company is the largest cigarette maker and the leading moist smokeless tobacco producer. Brand loyalty is stated as a basis of competition, and heavy regulation limits advertising and new entry. Volumes are declining 10% a year, which tempers durability.
  * [1] "PM USA is the largest cigarette company in the United States and substantially all cigarettes are manufactured and sold to customers in the United States."
  * [2] "has been the largest-selling cigarette brand"
  * [3] "USSTC is the leading producer and marketer of MST products."
  * [4] "The market for tobacco products is highly competitive, characterized by brand recognition and loyalty, with product quality, taste, price, product innovation, marketing, packaging and distribution constituting the significant methods of competition."
  * [5] "NJOY’s commercialized product portfolio of tobacco and menthol e-vapor products is fully covered by marketing granted orders (“MGO”) from the U.S. Food and Drug Administration (“FDA”)."
  * [6] "The Family Smoking Prevention and Tobacco Control Act (“FSPTCA”) provides the FDA with broad authority to regulate the design, manufacture, packaging, advertising, promotion, sale and distribution of tobacco products"
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Total smokeable products segment’s cigarettes shipment volume in the United States was 68.6 billion units in 2024, a decrease of 10.2% from 2023."
  * [8] "we recognize that the decreasing social acceptance of tobacco usage may impact our ability to attract and retain talent"
  * [9] "Middleton contracts with a third-party importer to supply substantially all of its cigars"
  * [10] "NJOY contracts with third-party importers to supply all of its products"

### TSN: TYSON FOODS, INC. (Consumer Staples, 10-K filed 2025-11-10)

> GENERAL the Company and its subsidiaries (collectively, the “Company,” “we,” “us,” “our,” “the Company” or “the Company”) (NYSE: the Company) is a world-class food company and recognized leader in protein. Founded in 1935 by John W. the Company, it has grown under four generations of family ...

**Overall: none.** The Company is a commodity protein processor with short-term customer agreements, price-based competition, volatile commodity inputs and a large customer (Walmart). Brands and the breeding genetics business offer some intangible strength, but the evidence is not enough for a durable moat.

* **intangibles = 1.** Brands like Jimmy Dean, Hillshire Farm and Ball Park exist, and Cobb-Vantress is a leading breeding stock supplier, but there is no evidence of pricing power or quantified share. Brand claims are generic.
  * [6] "Through our wholly-owned subsidiary, Cobb-Vantress, we are one of the leading poultry breeding stock suppliers in the world."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [1] "Our agreements with customers are generally short-term, primarily due to the nature of our products, industry practices and fluctuations in supply, demand and price for such products."
  * [2] "The principal competitive elements are advertising, consumer and trade promotions, price, product safety and quality, brand identification, innovation, breadth and depth of product offerings, availability of products, customer service and credit terms."
  * [3] "Walmart Inc. accounted for approximately"
  * [4] "The primary raw materials used in our prepared foods operations are commodity-based raw materials, including beef, pork, chicken, turkey, flour, vegetables, cheese, eggs, seasonings and other cooking ingredients."
  * [5] "buy live cattle on the open spot market"

### EOG: EOG RESOURCES INC (Energy, 10-K filed 2025-02-27)

> Business General the Company., a Delaware corporation organized in 1985, together with its subsidiaries (collectively, the Company), explores for, develops, produces and markets crude oil, natural gas liquids (NGLs) and natural gas primarily in major producing basins in the United States of America ...

**Overall: none.** The Company is a price-taking commodity producer with a sizeable, long-lived reserve base and a claimed low-cost position, but competes with larger rivals and sells at market prices. Any cost advantage is modest and not clearly durable, so no moat.

* **cost_advantage = 1.** The Company claims to be a low-cost producer with a large, stacked-pay Delaware Basin acreage position and owned infrastructure, but these are largely management aspirations without comparative figures. Acreage and reserves are a partial resource advantage and look modest to uncertain.
  * [1] "the Company is focused on being among the highest return and lowest cost producers, committed to strong environmental performance and playing a significant role in the long-term future of energy."
  * [2] "the Company implements its strategy primarily by emphasizing the drilling of internally generated prospects in order to find and develop low-cost reserves."
  * [3] "The Delaware Basin consists of approximately 4,800 feet of liquids-rich stacked pay potential offering the Company multiple co-development opportunities throughout its 395,000 net acre position."
  * [4] "In addition, key gathering, processing and transportation infrastructure was added in order to lower operating costs and increase price realizations."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [5] "In each case, the price received was based on market prices at that specific sales point or based on the price index applicable for that location."
  * [6] "Certain of the Company's competitors have financial and other resources substantially greater than those the Company possesses and have established strategic long-term positions or strong governmental relationships in countries or areas in which the Company may seek new or expanded entry."
  * [7] "Conversely, sales of crude oil and condensate and NGLs by the Company are made at unregulated market prices."
  * [8] "During 2024, three purchasers each accounted for more than 10% of the Company's total crude oil and condensate, NGLs and natural gas revenues and gathering, processing and marketing revenues."

### BBT: TRUIST FINANCIAL CORP (Financials, 10-K filed 2025-02-25)

> the Company is a purpose-driven financial services company committed to inspiring and building better lives and communities. Headquartered in Charlotte, North Carolina, the Company has leading market share in many of the high-growth markets in the U.S. and offers a wide range of products and ...

**Overall: none.** A large regional bank with strong deposit share in several states, but the text describes intense competition on pricing and fees, larger rivals and fintech entrants. It states no specific durable advantage, so no moat is evident from the text.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "The financial services industry is intensely competitive and constantly evolving."
  * [5] "Some also have greater scale, financial and operational resources, investment capacity, product and service offerings, and brand recognition."
  * [6] "Competition affects every aspect of our business, including product and service offerings, rates, pricing and fees, credit limits, and client service."
  * [7] "At the same time, non-banking entities, including financial technology companies, have increased competition by providing financial products and services directly to customers and indirectly through partnerships."
  * [8] "We expect that competition will only intensify in the future."

### COF: CAPITAL ONE FINANCIAL CORP (Financials, 10-K filed 2025-02-20)

> OVERVIEW General the Company, a Delaware corporation established in 1994 and headquartered in McLean, Virginia, is a diversified financial services holding company with banking and non-banking subsidiaries. the Company and its subsidiaries (the “Company” or “the Company One”) offer a broad array of ...

**Overall: none.** A large diversified bank and card issuer in highly competitive markets where customers choose on price, rewards and features. The text states no clear moat source, and regulation does not by itself confer excess returns. The loss of the Walmart program shows partner contracts can walk away.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "Each of our business segments operates in a highly competitive environment, and we face competition in all aspects of our business from numerous bank and non-bank providers of financial services."
  * [4] "In general, customers are attracted to credit card issuers largely on the basis of price, credit limit, reward programs, customer experience and other product features."
  * [5] "Our businesses generally compete on the basis of the quality and range of their products and services, transaction execution, innovation and price."
  * [6] "On May 21, 2024, our credit card program agreement with Walmart terminated (“Walmart Program Termination”)."

### EG: EVEREST GROUP, LTD. (Financials, 10-K filed 2025-02-27)

> The Company. the Company is a Bermuda-based reinsurance and insurance organization. As part of the Standard & Poor’s (“S&P”) 500 Index, we are a leading financial services institution focused on diversifying our portfolio and geographic presence. Through our direct and indirect subsidiaries ...

**Overall: none.** The Company has strong ratings, licensing breadth and long-standing relationships, but it operates in highly competitive, mature markets where competition is based on price and capacity. It also relies on a few large brokers. The advantage looks modest and not clearly durable.

* **intangibles = 1.** A+ financial strength ratings and broad licensing help the Company win business, but many large peers hold similar ratings. The company also describes a competitive market that differentiates on several factors. This is a modest advantage.
  * [1] "Strong financial ratings are particularly important for reinsurance and insurance companies given that customers rely on a company to pay covered losses well into the future. As a result, a highly rated company is generally preferred."
  * [2] "Group’s active operating subsidiaries are each rated A+ (“Superior”) by A.M. Best Company (“A.M. Best”)"
  * [3] "Such licensing enables U.S. domestic ceding company clients to take credit for uncollateralized reinsurance receivables from the Company Re in their statutory financial statements."
  * [4] "It is our long-standing client and broker relationships that help us continue to grow and maintain our global leadership position."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The global reinsurance market is highly competitive and mature."
  * [7] "Competition for clients might be based on pricing, capacity, coverage terms, conditions or other factors."
  * [8] "The global insurance market is highly competitive."
  * [9] "In addition, competitors might include investment companies, mutual companies, insurance companies, alternative risk providers (such as captives, catastrophe bonds and pools) and others"
  * [10] "The Reinsurance segment’s ten largest brokers accounted for an aggregate of approximately 60.5% of gross written premiums in 2024."

### MCO: MOODYS CORP /DE/ (Financials, 10-K filed 2025-02-14)

> Background As used in this report, except where the context indicates otherwise, the terms “Moody’s” or the “Company” refer to Moody’s Corporation, a Delaware corporation, and its subsidiaries. The Company’s executive offices are located at 7 World Trade Center at 250 Greenwich Street, New York, NY ...

**Overall: wide.** The ratings business has strong intangible advantages: regulatory registration, a long-established reputation and broad investor use. MA adds workflow embeddedness and retention. Against this, competitors exist, including other CRAs and in-house research, and revenue depends on debt issuance volume. The moat is durable but rests mainly on the ratings franchise.

* **intangibles = 2.** MIS holds an NRSRO registration and a 115-year reputation. Ratings are widely used by investors, which makes issuers want a Moody's rating. Regulatory registration and brand together act as barriers, but the text gives no market share figures.
  * [1] "MIS is registered with the SEC as an NRSRO and is subject to the SEC's oversight and examination authority."
  * [2] "Moody’s opinions on credit are used by institutional investors throughout the world, making an issuer’s debt potentially more attractive to a wide range of buyers."
  * [3] "Among ratings advisors, Moody’s has a strong position and is well-recognized for the depth and breadth of its analytical capabilities."
  * [4] "For more than 115 years, MIS has been a leading provider of credit ratings, research, and risk analysis helping businesses, governments, and other entities around the globe."
* **switching_costs = 1.** MA products are embedded in customer workflows and the company cites strong retention. MIS has annual fee arrangements and monitoring fees. The text gives no retention figures or contract lengths, so the evidence is moderate.
  * [5] "MA's integrated and technology-enabled solutions provide unique capabilities and insights that are embedded in customer workflows."
  * [6] "STRONG CUSTOMER RETENTION RATES"
  * [7] "However, annual fee arrangements with frequent debt issuers, annual fees from debt monitoring, commercial paper and medium-term note programs, bank deposit ratings, insurance company financial strength ratings, mutual fund ratings, and other areas partially mitigate MIS’s dependence on the volume or number of new debt securities issued in the global fixed-income markets."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [11] "MIS competes with other CRAs and with investment banks and brokerage firms that offer credit opinions and research. Many users of MIS’s ratings also have in-house credit research capabilities."
  * [12] "MA competes broadly in the financial information and enterprise risk software industries against various diversified competitors."
  * [13] "Rating fees paid by debt issuers account for most of the revenue of MIS."

### NTRS: NORTHERN TRUST CORP (Financials, 10-K filed 2025-02-24)

> the Company the Company (the “Corporation”) is a leading provider of wealth management, asset servicing, asset management and banking solutions to corporations, institutions, families and individuals. The Corporation is a financial holding company conducting business through various U.S. and ...

**Overall: none.** The Company has scale in custody and wealth management, but the filing states no mechanism that keeps competitors out. It describes intense competition from many rivals and offers only generic claims of differentiation and brand, so no moat can be supported from the text.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "the Company faces intense competition in all aspects and areas of its business."
  * [5] "Our competitors include a broad range of financial institutions and service companies, including other custodial banks, investment counseling firms, deposit-taking institutions, asset management firms, benefits consultants, trust companies, investment banking firms, insurance companies, and various financial technology companies, including software providers and data services firms."

### PYPL: PayPal Holdings, Inc. (Financials, 10-K filed 2025-02-04)

> OVERVIEW At the Company, our mission is to revolutionize commerce globally. Our products are designed to enable digital payments and simplify commerce experiences for consumers and merchants to make selling, shopping, and sending and receiving money simple, personalized, and secure, online or ...

**Overall: narrow.** A large two-sided network, trusted brands and global licenses give some advantage, but intense competition from banks, card networks, wallets and BNPL providers, low consumer switching costs and fee pressure limit durability. Narrow moat.

* **intangibles = 1.** Trusted brands (PayPal, Venmo) and regulatory licenses across many jurisdictions are stated as advantages, but the text gives limited evidence of pricing power and competition is intense.
  * [5] "we have built and strengthened well-recognized and trusted brands, including the Company, Venmo, and Braintree."
  * [4] "we believe that our regulatory licenses, which enable us to operate in markets around the world, are a distinct advantage and help support business growth."
  * [6] "Our Venmo digital wallet in the United States (“U.S.”) is a leading mobile application used to move money between our customers."
* **network_effects = 1.** A two-sided network of 434M accounts connecting consumers and merchants, with P2P driving user acquisition, directly states a network effect. Growth is slowing (accounts +2%) and competitors are many, so the advantage is real but not clearly durable.
  * [1] "We operate a global, two-sided network at scale that connects consumers and merchants with 434 million active accounts across approximately 200 markets as of December 31, 2024."
  * [2] "Our relationship on both sides of a transaction enables us to offer unique product experiences designed to remove friction, drive sales, and enhance shopping experiences."
  * [3] "P2P is an important source of customer engagement and also serves as a customer acquisition channel that facilitates organic growth by enabling potential users to establish active accounts with the Company or Venmo at the time they make or receive a P2P payment."
* Rated 0: switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "The global payments industry is highly competitive, dynamic, and innovative, and subject to regulatory scrutiny and oversight."
  * [8] "Many of the areas in which we compete evolve rapidly with innovative and disruptive technologies, shifting user preferences and needs, price sensitivity of consumers and merchants, and frequent introductions of new products and services."
  * [9] "Competition also may intensify as new competitors emerge, businesses combine or enter into new partnerships, and established companies in other segments expand to become competitive with various aspects of our business."
  * [11] "Interchange fees are being reviewed or challenged in various jurisdictions."

### SCHW: SCHWAB CHARLES CORP (Financials, 10-K filed 2025-02-26)

> General Corporate Overview The Charles the Company (CSC) is a savings and loan holding company. CSC engages, through its subsidiaries (collectively referred to as the Company or the Company), in wealth management, securities brokerage, banking, asset management, custody, and financial advisory ...

**Overall: narrow.** A scale-based cost advantage, a trusted brand and sticky custodial assets give a moat, but the field is crowded with banks, brokers and fintechs competing on price, so it is narrow rather than wide.

* **intangibles = 1.** Brand and reputation in a trust-dependent industry is claimed, but this is a general statement without evidence of pricing power.
  * [3] "In an industry dependent on trust, the Company reputation and brand across multiple constituents enable us to attract clients and employees while credibly introducing new products to the market."
* **switching_costs = 1.** Custody of $10 trillion in assets, RIA platform integration and archived records imply some stickiness, but the text does not state switching costs directly.
  * [4] "Providing bank, wealth, and asset management services to broker-dealer clients helps serve a wider array of needs, thereby deepening relationships, enhancing the stability of client assets, and enabling diversified revenue streams."
  * [5] "The site provides multi-year archiving of statements, trade confirms, and tax reports, along with document search capabilities."
* **cost_advantage = 2.** The Company states directly that scale and operating efficiency give it a cost advantage, as one of the largest firms spreading costs over a large client base. It is credible, though no figures are given.
  * [1] "As one of the largest investment services firms in the U.S., we are able to spread operating costs and amortize new investments over a large base of clients, and harness the resources to evolve capabilities to meet client needs."
  * [2] "Coupled with scale, our operating efficiency and sharing of infrastructure across different businesses creates a cost advantage that enables us to competitively price products and services while profitably serving clients of various sizes across multiple channels."
* Rated 0: network_effects, efficient_scale.
* Against a moat:
  * [7] "Within Investor Services, our competition in serving individual investors spans brokerage, wealth management, and asset management firms, as well as banks, trust companies, financial technology companies, and retirement service providers."
  * [8] "In the Advisor Services arena, we compete with institutional custodians, wirehouses, regional and independent broker-dealers, banks, and trust companies."

### ABBV: AbbVie Inc. (Health Care, 10-K filed 2025-02-14)

> Overview the Company or "the company" refer to the Company, or the Company and its consolidated subsidiaries, as the context requires. the Company is a global, diversified research-based biopharmaceutical company positioned for success with a comprehensive product portfolio that has leadership ...

**Overall: narrow.** The moat rests on patents, regulatory exclusivity, and strong brands such as Botox and Juvederm, but these are finite as shown by Humira biosimilar erosion, and government price-setting and competition threaten returns. The portfolio is diversified, with a long patent runway on Skyrizi and Rinvoq, supporting a narrow moat.

* **intangibles = 1.** Patents and regulatory exclusivity protect key products like Skyrizi and Rinvoq into 2033, and market-leading aesthetics brands plus biologics complexity add protection. However, patent cliffs (Humira biosimilars) and IRA price-setting limit durability.
  * [1] "The United States composition of matter patents covering risankizumab and upadacitinib are expected to expire in 2033."
  * [2] "These patents and applications, including various patents that expire during the period 2025 to the mid 2040s, in aggregate are believed to be of material importance in the operation of the Company business."
  * [3] "the Company Aesthetics portfolio consists of facial injectables, plastics and regenerative medicine, body contouring and skincare products, which hold market-leading positions in the U.S. and in key markets around the world."
  * [5] "The more complex nature of biologics and biosimilar products has led to close regulatory scrutiny over follow-on biosimilar products, which can reduce the effect of biosimilars on sales of the innovator biologic as compared to the sales erosion caused by generic versions of small molecule pharmaceutical products."
  * [6] "Significant investments in biologics infrastructure and manufacturing are necessary to produce biologic products."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "Humira faces direct biosimilar competition globally, and the Company will continue to face competitive pressure from these biologics and from orally administered products."
  * [9] "In addition, in the past few years, a number of other companies have started to develop, have successfully developed and/or are currently marketing products that are being positioned as competitors to Botox."
  * [10] "In August 2023, the U.S. Department of Health and Human Services (HHS), through the Centers for Medicare & Medicaid Services (the CMS), selected Imbruvica as one of the first 10 medicines subject to government-set prices beginning in 2026."
  * [11] "In 2024, three wholesale distributors (McKesson Corporation, Cardinal Health, Inc. and Cencora, Inc.) accounted for substantially all of the Company pharmaceutical product sales in the United States."
  * [12] "New products or treatments brought to market by the Company competitors could cause revenues for the Company products to decrease due to price reductions and sales volume decreases."

### DGX: QUEST DIAGNOSTICS INC (Health Care, 10-K filed 2025-02-20)

> ." Critical Accounting Policies The preparation of financial statements in conformity with accounting principles generally accepted in the United States requires us to make estimates and assumptions and select accounting policies that affect our reported financial results and the disclosure of ...

**Overall: none.** The excerpt covers only MD&A and financial statements and gives no direct evidence of a moat. Operating margin is about 13.6%, and the Company depends on payer fee schedules set by governments and insurers, which limits pricing power. Scale in lab testing may help, but the text does not state it, so there is no demonstrated moat.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "Reimbursements from domestic government payers are based on fee-for-service schedules set by governmental authorities, including traditional Medicare and Medicaid."
  * [5] "Under capitated arrangements with healthcare insurers, the Company recognizes revenue based on a predetermined monthly reimbursement rate for each member of an insurer's health plan regardless of the number or cost of services provided by the Company."
  * [6] "Our revenues are primarily comprised of a high volume of relatively low-dollar transactions, and about one-half of our total costs and expenses consist of employee compensation and benefits."

### GILD: GILEAD SCIENCES, INC. (Health Care, 10-K filed 2025-02-28)

> the Company (including its consolidated subsidiaries, referred to as “the Company,” the “company,” “we,” “our” or “us”) is a biopharmaceutical company that has pursued and achieved breakthroughs in medicine for more than three decades, with the goal of creating a healthier world for all people. We ...

**Overall: narrow.** The moat rests on patent and regulatory exclusivity for HIV and HCV drugs, which should last about a decade. Intense competition, generic entry, payer pricing pressure and wholesaler concentration limit it. Pipeline renewal is uncertain, so the moat is narrow.

* **intangibles = 1.** Patents and regulatory exclusivity on key HIV and HCV products run to 2033-2037, protecting the franchise for a decade or more. But the protection is finite and generic entry follows, with some settlements already granting licences. It is a real but time-limited advantage.
  * [1] "If we have a properly drafted and enforceable patent, it can be more difficult for our competitors to use our technology to create competitive products and more difficult for our competitors to obtain a patent that prevents us from using technology we create."
  * [2] "Biktarvy | 2033 | 2033"
  * [3] "Epclusa | 2033 | 2032"
  * [4] "Sunlenca | 2037 | 2037"
  * [5] "Ten years of regulatory/market exclusivity expected on approval."
  * [6] "Regulatory exclusivity in the U.S. expires in 2032."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [9] "We operate in a highly competitive environment."
  * [10] "(iv) new branded or generic products introduced into major markets, which may impact our ability to maintain pricing and market share."
  * [11] "During the year ended December 31, 2024, approximately 91% of our product sales in the U.S. and approximately 65% of our total worldwide revenues were from three large wholesalers: Cardinal Health, Inc., Cencora, Inc. and McKesson Corporation, and their specialty distributor affiliates."
  * [12] "As a result, the price increases we implement from time to time on certain products may have a limited effect on our net product sales in certain markets."
  * [13] "(iii) pricing pressures from private insurers and government payers as our products mature, which often result in a reduction of the net product prices;"

### HUM: HUMANA INC (Health Care, 10-K filed 2025-02-20)

> General Headquartered in Louisville, Kentucky, the Company. and its subsidiaries, referred to throughout this document as “we,” “us,” “our,” the “Company” or “the Company,” is committed to putting health first – for our teammates, our customers, and our company. Through our the Company insurance ...

**Overall: none.** The business is a highly competitive, government-dependent insurer with annual customer choice, heavy reliance on CMS and larger rivals. Provider discounts from scale are the only direct advantage and they are modest. No durable moat is evident in the text.

* **cost_advantage = 1.** Membership scale yields provider discounts, but competitors with larger membership exist, so the advantage looks modest.
  * [5] "Our membership base and the ability to influence where our members seek care generally enable us to obtain contractual discounts with providers."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "The health benefits industry is highly competitive."
  * [2] "Many of our competitors have a larger membership base and/or greater financial resources than our health plans in the markets in which we compete."
  * [3] "During 2024, 85% of our total premiums and services revenue were derived from contracts with the federal government, including 14% derived from our individual Medicare Advantage contracts in Florida with the Centers for Medicare and Medicaid Services, or CMS"
  * [4] "Generally, Medicare-eligible individuals enroll in one of our plan choices between October 15 and December 7 for coverage that begins on the following January 1."
  * [9] "Our individual Medicare Advantage products covered under Medicare Advantage contracts with CMS are renewed generally for a calendar year term unless CMS notifies us of its decision not to renew by May 1 of the calendar year in which the contract would end"

### IDXX: IDEXX LABORATORIES INC /DE (Health Care, 10-K filed 2025-02-21)

> COMPANY OVERVIEW the Company was incorporated in Delaware in 1983. We develop, manufacture, and distribute products and provide services primarily for the companion animal veterinary, livestock and poultry, dairy and water testing industries. We also provide human medical point-of-care and ...

**Overall: narrow.** The company has an installed base with recurring consumables, integrated software and regulatory approvals that support a narrow moat. Competition from Mars and Zoetis, the lack of material patents and the dependence on sole-source suppliers limit durability.

* **intangibles = 1.** EPA/FDA/USDA approvals and exclusive Catalyst slide supply create some barriers, mainly in water and dairy, but the company says no single patent is material and faces robust competition.
  * [4] "The agreements also prohibit Ortho from promoting and selling these chemistry slides in the veterinary sector, excluding the EU, other than to the Company."
  * [5] "Our Tecta Systems, Colilert, Colilert-18, Colisure, Quanti-Tray, Filta-Max xpress, Enterolert, and SimPlate for heterotrophic plate counts products have been approved by the EPA for use under various regulatory programs."
  * [6] "All test methods used for drug residue testing require validation by the FDA and acceptance by the NCIMS."
* **switching_costs = 1.** A large installed base of analyzers drives recurring proprietary consumables, and integrated IVLS and practice management software tie clinics in, though no explicit contract lock-in or retention figures are given.
  * [1] "Sales of consumables to customers who use our chemistry analyzers provide the majority of our instrument consumables revenues from our installed base of the Company VetLab instruments."
  * [2] "Our large practice management systems installed base provides access to veterinary channel transaction activity, enabling a syndicated data offering."
  * [3] "The the Company VetLab Station (“IVLS”) connects and integrates the diagnostic information from all the the Company VetLab analyzers, and thus provides reference laboratory information management system capability."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "In addition, we face robust competition as other companies have been successful in bringing competitive products and services to market, despite the protections afforded by these patent and license rights."
  * [9] "Several of our direct and potential competitors have substantially greater financial and managerial resources than us, as well as greater experience in manufacturing, marketing, research and development, and obtaining regulatory approvals than we do."
  * [10] "we do not believe that any single patent, copyright, trademark, trade secret, or license is material to our business as a whole, except for licenses related to our Catalyst instruments and consumables"
  * [11] "Our major competitors in most geographic locations in North America are Mars, Incorporated brands Antech Diagnostics and Heska; and Zoetis Inc. (including its wholly-owned subsidiary Abaxis, Inc.)."
  * [12] "In some cases, these third parties are sole or single-source suppliers."

### MOH: MOLINA HEALTHCARE, INC. (Health Care, 10-K filed 2025-02-11)

> OVERVIEW ABOUT the Company the Company, a FORTUNE 500 company, provides managed healthcare services under the Medicaid and Medicare programs, and through the state insurance marketplaces (the “Marketplace”). the Company was founded in 1980 as a provider organization serving low-income families in ...

**Overall: none.** Moat sources are modest: state licensing and procurement barriers and a strong RFP record. Against that, contracts are short, can be terminated, and are re-bid against large national competitors. Medicare is highly competitive, the company depends on a few states, and its medical cost ratio is high. These add up to no durable moat.

* **intangibles = 1.** State licensing and procurement processes and a high RFP win rate suggest some regulatory-based advantage, but contracts are re-bid competitively and the company states only 'partial' barriers to entry.
  * [1] "However, the licensing requirements and bidding and contracting procedures in some states may present partial barriers to entry into our industry."
  * [2] "We have achieved an 80% new contract win rate, and a 90% re-procurement win rate for Medicaid requests for proposal (“RFPs”) during this timeframe."
* **efficient_scale = 1.** Licensing and bidding requirements give partial barriers, and states award a limited number of contracts, but the market has many large competitors and is contested.
  * [1] "However, the licensing requirements and bidding and contracting procedures in some states may present partial barriers to entry into our industry."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [3] "Such contracts are subject to risk of loss in states that issue RFPs open to competitive bidding by other health plans."
  * [4] "Our state Medicaid contracts typically have terms of three to five years, contain renewal options exercisable by the state Medicaid agency, and allow either the state or the health plan to terminate the contract with or without cause."
  * [5] "There is increasing competition driven by renewed interest from large national health plans."
  * [6] "The Medicare market is highly competitive across the country, with large competitors, such as CVS Health Corporation, Humana Inc., and UnitedHealth Group Inc."
  * [7] "Competition can vary considerably from state to state, and some of our competitors have larger membership bases and/or greater financial resources than our health plans in the markets in which we compete."
  * [9] "These contracts have a one-year term ending on December 31, and must be renewed annually."
  * [10] "Our Medicaid contracts with the states of California, New York, Texas, and Washington each accounted for approximately 10% or more of our consolidated Medicaid premium revenues in the year ended December 31, 2024."

### REGN: REGENERON PHARMACEUTICALS, INC. (Health Care, 10-K filed 2025-02-05)

> This Annual Report on Form 10-K contains forward-looking statements that involve risks and uncertainties relating to future events and the future performance of the Company (where applicable, together with its subsidiaries, "the Company," "Company," "we," "us," and "our"), and actual events or ...

**Overall: narrow.** The business rests on patent-protected biologics and an innovation engine, which is a narrow, time-limited advantage. EYLEA already faces biosimilar and branded competition, and Dupixent faces many rival drugs. The company has no switching costs, network effects or cost advantage and sells to concentrated wholesalers. Patents and regulatory exclusivity are the only support for a moat, and they point to narrow at best.

* **intangibles = 1.** Patents on key products (Dupixent composition of matter to 2031, formulation and method patents into the late 2030s), regulatory exclusivity and FDA-approved biologics manufacturing give meaningful protection. Patents expire within roughly a decade for several key products, EYLEA already faces biosimilar competition, and the filing itself admits patents may not protect against competition.
  * [2] "US 7,608,693 Composition of Matter March 28, 2031"
  * [3] "In addition to our patent portfolio, in the United States and certain other countries, our competitive position may be enhanced due to the availability of market exclusivity under relevant law"
  * [4] "Each of these facilities is approved by the FDA and certain other regulatory agencies to manufacture our bulk drug materials and products."
  * [1] "Our remaining issued patents covering these technologies will expire between 2025 and 2032."
* Rated 0: switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [8] "There is no certainty that our existing patents or others, if obtained, will provide us protection from competition or provide commercial benefit."
  * [9] "Others may independently develop similar products or processes to those developed by us, duplicate any of our products or processes or, if patents are issued to us, design around any products and processes covered by our patents."
  * [10] "Competition among products approved for sale is based on efficacy, safety, reliability, ease of administration, dosing frequency, availability, price, patent and other intellectual property position, and other factors."
  * [11] "On a combined basis, our product sales to these customers accounted for 74% of our total gross product revenue for the year ended December 31, 2024."
  * [12] "we may be at a substantial competitive disadvantage in such areas as a result of, among other things, our inferior intellectual property position or lack of experience, trained personnel, and expertise."
  * [13] "Pavblu® (aflibercept-ayyh) (biosimilar referencing EYLEA)"
  * [14] "These payors may not cover or adequately reimburse for use of our products or may do so at levels that disadvantage them relative to competitive products."

### ALLE: Allegion plc (Industrials, 10-K filed 2025-02-18)

> BUSINESS Overview the Company ("the Company," "we," "us" or "the Company") is a leading global provider of security products and solutions that keep people and assets safe and secure in the places they live, learn, work and connect. We pioneer safety and security to create a safer and more ...

**Overall: narrow.** Strong legacy brands, specification-driven selling and product integration give a modest edge in a fragmented, competitive market. Competitors include Assa Abloy and dormakaba, and the company competes partly on price. The moat looks narrow at best.

* **intangibles = 1.** Century-old brands like Schlage and Von Duprin, plus specification relationships with architects, give some pull, but the text offers no pricing power evidence and the company says it is not dependent on any patents.
  * [1] "We consider our CISA, Interflex, LCN, Schlage, SimonsVoss, Von Duprin and other associated trademarks to be among our most valuable assets, and we have registered these trademarks in a number of countries."
  * [2] "Our leading brands include CISA®, Interflex®, LCN®, Schlage®, SimonsVoss®, and Von Duprin®."
  * [5] "Our field sales professionals are assisted by specification writers who work with architects, engineers and consultants to help design door openings and security systems to meet end-users’ functional, aesthetic and regulatory requirements."
* **switching_costs = 1.** Doors must fit locks, frames and keys, and access control is integrated with software, with a service base for automatic doors. This suggests modest lock-in, though it is not explicitly stated as switching costs.
  * [3] "Each door must fit exactly within its frame, be prepared precisely for its hinges, synchronize with its specific lockset and corresponding latch and align with a specific key to secure the door."
  * [4] "Our Stanley Access Technologies business offers extensive planned inspection, maintenance and repair services for its automatic entrance solutions throughout the U.S. and Canada."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The security products markets are highly competitive and fragmented throughout the world, with a number of large multi-national companies and thousands of smaller regional and local companies."
  * [7] "Although certain proprietary intellectual property rights are important to our success, we do not believe we are materially dependent on any particular patent or license, or any particular group of patents or licenses."
  * [8] "We compete based on the breadth, innovation and quality of our products and solutions, pricing, our ability to custom-configure solutions to meet individual end-user requirements and our global supply chain."
  * [9] "Our 10 largest customers represented approximately 27% of our total Net revenues in 2024."

### AME: AMETEK INC/ (Industrials, 10-K filed 2025-02-20)

> General Development of Business the Company (“the Company” or the “Company”) is incorporated in Delaware. Its predecessor was originally incorporated in Delaware in 1930 under the name American Machine and Metals, Inc. the Company is a leading global manufacturer of electronic instruments and ...

**Overall: none.** The company is a diversified instrument and component maker with niche positions and aerospace and medical qualification stickiness. Competition is described as highly competitive, with numerous rivals and price competition. Patents are not essential, and no mechanism is clearly stated or quantified. Customer diversification is a plus, but the evidence does not support a moat.

* **intangibles = 1.** Claims of niche market share and technological advantages are general, with no figures, and the company says no single patent or trademark is essential. The 70-year aerospace reputation is a modest brand-like asset.
  * [1] "the Company maintains significant market share in a number of targeted niche markets through its ability to produce and deliver high-quality, differentiated products at competitive prices."
  * [2] "the Company operates in highly specialized aerospace market segments in which it has proven technological or manufacturing advantages versus its competition."
* **switching_costs = 1.** Aerospace products are designed to customer specifications and the company has long-standing OEM relationships and an aftermarket business. This suggests some qualification-related stickiness, but the text does not state switching costs directly.
  * [3] "the Company has long-standing relationships with the world’s"
  * [4] "the Company aerospace products are designed to customer specifications and manufactured to stringent operational and reliability requirements."
  * [5] "the Company also is a leading provider of spare part sales, repairs and overhaul services to commercial aerospace."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "In general, the Company markets are highly competitive with competition based on technology, performance, quality, service and price."
  * [8] "In process and analytical instruments, numerous companies compete in each market on the basis of product quality, performance and innovation."
  * [9] "the Company, however, does not consider any single patent or trademark, or any group of them, essential either to its business as a whole or to either one of its reportable segments."
  * [10] "EMG’s businesses compete with a number of companies in each of its markets."
  * [11] "There also is competition from alternative materials and processes."

### CHRW: C. H. ROBINSON WORLDWIDE, INC. (Industrials, 10-K filed 2025-02-14)

> Overview the Company (“the Company,” “the company,” “we,” “us,” or “our”) is one of the largest global logistics providers in the world, with consolidated total revenues of $17.7 billion in 2024. We deliver logistics like no one else. Companies around the world look to us to reimagine supply ...

**Overall: none.** This is a large freight broker with scale and data advantages. The industry is fragmented and competitive, contracts are short with no volume commitments, and customers can switch easily. Any advantages are modest.

* **network_effects = 1.** The platform connects many shippers and carriers, which gives some matching benefit, but the text does not describe it as a reinforcing network effect, and the market is fragmented.
  * [6] "As one of the world’s largest logistics platforms, our proprietary technology connects 83,000 customers and 450,000 carriers."
* **cost_advantage = 1.** Scale and one of the largest freight datasets give purchasing and pricing advantages. They are stated generally, without cost comparisons, in a fragmented industry.
  * [7] "Our customers leverage our significant capacity, broad procurement options, global data insights, and substantial shipment volumes for better efficiency, service, and marketplace advantages;"
  * [10] "We have one of the largest datasets of shipments, routings, and carriers in the world."
* Rated 0: intangibles, switching_costs, efficient_scale.
* Against a moat:
  * [1] "The transportation services industry is highly competitive and fragmented."
  * [2] "We often compete with respect to price, scope of services, or a combination thereof, but believe that our most significant competitive advantages are:"
  * [3] "Most of our contractual rate commitments are for one year or less and allow for renegotiation."
  * [4] "As is typical in the transportation industry, most of these contracts do not include specific volume commitments."
  * [5] "Our standard contracts do not include volume commitments, and typically, the initial contract rate is modified each time we confirm an individual shipment with a contracted motor carrier."
  * [11] "During 2024, our largest customer accounted for approximately two percent of our consolidated total revenues."

### DE: DEERE & CO (Industrials, 10-K filed 2025-12-18)

> BUSINESS. This Annual Report on Form 10-K contains forward-looking statements that are subject to risks and uncertainties. All statements other than statements of historical fact included in this Annual Report on Form 10-K are forward-looking statements. Forward-looking statements provide our ...

**Overall: narrow.** The Company has a recognized brand, a big dealer network and proprietary parts and technology for its installed base, which support durable returns. The text offers little quantified evidence, and competition is intensifying with many global, regional and new entrants. This points to a narrow moat at best.

* **intangibles = 1.** Brand recognition and a large dealer network are cited as competitive factors, but there are no figures on price premium or share, and patents are not individually critical.
  * [1] "John the Company brand recognition is a competitive factor in North America and many other parts of the world."
  * [4] "Through the U.S. and Canada, we market products to approximately 2,050 independent dealer locations."
* **switching_costs = 1.** Proprietary parts for the installed base and integrated telematics give some lock-in, but nothing shows customers are unable to switch to competitors.
  * [2] "We supply parts, many of which are proprietary, to support the current product lines as well as products we have sold in the past."
  * [3] "Our advanced telematics systems remotely connect equipment owners, business managers, and dealers to equipment in the field through the John the Company Operations"
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The agricultural equipment industry continues to change and is becoming even more competitive through the emergence and global expansion of many competitors."
  * [7] "As technology increasingly enables enhanced productivity in agriculture, the industry is also attracting non-traditional competitors, including technology-focused companies and start-up ventures."
  * [8] "We believe that our rights under these patents and licenses are important to our operations and competitive position but do not regard any of our businesses as being dependent upon any single patent or family of patents."
  * [9] "The principal competitive factors in all markets include product performance, technology features, innovation, quality, distribution, sustainability, financing, customer service, and value."

### LII: LENNOX INTERNATIONAL INC (Industrials, 10-K filed 2025-02-11)

> Business References in this Annual Report on Form 10-K to “we,” “our,” “us,” “the Company,” “the Company” or the “Company” refer to the Company. and its subsidiaries, unless the context requires otherwise. The Company We are a global leader in energy-efficient climate-control solutions. We design, ...

**Overall: none.** The Company is one of a handful of large North American HVAC players with a company-owned distribution network, but the text offers no stated moat mechanism. Competition is intense, with rivals such as Carrier, Trane, Daikin and Johnson Controls, and competition runs on availability, price and performance. Patents are not material. The business section gives no direct evidence of a durable advantage.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [4] "Substantially all markets in which we participate are competitive."
  * [5] "The most significant competitive factors we face are product availability, reliability, energy efficiency, product performance, service, and price, with the relative importance of these factors varying among customer applications."
  * [6] "We hold numerous patents that relate to the design and use of our products. We consider these patents important, but no single patent is material to the overall conduct of our business."

### PWR: QUANTA SERVICES, INC. (Industrials, 10-K filed 2025-02-20)

> Business OVERVIEW the Company (together with its subsidiaries, “the Company,” “we,” “us” or “our”) is a leading provider of comprehensive infrastructure solutions for the electric and gas utility, renewable energy, technology, communications, pipeline and energy industries in the United States, ...

**Overall: none.** This is a labor-intensive contractor in a highly competitive market with low barriers to entry in places and price-based bidding. Preferred-provider status and long-term agreements give modest stickiness, but there is no strong, durable advantage.

* **switching_costs = 1.** Long-term maintenance agreements, preferred provider status and preferential bidding give some stickiness, but customers can rebid and price matters, so the advantage is modest.
  * [1] "Furthermore, many of our strategic relationships with customers take the form of strategic alliance or long-term maintenance agreements, which typically extend for an initial term and may include renewal options to extend the initial term."
  * [2] "Strategic alliance agreements also generally state an intention to work together over a period of time and/or on specific types of projects, and many provide us with preferential bidding procedures."
  * [3] "We strive to maintain our preferred status as we believe it provides us an advantage in the award of future work for the applicable customer."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "In addition, there are relatively few barriers to entry into some of the industries in which we operate and, as a result, organizations that have adequate financial resources and access to technical expertise may become a competitor."
  * [7] "A significant portion of our revenues is currently derived from unit price or fixed price agreements, and price is often an important factor in the award of such agreements."
  * [8] "Furthermore, companies that we engage as subcontractors, including pursuant to certain regulatory and customer requirements, may develop the expertise necessary to compete with us with respect to the award of prime contracts from our customers."
  * [9] "The in-house service organizations of our existing or prospective customers employ personnel who perform some of the same types of services we provide."
  * [10] "The industries and geographic markets in which we operate are highly competitive, and several of our competitors are large companies that have significant financial, technical and marketing resources."

### RSG: REPUBLIC SERVICES, INC. (Industrials, 10-K filed 2025-02-14)

> BUSINESS Overview Republic is one of the largest providers of environmental services in the United States, as measured by revenue. We operate across the United States and Canada through 367 collection operations, 248 transfer stations, 75 recycling centers, 208 active landfills, 2 treatment, ...

**Overall: narrow.** Vertically integrated local density, scarce permitted landfill capacity and long-term municipal contracts give the business a durable but not overwhelming position. The text does not state the barriers explicitly, and competition for acquisitions and private rivals remain. Narrow moat.

* **switching_costs = 1.** Long-term municipal contracts and exclusive franchises lock in revenue for periods, but they are generally re-bid and customers can switch at renewal, so the advantage is modest.
  * [2] "We seek to obtain long-term contracts for collecting recyclable, solid waste and industrial waste material under residential collection contracts with municipalities, exclusive franchise agreements, small-container and large-container contracts and environmental solutions service contracts."
* **cost_advantage = 1.** Route density, vertical integration and scale synergies are cited, but without quantified comparisons with competitors. The advantage looks local and modest.
  * [4] "We realize synergies from consolidating businesses into our existing operations, whether through acquisitions or public-private partnerships, which allows us to reduce capital expenditures and expenses associated with truck routing, personnel, fleet maintenance, inventories and back-office administration."
  * [5] "This approach of developing standardized processes with rigorous controls and tracking allows us to leverage our scale and deliver durable operational excellence."
* **efficient_scale = 1.** Landfills and disposal sites are permit-constrained, with permitting taking years and local density mattering, which supports local oligopoly-like economics. The text states this only indirectly, and the market is somewhat fragmented.
  * [1] "Our goal is to develop the best vertically integrated market position to enable us to build density and improve returns."
  * [3] "Development projects, while generally less capital intensive than acquisitions, typically require extensive permitting efforts that can take years to complete with no assurance of success."
* Rated 0: intangibles, network_effects.
* Against a moat:
  * [8] "We believe our ability to successfully complete these acquisitions is enhanced by the challenges facing many privately-held companies, including increasing competition in the environmental services industry"
  * [9] "Environmental solutions remains fragmented, which provides consolidation opportunities to drive scale."

### UAL: United Airlines Holdings, Inc. (Industrials, 10-K filed 2025-02-27)

> Overview the Company (together with its consolidated subsidiaries, "the Company" or the "Company") is a holding company and its wholly-owned subsidiary is the Company (together with its consolidated subsidiaries, "United"). United's shared purpose is "Connecting People. Uniting the World." United ...

**Overall: none.** Airline industry is highly competitive with price matching and free entry. The largest network and hub positions help somewhat but do not show durable excess returns; no moat.

* **network_effects = 1.** Hub-and-spoke network and Star Alliance give network benefits for connectivity, but these are scale/density advantages rather than true user-based network effects, and rivals have similar networks.
  * [2] "The hub and spoke system allows us to transport passengers between a large number of destinations with substantially more frequent service than if each route were served directly."
  * [3] "United is a member of Star Alliance, a global integrated airline network and the largest and most comprehensive airline alliance in the world."
* Rated 0: intangibles, switching_costs, cost_advantage, efficient_scale.
* Against a moat:
  * [6] "The domestic airline industry is highly competitive and dynamic."
  * [7] "Currently, any U.S. carrier deemed fit by the U.S. Department of Transportation (the "DOT") is largely free to operate scheduled passenger service between any two points within the United States."
  * [8] "In order to remain competitive and maintain passenger traffic levels, we often find it necessary to match competitors' discounted fares."
  * [9] "International competition has increased and may continue to increase in the future as a result of airline mergers and acquisitions, JBAs, alliances, restructurings, liberalization of aviation bilateral agreements and new or increased service by competitors."
  * [10] "Carriers with lower costs may offer lower fares to passengers, which could have a potential negative impact on the Company's revenues."

### CDW: CDW Corp (Information Technology, 10-K filed 2025-02-21)

> Our Company the Company (together with its subsidiaries, the “Company,” “the Company” or “we”), a Fortune 500 company and member of the S&P 500 Index, is a leading multi-brand provider of information technology (“IT”) solutions to business, government, education and healthcare customers in the ...

**Overall: none.** A large IT reseller in a fragmented, highly competitive market with available alternative supply and heavy dependence on a few vendors. Scale and vendor certifications give a modest edge, but there is no durable moat.

* **cost_advantage = 1.** The Company claims scale and favorable vendor pricing via certifications, but no figures compare its costs with competitors, and the advantage looks modest in a fragmented, competitive market.
  * [6] "These certifications also provide us with access to favorable pricing, tools and resources, including vendor incentive programs, which we use to provide additional value to our customers."
  * [7] "We believe the combination of our competitive advantages of scale, performance-driven culture and enhanced capabilities will help drive sustainable, profitable growth for us today and in the future."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "These are highly fragmented markets served by thousands of IT resellers and solutions providers."
  * [2] "The market for technology products, solutions and services is highly competitive and subject to economic conditions and rapid technological changes."
  * [3] "We believe competitive sources of supply are available in substantially all of the product categories that we offer."
  * [4] "While innovation can help our business as it creates new offerings for us to sell, it can also disrupt our business model and create new and stronger competitors."
  * [5] "In 2024, we generated $2.0 billion of Net sales from each of our three largest vendor partners."
  * [9] "These programs are at the discretion of our vendor partners and are typically tied to sales or other commitments to be met by us within a specified period."

### DELL: Dell Technologies Inc. (Information Technology, 10-K filed 2025-03-25)

> Company Overview the Company is a leader in the global technology industry focused on providing broad and innovative technology solutions for the data and artificial intelligence (“AI”) era. These solutions range from client devices and peripherals to infrastructure solutions across servers, ...

**Overall: none.** The company has scale in supply chain and distribution, but it competes in hardware markets with intense price competition, new entrants and hyperscalers buying directly from original design manufacturers. Switching costs and intangibles are weak, so no durable moat is evident.

* **cost_advantage = 1.** The company claims a large-scale supply chain and a direct sales model, which suggests some scale benefits. There are no figures or comparisons against competitors, and the company competes on price.
  * [4] "Our world-class supply chain operates at a significant scale with the ability to remain agile in a variety of environments."
  * [5] "We believe our direct sales channel is a significant competitive advantage and emphasizes direct communication with customers, allowing us to refine our products and marketing programs while providing insight to better navigate through supply chain challenges and complexity."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "We face ongoing product and price competition in all areas of our business from both branded and generic competitors, including companies that specialize in one or more of our product or service lines."
  * [2] "We also face competition from non-traditional IT companies, including large Infrastructure-as-a-Service providers, that often buy their infrastructure directly from original design manufacturers."
  * [3] "We believe that new businesses will continue to enter these markets and develop technologies that, if successfully commercialized, may compete with our products and services."
  * [8] "Although we use our patented inventions and license some of them to others, we are not substantially dependent on any single patent or group of related patents."
  * [9] "In some cases, where multiple sources of supply are not available, we rely on a single source or a limited number of sources of supply if we believe it is advantageous to do so because of performance, quality, support, delivery, capacity, or price considerations."

### FICO: FAIR ISAAC CORP (Information Technology, 10-K filed 2025-11-07)

> Business GENERAL the Company (NYSE: the Company) (together with its consolidated subsidiaries, the “Company,” which may also be referred to in this report as “we,” “us,” “our,” and “the Company”) is a global analytics software leader. We were founded in 1956 on the premise that data, used ...

**Overall: narrow.** The Score's status as the entrenched standard in U.S. credit decisions, embedded in lender models and GSE validation, is a durable intangible advantage, supported by switching costs in software and a data network in fraud. Offsetting this are bureau concentration (51% of revenue), the VantageScore competitor, and intense competition in software. This points to a narrow moat.

* **intangibles = 2.** The Score is the entrenched U.S. standard used in most credit decisions and validated by the GSEs, a strong brand/standard position. Competition from VantageScore and bureau concentration add some uncertainty.
  * [1] "the standard measure of consumer credit risk in the U.S. It is used in most U.S. credit decisions, by nearly all major banks, credit card issuers, mortgage lenders, and auto loan originators."
  * [2] "We also serve consumers through online services that enable people to access and understand their the Company Scores — the standard measure of consumer credit risk in the United States"
  * [7] "End users of our products include three-quarters of each of the largest 100 financial institutions in the U.S. and the largest 100 banks in the world."
  * [8] "including the ongoing validation and approval of the use of the the Company® Score by Fannie Mae, Freddie Mac, and the Federal Housing Finance Agency."
* **switching_costs = 1.** Multi-year subscriptions with contracted minimums, mission-critical embedded systems, and scores built into lenders' underwriting models create meaningful switching costs.
  * [3] "We typically sell our software as multi-year subscriptions, with payments based on usage metrics such as the number of accounts, transactions or decisioning use cases deployed, often subject to contracted minimum payments."
  * [4] "Our predictive models are frequently used in mission-critical transactional systems and drive decisions and actions in near real time."
  * [5] "Both scores maintain the same score to risk relationship as standard the Company® Scores, enhancing their compatibility with existing credit underwriting systems and models."
* **network_effects = 1.** The Falcon consortium data from 10,000+ institutions improves fraud models, a data network benefit, but it is limited to one product area.
  * [6] "Our models are continually improved using a proprietary, global data set of transaction data contributed by more than 10,000 institutions that participate in the the Company® Falcon® Intelligence Network."
* Rated 0: cost_advantage, efficient_scale.
* Against a moat:
  * [9] "The market for our solutions is intensely competitive and is constantly changing."
  * [10] "However, many of our competitors are larger than the Company, have more development, sales and marketing resources than the Company, and some have larger shares of our target geographic or product markets."
  * [11] "During fiscal 2025, 2024 and 2023, revenues generated from our agreements with Experian, TransUnion and Equifax collectively accounted for 51%, 45% and 41% of our total revenues, respectively."
  * [12] "VantageScore (a joint venture entity established by the three major U.S. consumer reporting agencies), which is selling a credit scoring product competitive with our products."
  * [13] "The patents we hold may not be upheld as valid and may not prevent the development of competitive products."

### NTAP: NetApp, Inc. (Information Technology, 10-K filed 2025-06-09)

> Business Overview the Company (the Company, we, us, or the Company) helps customers make their data infrastructure more seamless, more dynamic, and higher performing. We were incorporated in 1992, are headquartered in San Jose, California, and provide a full range of enterprise-class software, ...

**Overall: none.** The Company has a sticky ONTAP installed base and a distinctive cloud-embedded position. The filing, though, describes intense price competition, no long-term purchase commitments, a fragmented market, concentrated distributors and shared suppliers. Evidence of durable advantage is thin and moderate, so no moat is warranted.

* **intangibles = 1.** The exclusive-style position as the only enterprise-grade storage natively embedded in the major clouds is a differentiator. However, the company says it is not dependent on any individual IP right, and the position could be replicated or eroded by cloud providers.
  * [1] "As the only enterprise-grade storage service natively embedded in the world’s largest clouds, we power data across Amazon AWS, Microsoft Azure, and Google Cloud."
* **switching_costs = 1.** ONTAP is a foundational platform across on-premises and cloud, and the installed base is likely sticky. The text does not directly state switching costs, and purchases are cancelable without long-term commitments.
  * [2] "Customers can choose where to deploy primary workloads without re-architecting applications or databases."
  * [3] "the Company ONTAP software is our foundational technology that underpins the Company critical storage solutions in the on-premises data center and in private and public clouds."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "We typically do not enter into binding long-term purchase commitments with our customers, resellers, and distributors, meaning there is no guarantee that we will continue to receive large, recurring orders from them."
  * [6] "A substantial portion of our products is sold on the basis of standard purchase orders that are cancelable prior to shipment without penalty."
  * [7] "We face ongoing product and price competition in all areas of our business, including from both branded- and generic-product competitors."
  * [8] "Our markets are highly competitive, fragmented, and characterized by rapidly changing technology."
  * [9] "Sales to customers Arrow Electronics, Inc. and TD Synnex Corporation accounted for 21% and 24% of our net revenues, respectively, in fiscal 2025."
  * [10] "While our intellectual property rights are important to our success, we believe that our business is not materially dependent on any particular patent, trademark, copyright, license or other individual intellectual property right."
  * [11] "When industry supply is constrained or the supply chain is disrupted, our suppliers may allocate volumes away from us and to our competitors, who depend on many of the same suppliers as we do."

### ORCL: ORACLE CORP (Information Technology, 10-K filed 2025-06-18)

> Business the Company provides products and services that address enterprise information technology (IT) needs. Our products and services include enterprise applications and infrastructure offerings that are delivered worldwide through a variety of flexible and interoperable IT deployment models. ...

**Overall: narrow.** Embedded databases, ERP and applications with near-universal support renewal create strong switching costs and a durable installed base. Intense competition, low barriers in many segments, and open-source and hyperscaler alternatives limit the moat, so it is narrow.

* **intangibles = 1.** Database and Java are described as the most popular in the world, giving brand and standard-setting strength, but there are no patents or exclusive rights and no figures.
  * [5] "the Company Database is the world’s most popular enterprise database and is designed to enable reliable and secure storage, retrieval and manipulation of all forms of data."
  * [6] "Java is the world’s most popular programming language among professional developers"
* **switching_costs = 2.** Nearly all license customers buy and renew support, and customers run core operations on the Company's database and applications, which makes migration costly. The text gives no figures on retention or contract length beyond annual terms.
  * [1] "Substantially all license support customers renew their support contracts with us upon expiration in order to continue to benefit from technical support services and the periodic issuance of unspecified updates and enhancements, which current license support customers are entitled to receive."
  * [2] "Substantially all customers opt to purchase license support contracts when they purchase an the Company license."
  * [4] "Using the Company technologies, our customers build, deploy, run, manage and support their internal and external products, services and business operations"
  * [8] "We believe the Company Fusion Cloud ERP is a strategic suite of applications that is foundational to facilitating and extracting more business value out of the adoption of other OCA offerings"
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [9] "In addition, due to the low barriers to entry in many of our market segments, new technologies and new and growing competitors frequently emerge to challenge our offerings."
  * [10] "Our enterprise cloud, license and hardware offerings compete directly with certain offerings from some of the largest and most competitive companies in the world"
  * [11] "Our customers are demanding less complexity and lower total cost in the implementation, sourcing, integration and ongoing maintenance of their IT environments."
  * [12] "However, although we believe that our patents have value, neither our business as a whole nor any of our principal businesses are materially dependent on a single patent."
  * [13] "the adoption of open source alternatives to commercial software by enterprise software customers;"

### PANW: Palo Alto Networks Inc (Information Technology, 10-K filed 2025-08-29)

> General the Company. is a global cybersecurity provider and our vision is a world where each day is safer and more secure than the one before. We were incorporated in 2005 and are headquartered in Santa Clara, California. Our mission is to be the cybersecurity partner of choice for enterprises, ...

**Overall: narrow.** The company has a sizable installed base with renewing subscriptions and an integrated platform that create some stickiness, but the industry is intensely competitive, rivals are larger or equally well resourced, and the filing offers no direct evidence of a durable structural advantage. Sales also depend on a few distributors.

* **switching_costs = 1.** Renewing support contracts, a common PAN-OS operating system across the product line, and an installed base using Panorama management suggest moderate stickiness, but the company pushes consolidation and switching is not explicitly described as costly.
  * [1] "End-customers typically purchase these services for a one-year or longer term at the time of the initial product sale and typically renew for successive one-year or longer periods."
  * [2] "All of our hardware and software firewalls incorporate the PAN-OS® operating system and include the same rich set of features, ensuring consistent operation across our entire product line."
  * [3] "Many of our existing deployments continue to use Panorama as the security management solution."
* Rated 0: intangibles, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [5] "We operate in the intensely competitive enterprise security industry that is characterized by constant change and innovation."
  * [6] "However, some of our competitors may have substantially greater financial, technical, and other resources, greater name recognition, larger sales and marketing budgets, broader distribution, more diversified product lines, and larger and more mature intellectual property portfolios."
  * [7] "For fiscal 2025, 44.2% of our total revenue was derived from sales to three distributors."
  * [8] "As our market grows, it will attract more highly specialized vendors, as well as larger vendors that may continue to acquire or bundle their products more effectively."
  * [9] "We believe that competitors will try to develop products that are similar to ours and that may infringe our intellectual property rights."

### VRSN: VERISIGN INC/CA (Information Technology, 10-K filed 2025-02-13)

> BUSINESS Overview We are a global provider of critical internet infrastructure and domain name registry services, enabling internet navigation for many of the world’s most recognized domain names. We help enable the security, stability, and resiliency of the Domain Name System (“DNS”) and the ...

**Overall: wide.** The Company is the exclusive operator of the .com and .net registries and a root zone maintainer, which gives it durable monopoly-like positions backed by long-term renewing agreements. Price regulation and the dependence on ICANN and the DOC for renewal are the main risks. Social media and alternative namespaces pose limited competition. The moat is wide.

* **intangibles = 2.** The Company holds the exclusive registry rights for .com and .net under ICANN and DOC agreements, with presumptive renewal rights and automatic renewals. Competitors cannot operate these registries. Pricing is capped, and renewal depends on ICANN and the DOC, which adds some uncertainty.
  * [1] "We operate the .com, .net, and .name gTLDs and the IDN gTLDs under registry agreements with ICANN and also, with respect to the .com gTLD, a Cooperative Agreement with the U.S. Department of Commerce (“DOC”)."
  * [2] "Although the .com Registry Agreement contains a “presumptive” right of renewal, ICANN could terminate or refuse to renew the Registry Agreement in certain prescribed circumstances."
  * [3] "On November 30, 2024, the Cooperative Agreement was automatically renewed on the same terms for a successive six-year term and will automatically renew on November 30, 2030, unless the DOC provides written notice of non-renewal within 120 days prior to the end of the then-current term."
  * [4] "The RZMA is subject to an automatic eight-year renewal, unless earlier modified or terminated."
* **efficient_scale = 2.** The Company operates two of the thirteen root servers and is the sole operator of the .com and .net registries. This is a natural monopoly-like position, and a new entrant could not replicate it. Price caps limit the returns.
  * [5] "We operate two of the world’s thirteen root servers."
  * [1] "We operate the .com, .net, and .name gTLDs and the IDN gTLDs under registry agreements with ICANN and also, with respect to the .com gTLD, a Cooperative Agreement with the U.S. Department of Commerce (“DOC”)."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [9] "Under our agreements with ICANN, we are subject to certain restrictions in the operation of .com, .net and .name on pricing, bundling, marketing, methods of distribution, introduction of new registry services, and use of registrars, that do not apply to ccTLDs and other gTLDs and therefore may create a competitive disadvantage."
  * [10] "Demand for domain names could be negatively impacted to the extent end-users establish their online identities using social media (such as Facebook, Instagram or Tiktok) or transact business on e-commerce platforms (such as Amazon, Etsy and Taobao) instead of registering domain names."
  * [11] "Some of the software and protocols used in our business are in the public domain or are otherwise available to our competitors, and some are based on open standards set by organizations such as the Internet Engineering Task Force."

### ALB: ALBEMARLE CORP (Materials, 10-K filed 2025-02-12)

> Business. the Company was incorporated in Virginia in 1993. Our principal executive offices are located at 4250 Congress Street, Suite 900, Charlotte, North Carolina 28209. Unless the context otherwise indicates, the terms “the Company,” “we,” “us,” “our” or “the Company” mean the Company and its ...

**Overall: none.** The Company's owned lithium and bromine resources give it a possible cost position. However, lithium is increasingly priced against an index with aggressive new entrants, and the specialty and catalyst businesses compete on price and performance in highly competitive markets. No strong switching costs, network effects or intangible assets are stated.

* **cost_advantage = 1.** The Company has ownership stakes in the Greenbushes mine, which is a well-known low-cost resource, plus Atacama brine and Arkansas brine rights. The text does not compare costs with competitors, so the advantage is modest and uncertain.
  * [6] "We obtain lithium: (a) by purchasing lithium concentrate from our 49%-owned joint venture, Windfield Holdings Pty. Ltd. (“Windfield”), which directly owns 100% of the equity of Talison Lithium Pty. Ltd., a company incorporated in Australia (“Talison”) that owns the Greenbushes mine"
  * [7] "through solar evaporation of our ponds at the Salar de Atacama, in Chile, and in Silver Peak, Nevada."
  * [8] "Our bromine production operations in Arkansas are supported by an active brine rights leasing program."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "The global lithium market is highly competitive and growing very rapidly."
  * [2] "Competition in the global lithium market is increasingly based on index-based market pricing and differentiated via product quality, product diversity, reliability of supply and customer service."
  * [3] "It is characterized by aggressive expansion and entry from existing and new players, including automotive OEMs, commodity traders, junior miners, and large, well-capitalized diversified miners."
  * [4] "Product performance and quality, price and contract terms are the primary factors in determining which qualified supplier is awarded a contract."
  * [5] "Our Specialties business serves markets in the Americas, Asia, Europe and the Middle East, each of which is highly competitive."
  * [10] "most of which are readily available from numerous independent suppliers and are purchased or provided under contracts at prices we believe are competitive."

### MOS: MOSAIC CO (Materials, 10-K filed 2025-03-03)

> OVERVIEW The the Company Company is the world’s leading producer and marketer of concentrated phosphate and potash crop nutrients. Through our broad product offering, we are a single source supplier of phosphate- and potash-based crop nutrients and animal feed ingredients. We serve customers in ...

**Overall: none.** Phosphate and potash are commodities, and the Company states that competition is primarily on delivered price. Integrated rock supply, scale and large potash reserves give a moderate cost position, but state-owned and low-cost foreign producers limit it and returns will swing with the cycle. At best this is a narrow cost advantage, and the evidence against it is strong enough that no moat is the safer call.

* **cost_advantage = 1.** Vertical integration into phosphate rock, scale, owned sulfur logistics and one of the largest potash mines give the Company a real but only partial cost edge. The Company itself says it holds the advantage only over some competitors. Foreign and state-owned rivals may have lower costs, and the products stay commodities priced by the market.
  * [5] "We believe that our position as an integrated producer of phosphate rock provides us with a significant cost advantage over competitors that are non-integrated phosphate producers."
  * [6] "As the world’s second largest producer of concentrated phosphate, as well as the second largest miner of phosphate rock in the world and the largest in the U.S., we maintain an advantage over some competitors as the scale of operations effectively reduces production costs per unit."
  * [7] "We believe that our investments in sulfur logistical and melting assets continue to afford us a competitive advantage compared to other producers in cost and access to sulfur."
  * [8] "Esterhazy, the largest potash mine in the world, has completed full ramp up of capacity and production."
  * [9] "We believe that our potash cost structure is competitive in the industry and should improve as we continue to complete our potash expansion projects."
  * [13] "We account for approximately 70 to 80% of estimated annual production of concentrated phosphate crop nutrients in Brazil and 100% of estimated annual potash production in Brazil."
  * [15] "We believe the strategic location of our mines and chemical plants, in close proximity to our customers, and the benefit of our own distribution network, gives us an advantage over most of our competitors."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [1] "Because crop nutrients are global commodities available from numerous sources, crop nutrition companies compete primarily on the basis of delivered price."
  * [2] "Potash is a commodity available from several geographical regions around the world and, consequently, the market is highly competitive."
  * [3] "Phosphate producers compete primarily based on price, as well as product quality, service and innovation."
  * [4] "Foreign competitors may have access to less expensive raw materials, may not have to comply with as stringent regulatory requirements or are owned or subsidized by governments and, as a result, may have cost advantages over North American companies."

### PKG: PACKAGING CORP OF AMERICA (Materials, 10-K filed 2025-02-27)

> B USINESS the Company of America (“we,” “us,” “our,” “the Company,” or the “Company”) is the third largest producer of containerboard products and a leading producer of uncoated freesheet (UFS) paper in North America. We operate eight mills and 86 corrugated products plants and related facilities. ...

**Overall: none.** This is a commodity-oriented, cyclical, highly competitive industry with no dominant player. Price takers with many competitors and a declining UFS segment show no durable moat.

* Rated 0: intangibles, switching_costs, network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [3] "Our industries are highly competitive, with no single containerboard, corrugated packaging, or UFS paper producer having a dominant position."
  * [4] "Certain containerboard grades and UFS paper products cannot generally be differentiated by producer, which tends to intensify price competition."
  * [5] "Commodity grades of UFS paper are globally traded, with numerous worldwide manufacturers, and as a result, these products compete primarily on the basis of price."
  * [6] "According to industry sources, corrugated products are produced by about 380"
  * [7] "We have little influence over the timing and extent of price changes of our products, which may be unpredictable and volatile."
  * [8] "Some of our competitors are larger than we are and may have greater financial and other resources, greater manufacturing economies of scale, greater energy self-sufficiency, or lower operating costs, compared to our company."
  * [9] "In 2024, our sales revenue to ODP represented 58% of our Paper segment sales revenue and 4% of our consolidated sales revenue."

### ESS: ESSEX PROPERTY TRUST, INC. (Real Estate, 10-K filed 2025-02-21)

> OVERVIEW the Company (“the Company”), a Maryland corporation, is an S&P 500 company that operates as a self-administered and self-managed real estate investment trust (“REIT”). the Company owns all of its interest in its real estate and other investments directly or indirectly through the Company ...

**Overall: none.** The Company owns apartments in supply-constrained coastal markets, which helps, but the barriers benefit all owners and it faces numerous housing alternatives and competitors. There are no brand, switching cost, network or cost advantages, so no durable company-specific moat.

* **efficient_scale = 1.** Supply constraints in coastal West Coast markets (land scarcity, permitting barriers, geography) limit new competing housing, which supports property values. However, the market still has many owners and housing alternatives, so this is not a one-or-few-firm market and the benefit accrues to all owners there.
  * [1] "Constraints on new supply driven by: (i) low availability of developable land sites where competing housing could be economically built; (ii) political growth barriers, such as protected land, urban growth boundaries, and potential lengthy and expensive development permit processes; and (iii) natural limitations to development, such as mountains or waterways;"
* Rated 0: intangibles, switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [3] "There are numerous housing alternatives that compete with the Company’s communities in attracting tenants."
  * [4] "Some competitors are larger and have greater resources than the Company."
  * [5] "If the demand for the Company’s communities is reduced or if competitors develop and/or acquire competing housing, rental rates and occupancy may drop which may have a material adverse effect on the Company’s financial condition and results of operations."

### IRM: IRON MOUNTAIN INC (Real Estate, 10-K filed 2025-02-14)

> BUSINESS OVERVIEW the Company, a Delaware corporation ("IMI"), was founded in an underground facility near Hudson, New York in 1951 where it stored business records. Today, we are a global leader in information management services, and we are trusted by more than 240,000 customers in 61 countries, ...

**Overall: narrow.** The business has sticky, recurring customers with multi-year contracts and a trusted, compliant reputation, but competition is on price and from many providers, and no other source is strong. This points to a narrow moat at best.

* **intangibles = 1.** Long-standing trust and extensive compliance certifications support the data center and records businesses, but the advantage is not clearly quantified and competitors can also be certified.
  * [4] "our data center portfolio has achieved numerous certifications and received third party assurance reports, making it one of the most comprehensive compliance programs in the industry."
  * [6] "The world’s most heavily regulated organizations have trusted us as a data center operator for over 20 years, with five of the largest global hyperscalers among our customers."
* **switching_costs = 1.** Multi-year storage contracts, a long weighted average data center lease term and strong retention show customers are sticky; moving physical records is costly and risky.
  * [1] "In our Records Management business, our contracted storage rental fee agreements generally range from one to five years in length."
  * [2] "As of December 31, 2024, we stored more than 730 million cubic feet of physical volume and we have consistently experienced strong customer retention levels."
  * [3] "In our Global Data Center Business, our lease durations vary by customer, with a weighted average lease expiration of 10.6 years as of December 31, 2024."
* Rated 0: network_effects, cost_advantage, efficient_scale.
* Against a moat:
  * [7] "Competition for records and information customers is driven by factors such as pricing, reputation and reliability, the quality and security of storage solutions and the scope and scale of technology."
  * [8] "We also compete with numerous data center developers, owners and operators, many of whom own properties comparable to ours in several of the same metropolitan areas where our facilities are located."
  * [9] "We face competition from numerous storage and information management services providers globally, as well as storage and information management services that are managed and operated internally by organizations."

### O: REALTY INCOME CORP (Real Estate, 10-K filed 2025-02-25)

> Financial Statements the Company CORPORATION AND SUBSIDIARIES CONSOLIDATED BALANCE SHEETS (in thousands, except per share amounts) December 31, 2024 December 31, 2023 ASSETS Real estate held for investment, at cost: Land $ 17,320,520 $ 14,929,310 Buildings and improvements 40,974,535 34,657,094 ...

**Overall: none.** The Company is a large net lease REIT whose main edge is cheaper cost of capital from its investment grade ratings and scale. The text shows no brand, switching cost, network, or regulated advantage. Returns on capital are modest (net income about $0.87B on $69B assets), and tenant defaults and impairments show the weaker side of the business. The moat is at best a modest, uncertain cost-of-capital advantage.

* **cost_advantage = 1.** Investment grade credit ratings give cheaper funding, and the Spirit merger added scale for large net lease deals. This is a modest funding advantage, though real estate capital is largely accessible to many REITs.
  * [1] "Under our revolving credit facility, our investment grade credit ratings at December 31, 2024 provide for USD borrowings at Secured Overnight Financing Rate ("SOFR"), plus 0.725% with a SOFR adjustment charge of 0.10% and a revolving credit facility fee of 0.125%, for all-in pricing of 0.95% over SOFR"
  * [2] "The primary reason for the Merger was to expand our size, scale and diversification, in order to further position us as the real estate partner of choice for large net lease transactions."
  * [3] "As of December 31, 2024, the weighted average interest rate on our notes and bonds payable was 3.8%, and the weighted average remaining years until maturity was 6.6 years."
* Rated 0: intangibles, switching_costs, network_effects, efficient_scale.
* Against a moat:
  * [6] "The increase for credit losses on financing receivables is primarily due to a client in the convenience store industry that defaulted on its lease payments and was fully reserved for"

### CNP: CENTERPOINT ENERGY INC (Utilities, 10-K filed 2025-02-20)

> Business This combined Form 10-K is filed separately by three registrants: the Company., the Company Houston Electric, LLC and the Company Resources Corp. Information contained herein relating to any individual registrant is filed by such registrant solely on its own behalf. Each registrant makes ...

**Overall: narrow.** Regulated electric T&D monopolies with certificated or exclusive territories and a large gas distribution base give a durable position. Returns are capped by cost-of-service regulation, and gas franchises are non-exclusive and face electrification pressure, so the moat is narrow rather than wide.

* **intangibles = 1.** Exclusive Indiana territory certificates and a PUCT-issued CCN for Houston Electric are regulatory approvals that keep competitors out, but regulators also control rates and returns, so the advantage is modest. Gas franchises are mostly non-exclusive.
  * [3] "Houston Electric conducts its operations pursuant to a CCN issued by the PUCT that covers its present service area and facilities."
  * [5] "Indiana service territory certificates are exclusive."
* **efficient_scale = 2.** The electric T&D businesses are natural monopolies. No other utility serves Houston Electric's or Indiana Electric's areas, and entry would require regulatory approval, so a new entrant would not earn a return. Gas distribution is less protected, with non-exclusive franchises and competition from alternate fuels, but it is still a regulated local monopoly in practice.
  * [1] "There are no other electric transmission and distribution utilities in Houston Electric’s service area."
  * [2] "For another provider of transmission and distribution services to provide such services in Houston Electric’s territory, it would be required to obtain a CCN from the PUCT and, depending on the location of the facilities, may also be required to obtain franchises from one or more municipalities."
  * [4] "There are no other electric transmission and distribution utilities in Indiana Electric’s service area."
  * [6] "Houston Electric serves nearly all of the Houston/Galveston metropolitan area near the Texas Gulf Coast."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [8] "In most cases, franchises to provide natural gas utility services are not exclusive."
  * [9] "the Company and CERC’s Natural Gas compete primarily with alternate energy sources such as electricity and other fuel sources."
  * [10] "Houston Electric does not have long-term contracts with any of its customers, but rather operates using a continuous billing cycle, with meter readings being conducted and invoices being distributed to REPs each business day."
  * [11] "These initiatives could have a significant impact on the Company and its operations, and this impact could increase if other cities and jurisdictions in its service area enact similar initiatives."

### LNT: ALLIANT ENERGY CORP (Utilities, 10-K filed 2025-02-21)

> A. GENERAL the Company maintains its principal executive offices in Madison, Wisconsin. the Company operates as a regulated investor-owned public utility holding company, and its purpose-driven strategy is to serve its customers and build stronger communities. the Company primary focus is to ...

**Overall: narrow.** The company is a regulated monopoly utility with captive retail customers and cost recovery, which supports durable but regulator-limited returns. Municipalization and self-generation threats, non-exclusive franchises and rate regulation temper this. Overall this is a narrow moat.

* **intangibles = 1.** Regulatory approvals are needed to build generation, and Iowa law bars local regulation of gas sales. These are regulatory protections, but they are closer to the utility's franchise position, which is better captured under efficient scale. Franchises are non-exclusive and rate regulation limits returns.
  * [3] "IPL must obtain a certificate of public convenience, use and necessity (GCU Certificate) from the IUC in order to construct a new, or significantly alter (including fuel switching) an existing, EGU or energy storage facility located in Iowa with a nameplate generating capacity of 25 MW or more."
  * [4] "In Iowa, counties and cities are prohibited from regulating the sale of natural gas and propane, which supports IPL’s ability to provide gas utility service to retail customers and industries."
* **efficient_scale = 2.** Retail electric and gas customers cannot choose their supplier, and the utilities must serve everyone in regulated territories. This is a natural-monopoly position with cost recovery. Returns are capped by regulators, and some competitive threats exist, such as self-generation, municipalization and non-exclusive franchises.
  * [1] "Retail electric customers in Iowa and Wisconsin currently do not have the ability to choose their electric supplier, and IPL and WPL have obligations to serve all their retail electric customers."
  * [2] "Gas customers in Iowa and Wisconsin currently do not have the ability to choose their gas distributor, and IPL and WPL have obligations to serve all their gas customers."
  * [9] "Prudent expenditures incurred by IPL and WPL to comply with environmental requirements are eligible to be recovered in rates from their customers."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [6] "IPL and WPL still face competition from self-generation by large industrial customers, customer- and third party-owned generation (e.g. solar panels), alternative energy sources, and petitions to municipalize (Iowa) as well as service territory expansions by municipal utilities through annexations (Wisconsin)."
  * [7] "IPL provides utility services to incorporated communities as directed by the IUC and utilizes non-exclusive franchises, which cover the use of public right-of-ways for utility facilities in incorporated communities for a maximum term of 25 years."
  * [8] "In addition, the wholesale power market is competitive and IPL and WPL compete against independent power producers, other utilities and MISO market purchases to serve wholesale customers for their electric energy and capacity needs."

### PEG: PUBLIC SERVICE ENTERPRISE GROUP INC (Utilities, 10-K filed 2025-02-25)

> BUSINESS We were incorporated under the laws of the State of New Jersey in 1985 and our principal executive offices are located at 80 Park Plaza, Newark, New Jersey 07102. We are a public utility holding company that, acting through our wholly owned subsidiaries, is a predominantly regulated ...

**Overall: narrow.** The majority of earnings come from a franchised regulated utility with an exclusive territory and a rate base earning approved returns, which is durable. The merchant nuclear business is exposed to commodity pricing and regulatory uncertainty and weakens the picture. The regulatory compact caps returns, so the moat is narrow rather than wide.

* **intangibles = 1.** Regulatory franchise and approved rate returns (rate base, ROE) protect the utility, and nuclear licenses restrict entry; however, regulators can alter returns, and the licenses expire.
  * [1] "PSE&G—A New Jersey corporation, incorporated in 1924, which is a franchised public utility in New Jersey."
  * [5] "The Order provides for a $17.8 billion rate base, a 9.6% return on equity for PSE&G’s distribution business and a 55% equity component of its capitalization structure."
  * [6] "Our current approved transmission rates provide for a base ROE of 9.90% and a 50 basis point adder for our membership in PJM as an RTO."
  * [8] "The NRC has the ultimate authority to determine whether any U.S. nuclear generating unit may operate."
* **efficient_scale = 2.** PSE&G is a franchised T&D monopoly serving 74% of New Jersey's population with formula-based rate recovery and decoupling; a duplicate grid would not earn a return. This is a durable, regulated position, though regulators limit returns.
  * [1] "PSE&G—A New Jersey corporation, incorporated in 1924, which is a franchised public utility in New Jersey."
  * [2] "PSE&G distributes electricity and natural gas to end users in our respective franchised service territories."
  * [3] "We serve the most densely populated, commercialized and industrialized territory in New Jersey, including its six largest cities and approximately 300 suburban and rural communities."
  * [4] "We provide distribution service to 2.4 million electric customers and 1.9 million gas customers in a service area that covers approximately 2,600 square miles running diagonally across New Jersey."
  * [7] "over 90% of our Electric and Gas Distribution margin will only vary based upon the number of customers."
* Rated 0: switching_costs, network_effects, cost_advantage.
* Against a moat:
  * [10] "Our competitors include but are not limited to merchant generators, utility generators, energy marketers, retailers, private equity firms, and other financial entities."
  * [11] "Typically, the bid price of the last unit dispatched by PJM establishes the energy market-clearing price."
  * [12] "Changes previously ordered by FERC and implemented by PJM and other ISOs to eliminate contractual provisions that previously provided us a “right of first refusal” to construct new transmission projects in our service territory could result in third-party construction of transmission lines in our area in the future"
  * [13] "It is uncertain whether the OSA will be renewed."
