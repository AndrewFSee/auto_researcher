# Customer-supplier links: feasibility count (2024 10-Ks)

Generated 2026-11-14 14:01 by `scripts/customer_links.py`, model `qwen3:8b`, prompt `97826f76`. No returns were computed.

* 10-Ks filed in 2024 (one per company, sentiq's lake): 516; with text near a customer mention and a percentage of revenue or sales: 446; model calls 519 (1.2 h).
* Customers returned 1247; dropped: quote not in the text 78, named customer not in its quote 84, percentage not in the quote 112, no percentage in the quote 37, quote without revenue or sales 45, purchases from suppliers 21; unparseable replies 0. Percentages lined up with their names in "…, respectively" lists: 1.
* 10-Ks stating no customer reached 10%: 159.

## Count

| | Companies |
| --- | ---: |
| Any customer with a share of total or segment revenue | 302 |
| A *named* customer (total or segment revenue) | 161 |
| **A named, listed customer with 10%+ of total revenue (a link)** | **60** |
| Only a segment-level link | 3 |

Links: 91 (exact name match 64, alias 18, prefix 9); distinct customers 46.

Customers by kind (total or segment revenue):

* named and matched to a listed company: 113
* named, not matched (private, government, foreign, or a name the matcher misses): 219
* not named ("Customer A", "one customer"): 245

Most common unmatched names: United States (6); Russia (3); U.S. government (3); Customer A (3); Distributor A (3); Distributor B (3); Distributor C (3); Customer B (2); Customer C (2); retail customers (2); EGPC (2); CFE International (2); Japan (2); Florida (2); Samsung Electronics Co., Ltd. (2); F-35 program (1); U.S. government tenants (1); Investment-grade or publicly traded large cap tenants (1); GMI (1); GMNA (1); U.S. government (including foreign military sales through the U.S. government) (1); MEGlobal (1); Ukraine (1); Germany (1); Airbus (1)

## Check against inline XBRL

516 filings read; 116 tag a customer with 10% or more of revenue for the year reported (209 customer members).

* Filings where the text extraction found a customer: 105 of 116.
* Tagged customers whose percentage (within 1 point) the extraction found: 151 of 209.
* Missed filings: BALL, CCI, COP, DXCM, LIN, LLY, MRNA, PKG, STZ, TFX, VRTX.
* Tagged members naming a listed company (supplier, customer pairs): 79; found as links in the text: 65. Not found: BALL → KO (TheCocaColaCompanyMember), BMY → CAH (CardinalHealthIncMember), BMY → COR (CencoraMember), BMY → MCK (MckessonCorporationMember), CCI → TMUS (TMobileMember), CCI → VZ (VerizonWirelessMember), LIN → MBIN (MerchantMember), MNST → KO (CocaColaCompanyMember), MSCI → BLK (BlackrockIncMember), NEM → JPM (JpMorganChaseMember), NEM → RY (RoyalBankOfCanadaMember), NEM → TD (TorontoDominionBankMember), SBAC → VIV (TelefonicaMember), VRTX → MCK (McKessonCorporationMember).
* Text links in filings that tag customers but not that one: AMGN → CAH, AMGN → COR, AMGN → MCK, AMT → T, ANET → META, ANET → MSFT, CF → CHSCP, EA → MSFT, EA → SONY, FFIV → INGM, FFIV → SNX, MCK → CVS, MNST → CCEP, NTAP → SNX, QCOM → AAPL, SBAC → TIMB, SWKS → AAPL, TTWO → AAPL, TTWO → GOOGL, TTWO → MSFT, TTWO → SONY.
* Tagged members not matched to a listed company (most are unnamed: Customer A, one customer): CustomerOneMember (12); CustomerAMember (8); OneCustomerMember (6); LargestCustomerMember (5); CustomerCMember (4); CustomerBMember (4); CustomerTwoMember (4); CustomerDMember (3); USGovernmentMember (3); DistributorAMember (3); DistributorBMember (3); DistributorCMember (3); FixedPriceContractMember (3); FederalGovernmentMember (2); SalesChannelDirectlyToConsumerMember (2); SalesChannelThroughIntermediaryMember (2); ThreeLargestWholesalesMember (2); EgyptianGeneralPetroleumCompanyMember (2); OfficeDepotMember (2); GroupPurchasingOrganizationsMember (2); DellIncMember (1); LenovoGroupLimitedMember (1); HPInc.Member (1); USGovernmentContractsMember (1); USFederalGovernmentMember (1); VolkswagenMember (1); NetRevenuesLargestCustomerMember (1); CustomerEMember (1); DoDAndUSIntelligenceCommunityMember (1); CustomerThreeMember (1).

## Hand check

The 81 total-revenue links of the first report were checked by hand on 2026-11-14 against the passages the model read: correct 68, not a revenue customer 7, wrong % or year 6. The rules below were then added and tuned on those marks, so the 2024 figures after them are in-sample; another year needs its own check.

* Rules added after the check: a quote about purchases from suppliers is dropped; a share without a percentage needs one in the quote ("10% or more"); a table row under a heading for part of the company's revenue (product, segment, international) is a segment share; percentages in "A and B … 18% and 21%, respectively" are lined up with the names; an alias table maps subsidiaries, short names and delisted companies to tickers.
* Checked links the rules keep: 73 (correct 68, wrong % or year 5); dropped: 8 (not a revenue customer 7, wrong % or year 1).
* Percentages the rules corrected: ANET → Meta Platforms 18% → 21%.
* Precision of the kept links, as marked, counting corrected percentages as correct: 69 of 73 (95%).
* Marks to check again: AMT → AT&T (marked wrong % or year): the check page showed another passage of the filing, not the quoted one; to be checked again against the quote; FANG → Shell Trading (USA) Company ("Shell") (marked wrong % or year): the check page showed another passage of the filing, not the quoted one; to be checked again against the quote.
* Wrong links the rules still keep: AEP → NRG (wrong % or year: 41% is NRG and TXU combined, as a share of the AEP Texas subsidiary's revenue, not AEP consolidated revenue.); CHD → WMT (wrong % or year: Walmart was 23% of 2023 net sales; 24% is the 2022 and 2021 figure.).
* Unmatched names marked as listed companies: 41 of 172; now matched 22, dropped by the new checks 5 (BBY → 005930.KS (Samsung); BBY → HPQ (HP); BBY → KRX: 066570 (LG); NWSA → TSCDY, TSCO (Tesco); PXD → EQNR (Equinor Marketing & Trading US Inc.)), still unmatched 14.
* New links since the check: 18, of them 18 from names marked as listed (ticker agrees in 18; their shares are not yet checked); other new links: none.
* Names marked as listed that still match no ticker (foreign listings, the filer's own subsidiaries, suppliers, a group of three, a company off SEC's current list): AEP → AEP (AEP Subsidiaries); AMAT → 005930.KS (Samsung Electronics Co., Ltd.); BWA → VWAGY (Volkswagen); FICO → EFX (Experian, TransUnion and Equifax); L → CNA (CNA); NDAQ → NDAQ (Adenza); NWSA → NWSA (Foxtel Group); NWSA → SPOT (Spotify); OXY → WES (WES); QCOM → KRX: 005930 (Samsung); QCOM → 1810.HK (Xiaomi); QRVO → KRX: 005930 (Samsung Electronics Co., Ltd.); SBAC → TEF (Telefonica); SO → SO (Mississippi Power).

## Links

| Supplier | Customer (as written) | Ticker | Match | % | Share of | Year | Hand check |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| ABC | Walgreens Boots Alliance, Inc. ("WBA") | WBA | alias | 26 | total revenue | 2024 | not checked |
| ABC | Evernorth Health Services (formerly Express Scripts, Inc.) | CI | alias | 13 | total revenue | 2024 | not checked |
| ADSK | TD Synnex Corporation and its global affiliates (collectively, "TD Synnex") | SNX | prefix | 39 | total revenue | 2024 | correct |
| AEP | NRG Energy | NRG | exact | 41 | total revenue | 2023 | wrong % or year |
| AMAT | Taiwan Semiconductor Manufacturing Company Limited | TSM | exact | 11 | total revenue | 2024 | correct |
| AMGN | McKesson Corporation | MCK | exact | 33 | total revenue | 2023 | correct |
| AMGN | Cencora, Inc. (formerly AmerisourceBergen Corporation) | COR | prefix | 29 | total revenue | 2023 | correct |
| AMGN | Cardinal Health, Inc. | CAH | exact | 17 | total revenue | 2023 | correct |
| AMT | T-Mobile | TMUS | alias | 17 | total revenue | 2023 | not checked |
| AMT | AT&T | T | exact | 16 | total revenue | 2023 | wrong % or year |
| AMT | Verizon Wireless | VZ | alias | 12 | total revenue | 2023 | not checked |
| ANET | Meta Platforms | META | exact | 21 | total revenue | 2023 | wrong % or year |
| ANET | Microsoft | MSFT | exact | 18 | total revenue | 2023 | correct |
| ARNC | General Electric Company | GE | exact | 12 | total revenue | 2023 | correct |
| AXP | Delta | DAL | alias | 10 | total revenue | 2023 | not checked |
| BSX | Emerging Markets | EMRH | prefix | 16 | segment revenue | 2023 | not a revenue customer |
| BWA | Ford | F | alias | 14 | total revenue | 2023 | not checked |
| CAG | Walmart, Inc. and its affiliates | WMT | exact | 28 | total revenue | 2024 | correct |
| CAH | CVS Health | CVS | exact | 24 | total revenue | 2024 | correct |
| CAH | OptumRx | UNH | alias | 17 | total revenue | 2024 | not checked |
| CF | CHS | CHSCP | exact | 13 | total revenue | 2023 | correct |
| CHD | Walmart Inc. and its affiliates | WMT | exact | 24 | total revenue | 2023 | wrong % or year |
| CL | Walmart, Inc. | WMT | exact | 11 | total revenue | 2023 | correct |
| CLX | Walmart Stores, Inc. and its affiliates | WMT | exact | 25 | total revenue | 2024 | correct |
| CMI | PACCAR | PCAR | exact | 16 | total revenue | 2023 | correct |
| CPB | Wal-Mart Stores, Inc. and its affiliates | WMT | exact | 22 | total revenue | 2024 | correct |
| EA | Sony | SONY | exact | 37 | total revenue | 2024 | correct |
| EA | Microsoft | MSFT | exact | 16 | total revenue | 2024 | correct |
| EXE | Valero Energy Corporation | VLO | exact | 17 | total revenue | 2023 | correct |
| EXE | Shell Energy North America | SHEL | prefix | 10 | total revenue | 2023 | correct |
| FANG | DK Trading & Supply LLC | DK | alias | 18 | total revenue | 2023 | not checked |
| FANG | Shell Trading (USA) Company ("Shell") | SHEL | prefix | 14 | total revenue | 2023 | wrong % or year |
| FANG | Enterprise Crude Oil LLC | EPD | alias | 13 | total revenue | 2023 | not checked |
| FFIV | Ingram Micro, Inc. | INGM | exact | 16 | total revenue | 2024 | correct |
| FFIV | Synnex Corporation | SNX | alias | 15 | total revenue | 2024 | not checked |
| GILD | Cardinal Health, Inc. | CAH | exact | 26 | total revenue | 2023 | correct |
| GILD | Cencora, Inc. (formerly known as AmerisourceBergen Corporation) | COR | prefix | 19 | total revenue | 2023 | correct |
| GILD | McKesson Corporation | MCK | exact | 21 | total revenue | 2023 | correct |
| GIS | Walmart Inc. | WMT | exact | 22 | total revenue | 2024 | correct |
| HAS | Amazon.com, Inc. | AMZN | exact | 11 | total revenue | 2023 | correct |
| HAS | Walmart, Inc. | WMT | exact | 11 | total revenue | 2023 | correct |
| HRL | Walmart Inc. | WMT | exact | 15.6 | total revenue | 2024 | correct |
| HSY | McLane Company, Inc. | BRK-B | alias | 28 | total revenue | 2023 | not checked |
| JBL | Apple, Inc. | AAPL | exact | 11 | total revenue | 2024 | correct |
| K | Wal-Mart Stores, Inc. and its affiliates | WMT | exact | 15 | total revenue | 2023 | correct |
| KDP | Walmart | WMT | exact | 17 | total revenue | 2023 | correct |
| KHC | Walmart Inc. | WMT | exact | 21 | total revenue | 2023 | correct |
| KMB | Walmart Inc. | WMT | exact | 13 | total revenue | 2023 | correct |
| LW | McDonald’s Corporation | MCD | exact | 14 | total revenue | 2024 | correct |
| MAS | The Home Depot | HD | exact | 39 | total revenue | 2023 | correct |
| MCHP | Arrow Electronics | ARW | exact | 12 | total revenue | 2024 | correct |
| MCK | CVS Health Corporation | CVS | exact | 28 | total revenue | 2024 | correct |
| MKC | Wal-Mart Stores, Inc. | WMT | exact | 12 | total revenue | 2023 | correct |
| MKC | PepsiCo, Inc. | PEP | exact | 13 | total revenue | 2023 | correct |
| MNST | Coca-Cola Consolidated, Inc. | COKE | exact | 10 | total revenue | 2023 | correct |
| MNST | Coca-Cola Europacific Partners | CCEP | exact | 13 | total revenue | 2023 | correct |
| MO | Performance Food Group Company | PFGC | exact | 25 | total revenue | 2023 | correct |
| MO | McLane Company, Inc. | BRK-B | alias | 23 | total revenue | 2023 | not checked |
| MRO | Marathon Petroleum Corporation | MPC | exact | 22 | total revenue | 2023 | correct |
| MRO | Valero Marketing and Supply | VLO | alias | 11 | total revenue | 2023 | not checked |
| NTAP | Arrow Electronics, Inc. | ARW | exact | 22 | total revenue | 2024 | correct |
| NTAP | TD Synnex Corporation | SNX | exact | 22 | total revenue | 2024 | correct |
| NXPI | Avnet | AVT | exact | 21 | total revenue | 2023 | correct |
| PEP | Walmart Inc. | WMT | exact | 14 | total revenue | 2023 | correct |
| PFE | McKesson, Inc. | MCK | exact | 17 | total revenue | 2023 | correct |
| PFE | Cencora, Inc. (formerly AmerisourceBergen Corporation) | COR | prefix | 12 | total revenue | 2023 | correct |
| PFE | Cardinal Health, Inc. | CAH | exact | 10 | total revenue | 2023 | correct |
| PG | Walmart Inc. and its affiliates | WMT | exact | 16 | total revenue | 2024 | correct |
| PXD | Energy Transfer Crude Marketing LLC | ET | prefix | 28 | total revenue | 2023 | correct |
| PXD | Shell Trading US Company | SHEL | prefix | 16 | total revenue | 2023 | correct |
| PXD | Occidental Energy Marketing Inc. | OXY | alias | 11 | total revenue | 2023 | not checked |
| QCOM | Apple | AAPL | exact | 10 | total revenue | 2024 | correct |
| QRVO | Apple Inc. | AAPL | exact | 46 | total revenue | 2024 | correct |
| REGN | Besse Medical, a subsidiary of Cencora, Inc. | COR | alias | 51 | segment revenue | 2023 | not checked |
| REGN | McKesson Corporation | MCK | exact | 25 | segment revenue | 2023 | not a revenue customer |
| RTX | Boeing | BA | exact | 19 | segment revenue | 2023 | not a revenue customer |
| SBAC | T-Mobile | TMUS | alias | 32.5 | total revenue | 2023 | not checked |
| SBAC | AT&T Wireless | T | prefix | 19.5 | total revenue | 2023 | correct |
| SBAC | Verizon Wireless | VZ | alias | 14.6 | total revenue | 2023 | not checked |
| SBAC | Claro | AMX | alias | 20.2 | segment revenue | 2023 | not checked |
| SBAC | TIM | TIMB | exact | 15.7 | segment revenue | 2023 | not a revenue customer |
| SJM | Walmart Inc. | WMT | exact | 33 | total revenue | 2024 | correct |
| SWK | Lowe's | LOW | exact | 14 | total revenue | 2023 | correct |
| SWK | The Home Depot | HD | exact | 13 | total revenue | 2023 | correct |
| SWKS | Apple | AAPL | exact | 69 | total revenue | 2024 | correct |
| TER | Texas Instruments Inc. | TXN | exact | 10 | total revenue | 2023 | correct |
| TSN | Walmart Inc. | WMT | exact | 18 | total revenue | 2024 | correct |
| TTWO | Sony | SONY | exact | 10 | total revenue | 2024 | correct |
| TTWO | Apple | AAPL | exact | 10 | total revenue | 2024 | correct |
| TTWO | Google | GOOGL | alias | 10 | total revenue | 2024 | not checked |
| TTWO | Microsoft | MSFT | exact | 10 | total revenue | 2024 | correct |
| UPS | Amazon.com, Inc. and its affiliates | AMZN | exact | 11.8 | total revenue | 2023 | correct |
| VICI | Caesars | CZR | alias | 37 | total revenue | 2023 | not checked |
| VTRS | McKesson Corporation | MCK | exact | 10 | total revenue | 2023 | correct |
| VTRS | AmerisourceBergen Corporation | COR | exact | 10 | total revenue | 2023 | correct |
| WHR | Lowe's | LOW | exact | 13 | total revenue | 2023 | correct |
| XRAY | Henry Schein, Inc. | HSIC | exact | 14 | total revenue | 2023 | correct |
