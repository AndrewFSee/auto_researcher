"""Tests for the customer extraction's text handling: windows, checks, name matching, XBRL."""

from __future__ import annotations

from auto_researcher.llm.customers import (
    SEPARATOR,
    TickerIndex,
    candidate_windows,
    check_customer,
    collect_customers,
    core_name,
    flatten,
    group_windows,
    xbrl_concentrations,
)

KHC = ("Our largest customer, Walmart Inc., represented approximately 21% of our net sales in 2023 "
       "and 2022, and approximately 22% of our net sales in 2021.")


def test_windows_find_customer_disclosures_and_share_clauses_only():
    filler = "The weather was fine and nothing happened. " * 100
    text = flatten(filler + KHC + filler + "Sales to Lowe's accounted for 12% of net sales. " + filler
                   + "Our customers like our products. " + filler)
    spans = candidate_windows(text)
    covered = [text[a:b] for a, b in spans]
    assert any("Walmart Inc." in c for c in covered)
    assert any("Lowe's accounted for 12%" in c for c in covered)  # no "customer" word: the share clause
    assert not any("like our products" in c and "Walmart" not in c and "Lowe" not in c for c in covered)
    assert candidate_windows(flatten(filler)) == []


def test_flatten_joins_table_cells_and_windows_merge():
    assert flatten("Customer\nA\n28\n%\n") == "Customer A 28 %"
    text = "x" * 50 + "one customer accounted for 10% of revenue; another customer accounted for 12% of sales" + "y" * 50
    assert len(candidate_windows(text)) == 1  # two mentions, overlapping windows


def test_group_windows_respects_the_limit_and_splits_long_windows():
    text = "a" * 25_000
    groups = group_windows(text, [(0, 25_000)], limit=10_000)
    assert [len(g) for g in groups] == [10_000, 10_000, 5_000]
    groups = group_windows("b" * 3_000 + "c" * 3_000, [(0, 3_000), (3_000, 6_000)], limit=10_000)
    assert groups == ["b" * 3_000 + SEPARATOR + "c" * 3_000]


def _c(customer, percent, quote, named=True, share_of="total revenue"):
    return {"customer": customer, "named": named, "percent": percent, "share_of": share_of, "year": "2023",
            "quote": quote}


def test_check_customer_needs_quote_name_percent_and_revenue():
    assert check_customer(_c("Walmart Inc.", 21, KHC), KHC) is None
    assert check_customer(_c("Walmart Inc.", 21, "Walmart Inc. represented 21% of our net sales"), "other text") \
        == "not_in_text"
    assert check_customer(_c("the U.S. government", 21, KHC), KHC) == "name_not_in_quote"
    assert check_customer(_c("one customer", 21, KHC, named=False), KHC) is None  # labels need not appear
    assert check_customer(_c("Walmart Inc.", 25, KHC), KHC) == "percent_not_in_quote"
    assert check_customer(_c("Walmart Inc.", 0, KHC), KHC) is None  # no percentage given, but one is quoted
    books = "HarperCollins sells books to a customer base that includes Amazon, Apple and Tesco, a share of sales."
    assert check_customer(_c("Apple", 0, books), books) == "no_share_in_quote"
    bby = "Five suppliers, Apple, Samsung, HP, Sony and LG, represented approximately 55% of total merchandise purchased."
    assert check_customer(_c("Apple", 55, bby), "Risks that could affect our revenue. " + bby) == "purchases"
    vol = "Delta represented approximately 10 percent of worldwide network volumes."
    assert check_customer(_c("Delta", 10, vol), vol) == "no_revenue_in_quote"


def test_a_share_stated_for_a_group_of_customers_is_not_one_customers():
    wab = ("For the fiscal year ended December 31, 2019, our top five customers accounted for approximately 27% of "
           "net sales: BNSF Railway, Canadian National Railway, Indian Railways, Komatsu Ltd. and Union Pacific Railroad.")
    assert check_customer(_c("Union Pacific Railroad", 27, wab), wab) == "group_share"
    cci = ("Our four largest customers are AT&T, T-Mobile, Verizon Wireless and Sprint. Collectively, these four "
           "customers accounted for 83% of our 2017 site rental revenues.")
    assert check_customer(_c("AT&T", 83, cci), cci) == "group_share"
    # Each member's own share: a table row under the group's heading, "each", "respectively".
    bmy = ("Gross revenue to the three largest pharmaceutical wholesalers in the U.S. as a percentage of global gross "
           "revenues was as follows: Year Ended December 31, 2019 2018 2017 McKesson Corporation 26 % 25 % 24 %")
    assert check_customer(_c("McKesson Corporation", 26, bmy), bmy) is None
    lly = ("In 2019, 2018, and 2017, three wholesale distributors in the U.S. - McKesson Corporation, AmerisourceBergen "
           "Corporation, and Cardinal Health, Inc. - each accounted for between 14 percent and 21 percent of our "
           "consolidated total revenue")
    assert check_customer(_c("McKesson Corporation", 14, lly), lly) is None
    pxd = ("Sunoco Logistics Partners L.P., Occidental Energy Marketing Inc. and Plains Marketing L.P accounted for "
           "36 percent, 18 percent and 14 percent of the Company's oil and gas revenues, respectively.")
    assert check_customer(_c("Plains Marketing L.P", 14, pxd), pxd) is None
    # A group mentioned beside one customer's own share.
    both = "Our ten largest customers accounted for 45% of net sales, and Walmart alone accounted for 20% of net sales."
    assert check_customer(_c("Walmart", 20, both), both) is None
    # Members' own shares listed without "respectively"; a share in the next sentence; one customer's parts.
    has = ("In 2017 the Company's largest customers were Wal-Mart Stores, Inc., Toys\"R\"Us, Inc. and Target "
           "Corporation. Sales to these customers amounted to 19 % , 9 % and 9 % , of consolidated net revenues")
    assert check_customer(_c("Wal-Mart Stores, Inc.", 19, has), has) is None
    nbl = ("BP North American Funding (BP) and Shell Trading (US) (Shell) were the largest single purchasers of our "
           "2017 production. Sales to BP accounted for 10% of 2017 total crude oil, natural gas and NGL sales revenues")
    assert check_customer(_c("BP North American Funding (BP)", 10, nbl), nbl) is None
    rht = "the U.S. government and its agencies represented in the aggregate approximately 10% of our total revenue."
    assert check_customer(_c("U.S. government and its agencies", 10, rht), rht) is None


def test_collect_customers_dedupes_prefers_total_revenue_and_moves_receivables():
    table = ("The following table summarizes those customers with accounts receivable equal to 10% or more: "
             "Distributor A 29 % 26 %")
    replies = [
        (KHC, {"customers": [_c("Walmart Inc.", 22, KHC, share_of="segment revenue"),
                             _c("Walmart Inc.", 21, KHC)], "no_major_customer": False}),
        (table + " of sales", {"customers": [_c("Distributor A", 29, table + " of sales", named=False)]}),
        ("x", None),
    ]
    customers, counts = collect_customers(replies)
    assert counts == {"returned": 3, "unparsed": 1}
    walmart = next(c for c in customers if c["customer"] == "Walmart Inc.")
    assert walmart["share_of"] == "total revenue" and walmart["percent"] == 21
    assert next(c for c in customers if c["customer"] == "Distributor A")["share_of"] == "other"


def test_ticker_matching_exact_prefix_former_names_and_generic_words():
    assert core_name("Lowe's Companies, Inc.") == "lowes"
    assert core_name("Walmart Inc. and its affiliates") == "walmart"
    assert core_name("VALERO ENERGY CORP/TX") == "valero energy"
    assert core_name("BANK OF MONTREAL /CAN/") == "bank of montreal"
    index = TickerIndex([("WMT", "Walmart Inc.", 104169), ("ET", "Energy Transfer LP", 1276187),
                         ("RTX", "RTX Corp", 101829), ("GD", "GENERAL DYNAMICS CORP", 40533),
                         ("UNP", "UNION PACIFIC CORP", 100885)])
    assert index.match("Walmart Inc. and its affiliates") == ("WMT", 104169, "exact")
    assert index.match("Energy Transfer Crude Marketing LLC") == ("ET", 1276187, "prefix")
    assert index.match("Wal-Mart Stores, Inc. and its affiliates") == ("WMT", 104169, "prefix")
    assert index.match("Raytheon Technologies Corporation") is None
    index.add_former_names([("RAYTHEON TECHNOLOGIES CORP", 101829), ("SOME PRIVATE CO", 999)])
    assert index.match("Raytheon Technologies Corporation") == ("RTX", 101829, "exact")
    assert index.match("Some Private Co") is None  # not a listed company
    assert index.match("General Motors") is None  # "general" alone is not a match
    assert index.match("Union") is None
    assert index.match("McLane Company, Inc.") == ("BRK-B", 1067983, "alias")  # a Berkshire subsidiary
    assert index.match("Walgreens Boots Alliance, Inc.") == ("WBA", 1618921, "alias")  # off SEC's current list


def test_names_that_share_a_short_form_with_a_listed_company_are_not_it():
    index = TickerIndex([("KO", "COCA COLA CO", 21344), ("SUN", "Sunoco LP", 1552275),
                         ("EMRH", "EMERGING HOLDINGS INC", 1300306), ("JYNT", "JOINT Corp", 1612630),
                         ("SHEL", "Shell plc", 1306965)])
    assert index.match("The Coca-Cola Company") == ("KO", 21344, "exact")
    assert index.match("Coca-Cola European Partners") == ("CCEP", 1650107, "alias")
    assert index.match("Coca-Cola Bottlers' Sales & Services Company LLC") is None
    assert index.match("Sunoco Logistics Partners L.P.") == ("ET", 1276187, "alias")  # part of Energy Transfer
    assert index.match("Emerging Markets") is None and index.match("Emerging") is None
    assert index.match("joint ventures in which we have a controlling interest") is None  # no capital: no company
    assert index.match("Shell Trading (US) Company") == ("SHEL", 1306965, "prefix")
    assert index.match_member("mnst:CocaColaEuropeanPartnersMember") == ("CCEP", 1650107)


def test_table_rows_take_their_base_from_the_heading_and_respectively_lists_line_up():
    heading = ("Sales to each of these customers as a percentage of the Company's total gross product revenue are "
               "as follows: 2023 2022 Besse Medical 51% 55% McKesson Corporation 25% 24%")
    total = "The following customers represented at least 10% of our total revenues: 2023 2022 T-Mobile 32.5% 36.4% AT&T Wireless 19.5% 19.6%"
    customers, _ = collect_customers([
        (heading, {"customers": [_c("McKesson Corporation", 25, "McKesson Corporation 25% 24%")]}),
        (total, {"customers": [_c("AT&T Wireless", 19.5, "AT&T Wireless 19.5% 19.6%")]}),
    ])
    assert {c["customer"]: c["share_of"] for c in customers} == {"McKesson Corporation": "segment revenue",
                                                                  "AT&T Wireless": "total revenue"}
    anet = ("sales to our end customers Microsoft and Meta Platforms represented 18% and 21% of our total revenue, "
            "respectively, in fiscal 2023")
    customers, counts = collect_customers([(anet, {"customers": [_c("Microsoft", 18, anet), _c("Meta Platforms", 18, anet)]})])
    assert {c["customer"]: c["percent"] for c in customers} == {"Microsoft": 18, "Meta Platforms": 21}
    assert counts["realigned"] == 1


def test_match_member_reads_camel_case_xbrl_members():
    index = TickerIndex([("WMT", "Walmart Inc.", 104169), ("MCK", "MCKESSON CORP", 927653),
                         ("TMUS", "T-Mobile US, Inc.", 1283699), ("BLK", "BlackRock, Inc.", 2012383)])
    assert index.match_member("khc:WalMartStoresIncMember") == ("WMT", 104169)
    assert index.match_member("cah:McKessonCorporationMember") == ("MCK", 927653)
    assert index.match_member("vz:TMobileMember") == ("TMUS", 1283699)
    assert index.match_member("msci:BlackrockIncMember") == ("BLK", 2012383)
    assert index.match_member("us-gaap:CustomerAMember") is None
    assert index.match_member("enph:NetRevenuesLargestCustomerMember") is None


def test_xbrl_concentrations_reads_customer_revenue_facts_only():
    html = b"""<html><body>
    <ix:header><xbrli:context id="c1"><xbrli:entity><xbrli:segment>
      <xbrldi:explicitMember dimension="us-gaap:ConcentrationRiskByBenchmarkAxis">us-gaap:RevenueFromContractWithCustomerMember</xbrldi:explicitMember>
      <xbrldi:explicitMember dimension="us-gaap:ConcentrationRiskByTypeAxis">us-gaap:CustomerConcentrationRiskMember</xbrldi:explicitMember>
      <xbrldi:explicitMember dimension="srt:MajorCustomersAxis">khc:WalmartIncMember</xbrldi:explicitMember>
    </xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>2023-12-30</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:context id="c2"><xbrli:entity><xbrli:segment>
      <xbrldi:explicitMember dimension="us-gaap:ConcentrationRiskByBenchmarkAxis">us-gaap:AccountsReceivableMember</xbrldi:explicitMember>
      <xbrldi:explicitMember dimension="srt:MajorCustomersAxis">khc:WalmartIncMember</xbrldi:explicitMember>
    </xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2023-12-30</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:context id="c3"><xbrli:entity><xbrli:segment>
      <xbrldi:explicitMember dimension="us-gaap:ConcentrationRiskByBenchmarkAxis">us-gaap:RevenueFromContractWithCustomerMember</xbrldi:explicitMember>
    </xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>2023-12-30</xbrli:endDate></xbrli:period></xbrli:context>
    </ix:header>
    <p><ix:nonFraction name="us-gaap:ConcentrationRiskPercentage1" contextRef="c1" scale="-2" unitRef="pure">21</ix:nonFraction></p>
    <p><ix:nonFraction name="us-gaap:ConcentrationRiskPercentage1" contextRef="c2" scale="-2" unitRef="pure">30</ix:nonFraction></p>
    <p><ix:nonFraction name="us-gaap:ConcentrationRiskPercentage1" contextRef="c3" scale="-2" unitRef="pure">40</ix:nonFraction></p>
    </body></html>"""
    assert xbrl_concentrations(html) == [
        {"member": "khc:WalmartIncMember", "percent": 21.0, "start": "2023-01-01", "end": "2023-12-30"}]
