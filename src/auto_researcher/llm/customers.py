"""
Major customers from 10-K text: the customer-supplier links behind "customer momentum"
(Cohen & Frazzini, 2008), extracted for a feasibility count before any return test.

Companies must disclose customers that account for 10% or more of revenue, usually in a
sentence ("Walmart Inc. represented approximately 21% of our net sales") or a small table
in the notes. The model only *reads*: it lists each customer, its share and the sentence
that says so, and the quote is checked word for word against the filing (``verify_quote``
from the moat tagger). Nothing is judged, so the model's knowledge of how companies fared
later cannot leak into the links.

1. ``candidate_windows``: the filing text around every mention of a customer (or a
   share-of-sales clause) near a percentage and the word revenue or sales, merged and
   grouped into calls (``CALL_CHARS``).
2. ``extract_messages`` / ``EXTRACT_SCHEMA``: one local-model call per group (Ollama).
3. ``collect_customers``: verified customers per filing, de-duplicated.
4. ``match_ticker``: the customer's name against SEC's company list (exact core name, or
   a listed company's core name as the first words: "Energy Transfer Crude Marketing LLC"
   -> Energy Transfer).
5. ``xbrl_concentrations``: the revenue concentrations the filing tags in inline XBRL
   (``ConcentrationRiskPercentage1`` with a customer member), an independent check of
   what the text extraction misses.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable

import lxml.html

from auto_researcher.llm.filings import _contexts, _local
from auto_researcher.llm.moat import _GENERIC_FIRST, _SUFFIX, _compact, quote_key, verify_quote

BEFORE_CHARS = 400   # text kept before a mention (the sentence that starts the disclosure)
AFTER_CHARS = 1200   # and after it (a following table of customers and percentages)
CONTEXT_CHARS = 300  # how far around a mention the percentage and "revenue"/"sales" may be
LEAD_IN_CHARS = 300  # text before a quoted table row searched for its heading ("as a percentage of revenue")
CALL_CHARS = 10_000  # about 2,800 tokens per call: the windows, the prompt and the reply fit in 8,192
SEPARATOR = "\n[...]\n"
MAX_CUSTOMERS = 12   # per call: a model stuck repeating one customer runs out of tokens otherwise

_MENTION = re.compile(r"\b(?:customers?|clients?|purchasers?|distributors?|wholesalers?)\b", re.I)
_SHARE_CLAUSE = re.compile(r"(?:accounted for|represented|comprised)\s+(?:approximately\s+|about\s+)?\d", re.I)
_NEAR = re.compile(r"\b10\s?%|\bten percent|\b10 percent|accounted for|represent|largest", re.I)
_PERCENT = re.compile(r"\d\s?%|percent", re.I)
_REVENUE = re.compile(r"revenue|sales", re.I)

SHARE_OF = ("total revenue", "segment revenue", "other")

EXTRACT_SYSTEM = """\
You read excerpts from one company's annual report (10-K). Find the company's major customers: each \
customer the excerpts say accounted for a stated percentage of the company's revenue or sales.

For each one, give:
- "customer": the customer's name exactly as written ("Walmart Inc."), or the label used when it is \
not named ("Customer A", "one customer").
- "named": true if the excerpt names the customer, false for labels such as "Customer A", "one \
customer" or "our largest distributor".
- "percent": its share in the most recent year given, as a number (16 for 16%); 0 if the excerpt gives \
only a threshold ("more than 10%").
- "share_of": "total revenue" if it is a share of the company's total revenue or net sales; "segment \
revenue" if a share of one segment's, business's or product line's revenue; "other" for anything else \
(receivables, purchases, a region's sales).
- "year": the fiscal year of that percentage as written ("2023"), or "" if not stated.
- "quote": the sentence or table row that gives the customer and its percentage, copied word for word.

List a customer only if the excerpt states its share of revenue or sales. Leave out:
- statements that no customer reached a threshold ("no single customer accounted for 10% or more"): \
these are not customers;
- countries, regions, markets, segments, products, sales channels and customer types ("national \
accounts", "customers in Japan");
- groups ("our five largest customers");
- shares of receivables, purchases, volumes or loans (unless the same sentence also gives a share of \
revenue or sales, which is the one to report);
- the company's suppliers and its own subsidiaries.

List each customer once, with its most recent share.

Also give "no_major_customer": true if the excerpts state that no single customer accounted for 10% or \
more of revenue or sales, otherwise false.

Reply with JSON only. If the excerpts give no customer's share of revenue or sales, return an empty \
list."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "customers": {"type": "array", "maxItems": MAX_CUSTOMERS, "items": {
            "type": "object",
            "properties": {
                "customer": {"type": "string"},
                "named": {"type": "boolean"},
                "percent": {"type": "number"},
                "share_of": {"type": "string", "enum": list(SHARE_OF)},
                "year": {"type": "string"},
                "quote": {"type": "string"},
            },
            "required": ["customer", "named", "percent", "share_of", "year", "quote"],
        }},
        "no_major_customer": {"type": "boolean"},
    },
    "required": ["customers", "no_major_customer"],
}


def flatten(text: str) -> str:
    """The filing text on one line: the lake's text puts every table cell on its own line."""
    return " ".join(text.split())


def candidate_windows(text: str) -> list[tuple[int, int]]:
    """Spans of ``text`` (flattened) around each mention of a customer, or a share-of-sales
    clause, with a percentage and "revenue" or "sales" nearby; overlapping spans merged."""
    spans = []
    for m in (*_MENTION.finditer(text), *_SHARE_CLAUSE.finditer(text)):
        near = text[max(0, m.start() - CONTEXT_CHARS): m.end() + CONTEXT_CHARS]
        if _NEAR.search(near) and _PERCENT.search(near) and _REVENUE.search(near):
            spans.append((max(0, m.start() - BEFORE_CHARS), min(len(text), m.end() + AFTER_CHARS)))
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def group_windows(text: str, spans: Iterable[tuple[int, int]], limit: int = CALL_CHARS) -> list[str]:
    """The windows' text grouped into calls of at most ``limit`` characters (a longer window
    is split)."""
    pieces = [text[a + k: min(b, a + k + limit)] for a, b in spans for k in range(0, b - a, limit)]
    groups: list[list[str]] = []
    size = 0
    for p in pieces:
        if not groups or size + len(p) + len(SEPARATOR) > limit:
            groups.append([])
            size = 0
        groups[-1].append(p)
        size += len(p) + len(SEPARATOR)
    return [SEPARATOR.join(g) for g in groups]


def extract_messages(excerpts: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}"}]


def extract_version() -> str:
    """Short hash of the prompt and the windowing, for naming cached replies."""
    parts = [EXTRACT_SYSTEM, EXTRACT_SCHEMA, BEFORE_CHARS, AFTER_CHARS, CONTEXT_CHARS, CALL_CHARS,
             _MENTION.pattern, _SHARE_CLAUSE.pattern, _NEAR.pattern]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8]


def _percent_in(percent: float, quote: str) -> bool:
    """Does the quote state ``percent`` (16 -> "16%", "16 percent", "16.0 %"; a whole
    number may be the quote's figure rounded, 18 -> "18.4 %")?"""
    numbers = {float(x) for x in re.findall(r"\d+(?:\.\d+)?(?=\s?%|\s+percent)", quote)}
    tolerance = 0.5 if percent == int(percent) else 0.051
    return any(abs(n - percent) <= tolerance for n in numbers)


def revenue_context(quote: str, excerpts: str) -> str:
    """The text that says what a quoted share is a share of: the quote, or, for a quote
    without "revenue" or "sales" (a table row: "McKesson, Inc. 17 % 8 % 9 %"), the
    ``LEAD_IN_CHARS`` before it in the excerpts (the table's heading) followed by the quote,
    compacted as ``verify_quote`` compares them."""
    if _REVENUE.search(quote):
        return quote
    text, q = _compact(excerpts), _compact(quote_key(quote))
    i = text.find(q)
    return text[max(0, i - LEAD_IN_CHARS): i] + q if i >= 0 else quote


_PURCHASES = re.compile(r"\bpurchas(?:e|es|ed)\b|\bmerchandise\b|\bvendors?\b|\bsuppliers?\b", re.I)
_ANY_SHARE = re.compile(r"\d\s?%|\d\s+percent|\bten percent", re.I)
# A table heading's base that is not the whole company's revenue ("total gross product revenue",
# "international site leasing revenue"), read just before the heading's last "revenue"/"sales".
_PART_BASE = re.compile(r"international|domestic|segment|product|leasing|commodit")
HEADING_CHARS = 60
# A share stated for a group of customers ("our top five customers accounted for approximately
# 27% of net sales: BNSF Railway, ..."), unless the quote gives each member's own share ("three
# wholesalers ... each accounted for between 14 percent and 21 percent", "... respectively").
_CUSTOMERS = r"(?:customers|clients|purchasers|distributors|wholesalers|wholesale distributors)"
_GROUP = re.compile(
    r"\b(?:(?:top|largest|biggest)\s+(?:\w+\s+)?|(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:largest\s+|biggest\s+|major\s+|principal\s+)?)" + _CUSTOMERS + r"\b|\bin (?:the )?aggregate\b|\baggregate of\b",
    re.I)
_EACH = re.compile(r"\beach\b|\brespectively\b|\bindividually\b|\bseparately\b", re.I)
_OWN_PARTS = re.compile(r"\b(?:its|their)\s+(?:agencies|affiliates|subsidiaries|divisions)\b", re.I)
_NEXT_SHARE = re.compile(r"\s*(?:%|percent)?\s*(?:,|and|,\s*and)\s*(?:approximately\s+)?\d+(?:\.\d+)?\s?(?:%|percent)", re.I)
_SENTENCE_END = re.compile(r"[.;]\s+(?=[A-Z])")
GROUP_CHARS = 150  # how far after the group's words its share may be


def group_share(percent: float, quote: str, name: str = "") -> bool:
    """Is ``percent`` the share of a group of customers the quote names, not one customer's:
    the first percentage after a group's words ("top five customers", "in the aggregate"),
    within ``GROUP_CHARS`` and the same sentence, and not the first of a list of members' own
    shares ("two customers accounted for 19 percent and 17 percent"). Not when the quote gives
    each member's share ("each", "respectively"), the share right after the customer's name is
    its own (a table row under a heading about "the three largest wholesalers": "McKesson
    Corporation 24 % 22 % 21 %"), or the aggregate is of one customer's parts ("the U.S.
    government and its agencies represented in the aggregate approximately 10%")."""
    if percent <= 0 or _EACH.search(quote):
        return False
    at = quote.lower().find(name.lower()) if name else -1
    if at >= 0:
        own = _SHARE.search(quote, at + len(name), at + len(name) + 20)
        if own and abs(float(own.group()) - percent) <= 0.5:
            return False
    for m in _GROUP.finditer(quote):
        if "aggregate" in m.group().lower() and _OWN_PARTS.search(quote):
            continue
        first = _SHARE.search(quote, m.end(), m.end() + GROUP_CHARS)
        if not first or abs(float(first.group()) - percent) > 0.5:
            continue
        if _SENTENCE_END.search(quote, m.end(), first.start()) or _NEXT_SHARE.match(quote, first.end()):
            continue
        return True
    return False


def check_customer(c: dict, excerpts: str) -> str | None:
    """Why a returned customer is dropped, or None to keep it: the quote must be in the text,
    a named customer's name must be in the quote, a stated percentage must be in the quote
    (and at most 100), a share without a percentage needs one in the quote ("10% or more"), the quote
    must mention revenue or sales, and not purchases from suppliers ("our five suppliers
    represented 55% of merchandise purchased"), and the percentage must not be a group's
    (``group_share``)."""
    name, quote = " ".join(str(c.get("customer", "")).split()), " ".join(str(c.get("quote", "")).split())
    percent = float(c.get("percent") or 0)
    if not name or not verify_quote(quote, excerpts):
        return "not_in_text"
    if c.get("named") and quote_key(name) not in quote_key(quote):
        return "name_not_in_quote"
    if percent > 100 or (percent > 0 and not _percent_in(percent, quote)):
        return "percent_not_in_quote"
    if percent == 0 and not _ANY_SHARE.search(quote):
        return "no_share_in_quote"
    if not _REVENUE.search(revenue_context(quote, excerpts)):
        return "no_revenue_in_quote"
    if _PURCHASES.search(quote):
        return "purchases"
    if group_share(percent, quote, name):
        return "group_share"
    return None


def _table_base(context: str) -> str | None:
    """For a table row's ``revenue_context``: "other" if the nearer of the heading's words is
    "receivable", "segment revenue" if the heading's base is part of the company (product,
    segment, international), else None."""
    last = {w: max((m.end() for m in re.finditer(w, context)), default=-1) for w in ("receivable", "revenue|sales")}
    if last["receivable"] > last["revenue|sales"]:
        return "other"
    start = max(m.start() for m in re.finditer("revenue|sales", context))
    return "segment revenue" if _PART_BASE.search(context[max(0, start - HEADING_CHARS): start]) else None


_SHARE = re.compile(r"\d+(?:\.\d+)?(?=\s?%|\s+percent)")


def align_respectively(items: list[dict]) -> int:
    """Customers quoted together in "A and B represented 18% and 21% of revenue,
    respectively" get the percentage at their name's place in the list; the model sometimes
    gives both the first one. Only when the quote has exactly one percentage per name between
    the last name and "respectively". Returns how many percentages were changed."""
    changed = 0
    by_quote: dict[str, list[dict]] = {}
    for item in items:
        if item["named"] and item["percent"] > 0 and "respectively" in item["quote"].lower():
            by_quote.setdefault(item["quote"], []).append(item)
    for quote, group in by_quote.items():
        low = quote.lower()
        places = [(low.find(item["customer"].lower()), item) for item in group]
        if len(group) < 2 or any(p < 0 for p, _ in places):
            continue
        places.sort(key=lambda x: x[0])
        last = places[-1][0] + len(places[-1][1]["customer"])
        shares = [float(x) for x in _SHARE.findall(quote[last: low.find("respectively", last)])]
        if len(shares) != len(places):
            continue
        for (_, item), share in zip(places, shares):
            if item["percent"] != share:
                item["percent"] = share
                changed += 1
    return changed


def _rank(item: dict) -> tuple:
    """Which of two entries for one customer to keep: a share of total revenue, then the
    latest year, then the larger percentage."""
    year = re.search(r"(?:19|20)\d\d", item["year"])
    return item["share_of"] == "total revenue", int(year.group()) if year else 0, item["percent"]


def collect_customers(replies: list[tuple[str, dict | None]]) -> tuple[list[dict], dict[str, int]]:
    """Customers from one filing's replies, each ``(excerpts, parsed reply)``, checked by
    ``check_customer`` against the excerpts the call saw; one entry per customer name (a
    total-revenue share preferred, then the latest year, then the larger percentage), and counts of what was
    dropped. A share of total or segment revenue whose quote is about receivables and not
    revenue is moved to "other"; a table row under a heading for part of the company's revenue
    becomes "segment revenue"; percentages listed with "respectively" are lined up with the
    names (``align_respectively``)."""
    counts = Counter(returned=0, unparsed=0)
    best: dict[str, dict] = {}
    for excerpts, reply in replies:
        if reply is None:
            counts["unparsed"] += 1
            continue
        for c in reply.get("customers", []):
            counts["returned"] += 1
            problem = check_customer(c, excerpts)
            if problem:
                counts[problem] += 1
                continue
            name = " ".join(str(c["customer"]).split())
            quote = " ".join(str(c["quote"]).split())
            share_of = c.get("share_of") if c.get("share_of") in SHARE_OF else "other"
            context = revenue_context(quote, excerpts).lower()
            if context != quote.lower():  # a table row: its heading decides
                base = _table_base(context)
                if base == "other" or (base and share_of != "other"):
                    share_of = base
            elif re.search(r"receivable", quote, re.I) and not re.search(
                    r"revenue|net sales", quote.lower().split("receivable")[0]):
                share_of = "other"
            item = {"customer": name, "named": bool(c.get("named")), "percent": float(c.get("percent") or 0),
                    "share_of": share_of, "year": str(c.get("year", "")), "quote": quote}
            key = core_name(name) if item["named"] else quote_key(name)  # "BlackRock" = "BlackRock, Inc."
            if key not in best or _rank(item) > _rank(best[key]):
                best[key] = item
    customers = list(best.values())
    if moved := align_respectively(customers):
        counts["realigned"] = moved
    return customers, dict(counts)


# --------------------------------------------------------------------------- #
# Names to tickers
# --------------------------------------------------------------------------- #

_TRAILING = re.compile(r"\b(?:and|&)\s+(?:its|their)\s+(?:affiliates|subsidiaries|affiliated companies)\b.*$", re.I)
_STATE_TAG = re.compile(r"(?:\s*/\s*[a-z]{0,4})+/?\s*$", re.I)  # SEC titles: "VALERO ENERGY CORP/TX", "/DE/"


def core_name(name: str) -> str:
    """Lower case, legal suffixes, punctuation, SEC's state tags and "and its affiliates"
    removed: "Lowe's Companies, Inc." -> "lowes"."""
    name = _TRAILING.sub("", _STATE_TAG.sub("", name)).replace("'", "").replace("’", "")
    return " ".join(_SUFFIX.sub(" ", name).lower().split())


# Customer names that no listed company's name, current or former, begins: a subsidiary named
# for itself, a short name, or a company SEC's current list has dropped. The keys are the
# leading words of a customer's core name without spaces or punctuation; the values are
# (ticker, CIK). From the hand check of the 2024 10-Ks' unmatched names. Foreign companies
# without a U.S. listing (Samsung, LG, Xiaomi, Volkswagen) are left out.
ALIASES: dict[str, tuple[str, int]] = {
    "walgreensbootsalliance": ("WBA", 1618921),  # taken private in 2025, off SEC's current list
    "wba": ("WBA", 1618921),
    "evernorth": ("CI", 1739940),
    "expressscripts": ("CI", 1739940),
    "optumrx": ("UNH", 731766),
    "mclane": ("BRK-B", 1067983),
    "bessemedical": ("COR", 1140859),
    "tmobile": ("TMUS", 1283699),
    "verizon": ("VZ", 732712),
    "delta": ("DAL", 27904),
    "ford": ("F", 37996),
    "google": ("GOOGL", 1652044),
    "caesars": ("CZR", 1590895),
    "valero": ("VLO", 1035002),
    "dktrading": ("DK", 1694426),
    "enterprisecrudeoil": ("EPD", 1061219),
    "equinor": ("EQNR", 1140625),
    "occidental": ("OXY", 797468),
    "synnex": ("SNX", 1177394),
    "claro": ("AMX", 1129137),
}

# Customer names whose leading words are a listed company's short name but which are another
# company, or no company: (ticker, CIK) of the company meant, or None. Checked before any other
# match, longest key first; keys as in ``ALIASES``. From a review of every prefix match, and every
# link to a company named by common words, in the 2018-2025 10-Ks.
SEPARATE: dict[str, tuple[str, int] | None] = {
    "cocacolaeuropean": ("CCEP", 1650107),    # Coca-Cola European Partners (Europacific since 2021), not KO
    "cocacolaeuropacific": ("CCEP", 1650107),
    "cocacolabottlerssales": None,            # the bottlers' purchasing cooperative, not KO
    "sunocologistics": ("ET", 1276187),       # merged into Energy Transfer in 2017; not Sunoco LP (SUN)
    "emerging": None,                         # "Emerging Markets" regions and funds, not Emerging Holdings
    "metronewyork": None,                     # a region, not Metro Inc.
    "lightmanufacturing": None,               # an industry, not Light S.A.
    "amerenillinois": None,                   # Ameren's own subsidiary
}


class TickerIndex:
    """SEC's list of listed companies by core name, for matching customer names."""

    def __init__(self, companies: Iterable[tuple[str, str, int]]):
        """``companies``: (ticker, title, cik), primary listing first (SEC's file order)."""
        self.by_core: dict[str, tuple[str, int]] = {}
        for ticker, title, cik in companies:
            core = core_name(title)
            if len(core) >= 3:
                self.by_core.setdefault(core, (ticker, int(cik)))

    def add_former_names(self, names: Iterable[tuple[str, int]]) -> None:
        """Earlier names of listed companies, ``(name, cik)`` (from filing headers: "RAYTHEON
        TECHNOLOGIES CORP" for RTX), mapped to the company's current ticker."""
        by_cik = {cik: ticker for ticker, cik in self.by_core.values()}
        for name, cik in names:
            core = core_name(name)
            if int(cik) in by_cik and len(core) >= 3:
                self.by_core.setdefault(core, (by_cik[int(cik)], int(cik)))
        self.__dict__.pop("_compact_cache", None)

    def match_member(self, member: str) -> tuple[str, int] | None:
        """(ticker, cik) for an XBRL customer member ("khc:WalMartStoresIncMember"), comparing
        names without spaces: the member's leading words, longest first, against listed core
        names, or a listed core name starting with the whole member name (T-Mobile US)."""
        words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", _local(member).removesuffix("Member"))
        core = core_name(" ".join(words)).split()
        if not core:
            return None
        compact = self._compact_names()
        for n in range(len(core), 0, -1):
            if "".join(core[:n]) in SEPARATE:
                return SEPARATE["".join(core[:n])]
        for n in range(len(core), 0, -1):
            head = "".join(core[:n])
            if head in compact and (n > 1 or (len(head) >= 4 and head not in _GENERIC_FIRST)):
                return compact[head]
            if head in ALIASES:
                return ALIASES[head]
        whole = "".join(core)
        starts = sorted(k for k in compact if k.startswith(whole))
        return compact[starts[0]] if len(whole) >= 6 and starts else None

    def _compact_names(self) -> dict[str, tuple[str, int]]:
        """Core names without spaces or punctuation ("wal-mart stores" -> "walmartstores")."""
        if not hasattr(self, "_compact_cache"):
            self._compact_cache: dict[str, tuple[str, int]] = {}
            for k, v in self.by_core.items():
                self._compact_cache.setdefault(re.sub(r"[^a-z0-9]", "", k), v)
        return self._compact_cache

    @classmethod
    def from_sec_json(cls, data: dict) -> TickerIndex:
        rows = data.get("data", data)
        return cls((r["ticker"], r["title"], r["cik_str"]) for r in rows.values())

    def match(self, name: str) -> tuple[str, int, str] | None:
        """(ticker, cik, "exact" | "alias" | "prefix") for a customer name, or None. An alias is
        an ``ALIASES`` key as the customer's leading words (longest wins); a prefix match is a
        listed company's core name as the leading words of the customer's (longest wins), not
        a single generic or short word, in a name written with a capital ("joint ventures in
        which we have a controlling interest" is no company). A name in ``SEPARATE`` gets the
        company given there, or none. Names are compared without spaces or punctuation
        ("Wal-Mart Stores" = "Walmart Stores")."""
        compact = self._compact_names()
        words = [re.sub(r"[^a-z0-9]", "", w) for w in core_name(name).split()]
        words = [w for w in words if w]
        for n in range(len(words), 0, -1):
            if "".join(words[:n]) in SEPARATE:
                other = SEPARATE["".join(words[:n])]
                return (*other, "alias") if other else None
        if "".join(words) in compact:
            return (*compact["".join(words)], "exact")
        for n in range(len(words), 0, -1):
            if "".join(words[:n]) in ALIASES:
                return (*ALIASES["".join(words[:n])], "alias")
        first = re.search(r"[A-Za-z]", name)
        if not first or first.group().islower():
            return None
        for n in range(len(words) - 1, 0, -1):
            head = words[:n]
            if n == 1 and (len(head[0]) < 4 or head[0] in _GENERIC_FIRST):
                continue
            if "".join(head) in compact:
                return (*compact["".join(head)], "prefix")
        return None


# --------------------------------------------------------------------------- #
# XBRL check
# --------------------------------------------------------------------------- #

_GROUP_MEMBER = re.compile(r"(?i)customers(?:member)?$|^(?:five|ten|top)|\d+largest|revenuefromcontract")


def _number(text: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.]", "", text))
    except ValueError:
        return None


def xbrl_concentrations(html: bytes) -> list[dict]:
    """Revenue concentrations tagged in a filing's inline XBRL: ``ConcentrationRiskPercentage1``
    facts whose context has a customer member and a revenue or sales benchmark. Each is
    ``{"member", "percent", "start", "end"}`` (percent as 21.0 for 21%)."""
    doc = lxml.html.fromstring(html, parser=lxml.html.HTMLParser(huge_tree=True))
    contexts = _contexts(doc)
    out = []
    for el in doc.iter("ix:nonfraction"):
        if _local(el.get("name", "")).lower() != "concentrationriskpercentage1":
            continue
        ctx = contexts.get(el.get("contextref", ""))
        value = _number(el.text_content())
        if ctx is None or value is None:
            continue
        dims = dict(ctx.dims)
        benchmark = next((m for a, m in dims.items() if "benchmark" in a.lower()), "")
        others = {a: m for a, m in dims.items() if "benchmark" not in a.lower() and "type" not in a.lower()}
        customer = next((m for a, m in others.items() if "customer" in a.lower()), None)
        if customer is None:
            customer = next((m for m in others.values() if "customer" in m.lower()), None)
        if (customer is None or _GROUP_MEMBER.search(_local(customer))
                or not re.search(r"revenue|sales", benchmark, re.I)):
            continue
        scale = int(el.get("scale") or 0)
        out.append({"member": customer, "percent": round(value * 10 ** scale * 100, 2),
                    "start": ctx.start.isoformat() if ctx.start else None, "end": ctx.end.isoformat()})
    return out
