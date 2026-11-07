"""
Tag the sources of an economic moat in a 10-K's business section (Item 1).

Two passes with a local model (Ollama, JSON output constrained by a schema):

1. **Extract**, per chunk of Item 1: quotes that are evidence for one of five moat
   sources (the Morningstar taxonomy: intangible assets, switching costs, network
   effects, cost advantage, efficient scale) or evidence *against* a moat. Every quote
   is checked word for word against the chunk (``verify_quote``); quotes not found are
   dropped and counted, so the rating rests only on text the filing contains.
2. **Rate**, per company: the verified quotes, numbered, and the opening of Item 1 (what
   the business is). Each source gets 0 (none or generic), 1 (specific but modest) or
   2 (strong, specific, durable), citing quote numbers, and the company an overall
   none / narrow / wide.

A Claude model does both in one call over the whole of Item 1 (``one_call_params``;
quotes numbered in the reply, checked by ``verify_evidence``). The local 8B model quotes
well but rates badly (it credits mechanisms its own reasons deny), so the pilot's tags
come from Claude Sonnet 5.5; see ``docs/results/moat_pilot.md``.

The company's name and ticker are replaced by "the Company" (``anonymize``) and the
model is told to use only the text, because it knows how these companies fared after
the filing. Product and brand names still identify many companies, so this reduces
the lookahead risk but does not remove it.

Prompt ``v2`` (``ONE_CALL_SYSTEM_V2``) adds the company's financial history as filed up
to the 10-K (``financial_table``) and a step that judges returns on capital before the
overall call; the sources are still rated from quotes only. It was written after the
hand check of the ``v1`` tags, which found them too low mostly where the moat shows in
returns that Item 1 does not mention.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

SOURCES = {
    "intangibles": "Intangible assets: a brand customers pay more for or seek out, patents, "
                   "regulatory approvals or exclusive licences that keep competitors out.",
    "switching_costs": "Switching costs: customers would lose time, money or take a risk by "
                       "moving to a competitor (deep integration, retraining, long contracts, "
                       "data or certification tied to the product).",
    "network_effects": "Network effects: the product becomes more valuable to each user as more "
                       "users, merchants, developers or partners join (marketplaces, payment "
                       "networks, platforms, exchanges).",
    "cost_advantage": "Cost advantage: structurally lower costs than competitors from scale, "
                      "proprietary processes, unique resources, reserves or location.",
    "efficient_scale": "Efficient scale: a market that supports only one or a few profitable "
                       "firms, so a new entrant would not earn a return (pipelines, regulated "
                       "utilities, railroads, airports, niche industrial markets).",
}
AGAINST = ("against: evidence that the company lacks a moat: commodity products, competition "
           "mainly on price, many similar competitors, customers who can switch easily, "
           "dependence on a few customers or suppliers with bargaining power.")
KINDS = (*SOURCES, "against")
RATINGS = (0, 1, 2)
MOATS = ("none", "narrow", "wide")

CHUNK_CHARS = 12_000  # about 3,000 tokens: the chunk, the prompt and the reply fit in 8,192
MAX_QUOTES = 8        # per chunk (extract) and per kind (rate)
MIN_QUOTE_CHARS = 20
OPENING_CHARS = 1500

EXTRACT_SYSTEM = f"""\
You read part of the business section of a company's annual report (10-K) and pick out \
evidence about whether the company has a durable competitive advantage (an economic moat). \
The company's name is replaced by "the Company". Use only the text given; ignore anything \
you may know about the company.

Kinds of evidence:
{chr(10).join(f"- {k}: {v}" for k, v in SOURCES.items())}
- {AGAINST}

Rules:
- Quote one sentence, or part of one, exactly as written, word for word. Do not paraphrase, \
shorten the middle, join sentences or correct the wording.
- A quote counts only if it says why competitors cannot easily take this business: for \
example sole-source or exclusive positions, a market share or rank with figures, a large \
installed base that buys parts, service or upgrades from the company, multi-year or \
renewing contracts, products customers build into their own systems or workflows, a \
regulated or franchised service territory, patents on named key products with expiry \
dates, users or partners whose number makes the product more useful, costs or scale \
compared with competitors, owned resources or locations others cannot get.
- These do not count: descriptions of products, customers, segments or facilities; size, \
revenue or backlog; R&D spending or headcount; statements that the company owns \
trademarks, copyrights, patents or domain names in general; claims almost any company \
makes ("we face intense competition", "we believe our brand is valuable").
- "against" needs evidence about competition: commodity products or prices set by the \
market, competition mainly on price, low barriers to entry, customers who switch easily. \
General risks (supply chains, regulation, the economy) are not evidence against a moat.
- At most {MAX_QUOTES} items. If the text has no such evidence, return an empty list."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"evidence": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": list(KINDS)}, "quote": {"type": "string"}},
        "required": ["kind", "quote"]}}},
    "required": ["evidence"],
}

RATE_SYSTEM = f"""\
You judge whether a company has an economic moat: a competitive advantage that lets it earn \
high returns on capital for many years. You get the opening of its business description \
and numbered quotes from its annual report (10-K), grouped by the kind of evidence. The \
company's name is replaced by "the Company". Use only this text; ignore anything you may \
know about the company.

For each source of advantage, first explain in a sentence or two what the quotes show, \
then cite the numbers of the quotes the rating rests on (any group; the grouping is only a \
first guess), then rate it:
- 0: no quote states this mechanism. Quotes that only suggest it, could imply it, or \
are generic count as 0.
- 1: a quote states this mechanism directly, but the advantage looks modest, narrow or \
uncertain
- 2: quotes state this mechanism directly and show a large advantage that should last ten \
years or more

Rate a source above 0 only if a quote shows that mechanism itself. Being large, well known, \
innovative, or a leader is not by itself a source; a loyalty program or a broad product \
range is not a network effect; investment in technology is not a cost advantage; a large \
backlog is not efficient scale. If your explanation says the evidence is weak, indirect or \
not really this kind of advantage, the rating is 0.

Then the overall moat: "wide" (advantages should last 20 years or more), "narrow" (about \
10 years) or "none", after weighing the evidence against a moat.

Sources:
{chr(10).join(f"- {k}: {v}" for k, v in SOURCES.items())}"""

# The reply is generated in property order: the reason comes before the rating it justifies.
_SOURCE_RATING = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "quotes": {"type": "array", "items": {"type": "integer"}},
                   "rating": {"type": "integer", "enum": list(RATINGS)}},
    "required": ["reason", "quotes", "rating"],
}
RATE_SCHEMA = {
    "type": "object",
    "properties": {**{k: _SOURCE_RATING for k in SOURCES},
                   "overall": {"type": "object",
                               "properties": {"reason": {"type": "string"},
                                              "moat": {"type": "string", "enum": list(MOATS)}},
                               "required": ["reason", "moat"]}},
    "required": [*SOURCES, "overall"],
}

_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|"
                     r"llc|lp|l\.p|n\.v|s\.a|ag|se|holdings?|group|the|de|md|nv|new)\b\.?|/[a-z]{2}/|[,.&]",
                     re.I)
_LEFT_SUFFIX = re.compile(r"the Company,?\s+(?:Inc|Incorporated|Corp|Corporation|Ltd|Limited|plc|N\.V|S\.A)\b\.?(?!\w)")
_PAREN = re.compile(r"\(([^()]{0,400})\)")
_QUOTED = re.compile(r"[“\"]([^”\"]{2,40}?)[”\"]")
_SELF = {"we", "us", "our", "the company", "company", "u.s"}
_URL = re.compile(r"(?:https?://)?\b(?:[\w-]+\.)+(?:com|net|org|io)\b(?:/[^\s]*)?", re.I)
_NOT_COMPARED = re.compile(r"[\s®™℠©|]")
_GENERIC_FIRST = {"american", "general", "united", "first", "national", "international", "global",
                  "southern", "northern", "eastern", "western", "pacific", "atlantic", "public",
                  "consolidated", "royal", "digital", "universal", "advanced", "applied", "regions",
                  "citizens", "federal", "texas", "republic", "dominion"}
_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})


def name_variants(company: str) -> list[str]:
    """The conformed SEC name and its core without legal suffixes ("MICRON TECHNOLOGY INC"
    -> ["MICRON TECHNOLOGY INC", "MICRON TECHNOLOGY"]), longest first."""
    core = " ".join(_SUFFIX.sub(" ", company).split())
    variants = {company.strip(), core}
    return sorted((v for v in variants if len(v) >= 3), key=len, reverse=True)


def short_name(company: str) -> str | None:
    """The first word of a name's core ("Micron"), when the core has more words and the first
    is not a generic one ("American", "General")."""
    words = " ".join(_SUFFIX.sub(" ", company).split()).split()
    if len(words) < 2 or len(words[0]) < 4 or words[0].lower() in _GENERIC_FIRST:
        return None
    return words[0].capitalize()


def self_aliases(text: str, head_chars: int = 3000) -> list[str]:
    """Names a filing defines for itself in its opening, as in (“C.H. Robinson,” “we,” “us”
    or “our”): the quoted terms of the first parenthesis that also defines "we" or "the
    Company"."""
    for p in _PAREN.finditer(text[:head_chars]):
        terms = [t.strip(" ,.") for t in _QUOTED.findall(p.group(1))]
        if any(t.lower() in ("we", "the company") for t in terms):
            return [t for t in terms if t.lower() not in _SELF and len(t) >= 3]
    return []


def anonymize(text: str, company: str, tickers: list[str]) -> str:
    """Replace the company's name (``name_variants``, any case), its short name and the names
    it defines for itself (``self_aliases``; both as written, so the same words in prose
    survive), its tickers (as words, upper case only, so "ON" or "ALL" in prose survive)
    with "the Company", and web addresses with "[website]"."""
    poss = r"(?:['’]s)?(?![\w])"
    aliases = self_aliases(text)
    text = _URL.sub("[website]", text)
    for v in name_variants(company):
        words = [re.escape(w) for w in v.split()]
        text = re.sub(r"(?<![\w])" + r"[\s,.&]+".join(words) + poss, "the Company", text, flags=re.I)
    short = short_name(company)
    for form in sorted({*aliases, *((short, short.upper()) if short else ())}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(form)}{poss}", "the Company", text)
    for t in tickers:
        text = re.sub(rf"(?<![\w.]){re.escape(t)}(?![\w])", "the Company", text)
    return _LEFT_SUFFIX.sub("the Company", text)


def chunk_lines(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on line breaks (one paragraph per line) into chunks of at most ``max_chars``;
    a single longer line is cut at sentence ends."""
    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        while len(line) > max_chars:
            cut = line.rfind(". ", 0, max_chars) + 1 or max_chars
            pieces.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            pieces.append(line)
    chunks, current = [], ""
    for p in pieces:
        if current and len(current) + 1 + len(p) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n{p}" if current else p
    return chunks + [current] if current else chunks


def normalize(text: str) -> str:
    return " ".join(text.translate(_TRANSLATE).lower().split())


def quote_key(quote: str) -> str:
    return normalize(quote).strip(" .;:\"'")


def _compact(text: str) -> str:
    """Lower case without white space, trademark signs or table bars: the text extracted
    from 10-K HTML puts spaces around "®" and dashes ("Tinder ®", "PSE&G —A") where a
    model quoting it does not."""
    return _NOT_COMPARED.sub("", normalize(text))


def verify_quote(quote: str, text: str) -> bool:
    """Is ``quote`` in ``text`` character for character (ignoring case, white space,
    trademark signs, table bars, curly quotes and dash style, and end punctuation)? Quotes
    under ``MIN_QUOTE_CHARS`` are rejected."""
    q = quote_key(quote)
    return len(q) >= MIN_QUOTE_CHARS and _compact(q) in _compact(text)


def extract_messages(chunk: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Part of the business section:\n\n{chunk}"}]


def collect_evidence(replies: list[tuple[str, dict]]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Verified, de-duplicated quotes by kind (at most ``MAX_QUOTES`` each, in text order)
    from (chunk, parsed reply) pairs, and counts of quotes kept, rejected and duplicated."""
    by_kind: dict[str, list[str]] = {k: [] for k in KINDS}
    seen: set[str] = set()
    counts = {"kept": 0, "not_in_text": 0, "duplicate": 0, "over_cap": 0}
    for chunk, reply in replies:
        for item in reply.get("evidence", [])[:MAX_QUOTES]:
            kind, quote = item.get("kind"), " ".join(str(item.get("quote", "")).split())
            if kind not in by_kind or not verify_quote(quote, chunk):
                counts["not_in_text"] += 1
            elif quote_key(quote) in seen:
                counts["duplicate"] += 1
            elif len(by_kind[kind]) >= MAX_QUOTES:
                counts["over_cap"] += 1
            else:
                seen.add(quote_key(quote))
                by_kind[kind].append(quote)
                counts["kept"] += 1
    return by_kind, counts


def number_quotes(by_kind: dict[str, list[str]]) -> list[tuple[int, str, str]]:
    """(number, kind, quote), numbered from 1 in ``KINDS`` order."""
    flat = [(kind, q) for kind in KINDS for q in by_kind.get(kind, [])]
    return [(i, kind, q) for i, (kind, q) in enumerate(flat, 1)]


def rate_messages(opening: str, numbered: list[tuple[int, str, str]]) -> list[dict[str, str]]:
    groups = []
    for kind in KINDS:
        lines = [f"[{i}] {q}" for i, k, q in numbered if k == kind]
        groups.append(f"{kind}:\n" + ("\n".join(lines) if lines else "(none)"))
    user = (f"Opening of the business description:\n{opening[:OPENING_CHARS]}\n\n"
            "Quotes:\n\n" + "\n\n".join(groups))
    return [{"role": "system", "content": RATE_SYSTEM}, {"role": "user", "content": user}]


def check_rating(rating: dict, numbered: list[tuple[int, str, str]]) -> dict:
    """Drop cited numbers that are not quotes for a moat (missing, or "against"), and lower
    to 0 any rating above 0 left without a citation. Any source may cite a quote the
    extract pass put under another source. Returns the cleaned rating and the changes."""
    kind_of = {i: k for i, k, _ in numbered}
    out = {k: rating[k] for k in ("returns", "overall") if k in rating}
    fixes = []
    for kind in SOURCES:
        r = dict(rating[kind])
        cited = list(dict.fromkeys(i for i in r.get("quotes", []) if kind_of.get(i, "against") != "against"))
        if set(cited) != set(r.get("quotes", [])):
            fixes.append(f"{kind}: dropped citations {sorted(set(r['quotes']) - set(cited))}")
        r["quotes"] = cited
        if r["rating"] > 0 and not cited:
            fixes.append(f"{kind}: {r['rating']} -> 0 (no quote for a moat cited)")
            r["rating"] = 0
        out[kind] = r
    return {**out, "fixes": fixes}


MAX_EVIDENCE = 30  # one-call (Claude) path: quotes for the whole of Item 1

ONE_CALL_SYSTEM = f"""\
You read the business section (Item 1) of a company's annual report (10-K) and judge whether \
the company has an economic moat: a competitive advantage that lets it earn high returns on \
capital for many years. The company's name is replaced by "the Company". Use only the text \
given; ignore anything you may know about the company or what happened to it later.

Step 1, evidence. List up to {MAX_EVIDENCE} quotes, numbered from 1 in the order you list \
them, each with its kind:
{chr(10).join(f"- {k}: {v}" for k, v in SOURCES.items())}
- {AGAINST}

Quote rules:
- Quote one sentence, or part of one, exactly as written, word for word. Do not paraphrase, \
shorten the middle, join sentences or correct the wording. Quotes are checked against the \
text and dropped if not found.
- A quote counts only if it says why competitors cannot easily take this business: for \
example sole-source or exclusive positions, a market share or rank with figures, a large \
installed base that buys parts, service or upgrades from the company, multi-year or \
renewing contracts, products customers build into their own systems or workflows, a \
regulated or franchised service territory, patents on named key products with expiry \
dates, users or partners whose number makes the product more useful, costs or scale \
compared with competitors, owned resources or locations others cannot get.
- These do not count: descriptions of products, customers, segments or facilities; size, \
revenue or backlog; R&D spending or headcount; statements that the company owns \
trademarks, copyrights, patents or domain names in general; claims almost any company \
makes ("we face intense competition", "we believe our brand is valuable").
- "against" needs evidence about competition: commodity products or prices set by the \
market, competition mainly on price, low barriers to entry, customers who switch easily. \
General risks (supply chains, regulation, the economy) are not evidence against a moat.

Step 2, ratings. For each of the five sources: a sentence or two on what the quotes show, \
the numbers of the quotes the rating rests on (any kind except "against"), and the rating:
- 0: no quote states this mechanism; quotes that only suggest it or are generic count as 0
- 1: a quote states this mechanism directly, but the advantage looks modest, narrow or \
uncertain
- 2: quotes state this mechanism directly and show a large advantage that should last ten \
years or more
Being large, well known, innovative or a leader is not by itself a source; a loyalty program \
or a broad product range is not a network effect; investment in technology is not a cost \
advantage; a large backlog is not efficient scale.

Step 3, overall: "wide" (advantages should last 20 years or more), "narrow" (about 10 \
years) or "none", after weighing the evidence against a moat."""


def _strict(schema: dict) -> dict:
    """Copy of a schema with ``additionalProperties: false`` on every object (required by
    Claude's structured outputs)."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    return schema


ONE_CALL_SCHEMA = _strict({
    "type": "object",
    "properties": {"evidence": EXTRACT_SCHEMA["properties"]["evidence"], **RATE_SCHEMA["properties"]},
    "required": ["evidence", *RATE_SCHEMA["required"]],
})


_INTRO, _REST = ONE_CALL_SYSTEM.split("\n\nStep 1", 1)
_STEPS_1_2 = _REST.split("\n\nStep 3", 1)[0]
RETURNS_VERDICTS = ("above", "near", "below", "unclear")

ONE_CALL_SYSTEM_V2 = f"""\
{_INTRO}

You also get the company's financial history: up to five fiscal years of growth, margins \
and returns, computed from the figures it filed with the SEC up to this report (nothing \
later). Use it in steps 3 and 4 only; the sources in step 2 still need quotes.

Step 1{_STEPS_1_2}

Step 3, returns. In a sentence or two, say what the financial history shows: return on \
invested capital (ROIC) against a cost of capital of about 8-10%, or for banks, insurers \
and other financial companies return on equity (ROE) and on assets instead, and how \
steady margins and returns have been. Then classify the returns: "above" (clearly and \
consistently above the cost of capital), "near", "below", or "unclear" (too few figures, \
or figures distorted by acquisitions, write-downs or one-off items).

Step 4, overall: "wide" (advantages should last 20 years or more), "narrow" (about 10 \
years) or "none", after weighing the evidence against a moat and the returns. A moat is \
an advantage the text shows, and it should show up in returns:
- Returns consistently above the cost of capital, with at least one source rated 1 or \
more, support a narrow moat even when each source is modest. Wide needs a source rated 2 \
and returns well above the cost of capital.
- Returns near or below the cost of capital for several years count against a moat, \
whatever the text claims; only a strong source outweighs them, such as a regulated \
franchise whose returns are capped by its regulator.
- High returns with no source in the text are not a moat by themselves; say so in the \
reason."""

ONE_CALL_SCHEMA_V2 = _strict({
    "type": "object",
    "properties": {
        "evidence": EXTRACT_SCHEMA["properties"]["evidence"],
        **{k: _SOURCE_RATING for k in SOURCES},
        "returns": {"type": "object",
                    "properties": {"reason": {"type": "string"},
                                   "verdict": {"type": "string", "enum": list(RETURNS_VERDICTS)}},
                    "required": ["reason", "verdict"]},
        "overall": RATE_SCHEMA["properties"]["overall"],
    },
    "required": ["evidence", *SOURCES, "returns", "overall"],
})
# v3: v2's fixed rules moved calls the wrong way on the hand-checked set (high returns +
# a modest source -> narrow for low-margin resellers; one cost of capital for utilities
# and REITs), so returns now corroborate the text instead of deciding the call.
ONE_CALL_SYSTEM_V3 = f"""\
{_INTRO}

You also get the company's financial history: up to five fiscal years of growth, margins \
and returns, computed from the figures it filed with the SEC up to this report (nothing \
later). Use it in steps 3 and 4 only; the sources in step 2 still need quotes.

Step 1{_STEPS_1_2}

Step 3, returns. In a sentence or two, say what the financial history shows. Compare \
return on invested capital (ROIC) with the company's cost of capital: about 8-10% for \
most companies, about 5-7% for regulated utilities and real estate companies, whose \
long-lived assets are financed cheaply. For banks, insurers and other financial \
companies, compare return on equity (ROE) with about 10% instead. Discount returns that \
are inflated by small or negative equity (buybacks, leverage) or by a thin capital base \
with low margins, as for distributors and resellers. Then classify the returns: "above" \
(clearly and consistently above the cost of capital), "near", "below", or "unclear" \
(too few figures, or figures distorted by acquisitions, write-downs or one-off items).

Step 4, overall: "wide" (advantages should last 20 years or more), "narrow" (about 10 \
years) or "none", after weighing the evidence against a moat. Returns corroborate the \
text; they do not replace it:
- High returns do not turn modest scale, contracts or relationships into a moat when the \
text describes a commodity-like or price-driven market.
- Returns well above the cost of capital for all five years make a moat the text states \
plainly more credible, even when each source is rated 1. Very high returns (say above \
25%) sustained for five years, together with a mechanism the text describes but does \
not quantify (a brand, a distribution network, an installed base), can support narrow.
- Returns below the cost of capital for several years count against a moat, unless they \
are the capped returns of a regulated franchise: then the franchise is the moat.
- Wide needs advantages that should last 20 years or more: a source rated 2, or several \
sources that reinforce each other, with returns well above the cost of capital \
throughout."""
PROMPTS = {"v1": (ONE_CALL_SYSTEM, ONE_CALL_SCHEMA), "v2": (ONE_CALL_SYSTEM_V2, ONE_CALL_SCHEMA_V2),
           "v3": (ONE_CALL_SYSTEM_V3, ONE_CALL_SCHEMA_V2)}


def one_call_params(model: str, item1: str, effort: str = "high", max_tokens: int = 16000,
                    prompt: str = "v1", financials: str | None = None) -> dict:
    """Messages API request: the fixed system prompt (cached across companies), Item 1 (and
    for ``v2`` the financial history) as the user turn, adaptive thinking and the JSON
    schema. No sampling parameters (rejected by the current models)."""
    system, schema = PROMPTS[prompt]
    content = f"Business section (Item 1):\n\n{item1}"
    if prompt != "v1":
        content += f"\n\nFinancial history (fiscal years, as filed up to this report):\n\n" \
                   f"{financials or 'Not available.'}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }


# Ratios per fiscal year, from the point-in-time items of ``data.sec_fundamentals``.
# ROIC uses operating income taxed at the 21% US rate over equity + debt - cash.
TAX_RATE = 0.21
# No gross margin: cost of revenue is tagged in under half of company-years and means
# little for utilities and financial companies (a utility showed 97-100%).
RATIOS = ("Revenue growth", "Operating margin", "Net margin", "FCF margin", "ROIC", "ROE", "ROA")


def financial_ratios(years: dict) -> dict[str, dict[str, float | None]]:
    """``years`` maps fiscal year end -> {item: value}, oldest first, including the year
    before the first one shown (for growth and averages). Returns ratios for every year
    but that first."""
    def div(a, b):
        return a / b if a is not None and b not in (None, 0) and not (b < 0) else None

    def avg(a, b):
        """Average of two year-end balances; None unless both are positive (or the prior
        one is missing), since returns on near-zero or negative capital mean nothing."""
        if a is None or a <= 0 or (b is not None and b <= 0):
            return None
        return a if b is None else (a + b) / 2

    def capital(v):
        debt = (None if v.get("debt_current") is None and v.get("debt_noncurrent") is None
                else (v.get("debt_current") or 0) + (v.get("debt_noncurrent") or 0))
        debt = debt if debt is not None else v.get("debt_total")
        if v.get("equity") is None:
            return None
        return v["equity"] + (debt or 0) - (v.get("cash") or 0)

    out = {}
    ends = list(years)
    for prev, end in zip(ends, ends[1:]):
        v, p = years[end], years[prev]
        rev = v.get("revenue")
        fcf = (v["operating_cash_flow"] - (v.get("capex") or 0)) if v.get("operating_cash_flow") is not None else None
        nopat = v["operating_income"] * (1 - TAX_RATE) if v.get("operating_income") is not None else None
        cap = avg(capital(v), capital(p))
        out[end] = {
            "Revenue growth": div(rev, p.get("revenue")) - 1 if div(rev, p.get("revenue")) is not None else None,
            "Operating margin": div(v.get("operating_income"), rev),
            "Net margin": div(v.get("net_income"), rev),
            "FCF margin": div(fcf, rev),
            "ROIC": div(nopat, cap),
            "ROE": div(v.get("net_income"), avg(v.get("equity"), p.get("equity"))),
            "ROA": div(v.get("net_income"), avg(v.get("total_assets"), p.get("total_assets"))),
        }
    return out


def financial_table(ratios: dict[str, dict[str, float | None]]) -> str | None:
    """Markdown table, one row per fiscal year; None if there is nothing to show."""
    if not ratios or all(x is None for r in ratios.values() for x in r.values()):
        return None
    lines = ["| Fiscal year end | " + " | ".join(RATIOS) + " |", "| --- |" + " ---: |" * len(RATIOS)]
    def cell(x):
        return "n/a" if x is None else ">100%" if x > 1 else "<-100%" if x < -1 else f"{x:.0%}"

    for end, r in ratios.items():
        lines.append(f"| {end} | " + " | ".join(cell(r[k]) for k in RATIOS) + " |")
    return "\n".join(lines + ["", "ROIC: operating income after a 21% tax, over average equity + debt - cash. "
                                  "n/a: not reported in the filings' XBRL data, or not meaningful (for example "
                                  "negative or near-zero capital)."])


# A located Item 1 that starts mid-sentence (the span began at a cross-reference) or
# holds MD&A or financial statement headings early is another part of the 10-K.
_LEAD = " \t\r\n​ ﻿.,;:”\"’'"
_HEADING_START = re.compile(r"(?i)^.{0,12}(business|usiness|siness|description|general|overview|item\s*1)")
_WRONG_HEADING = re.compile(
    r"CONSOLIDATED (BALANCE SHEETS?|STATEMENTS? OF)|(?m:^[ \t]*Critical Accounting (Policies|Estimates)[ \t]*$)")


def item1_problem(text: str, window: int = 10_000) -> str | None:
    """Why ``text`` does not look like a 10-K's Item 1, or None if it does."""
    start = " ".join(text[:300].split()).lstrip(_LEAD)
    if start.startswith("the Company"):  # the name, after ``anonymize``
        start = "T" + start[1:]
    if not (start[:1].isupper() or start[:1].isdigit() or start[:1] == "(" or _HEADING_START.match(start)):
        return "starts mid-sentence"
    m = _WRONG_HEADING.search(text[:window])
    return f"'{m.group(0).strip()}' in the first {window:,} characters" if m else None


def verify_evidence(evidence: list[dict], text: str) -> tuple[list[tuple[int, str, str]], dict[str, int]]:
    """The quotes found word for word in ``text``, keeping the reply's numbering (from 1) so
    the ratings' citations still point at them, and counts of quotes kept and dropped."""
    numbered, seen = [], set()
    counts = {"kept": 0, "not_in_text": 0, "duplicate": 0, "over_cap": 0}
    for i, item in enumerate(evidence, 1):
        kind, quote = item.get("kind"), " ".join(str(item.get("quote", "")).split())
        if i > MAX_EVIDENCE:
            counts["over_cap"] += 1
        elif kind not in KINDS or not verify_quote(quote, text):
            counts["not_in_text"] += 1
        elif quote_key(quote) in seen:
            counts["duplicate"] += 1
        else:
            seen.add(quote_key(quote))
            numbered.append((i, kind, quote))
            counts["kept"] += 1
    return numbered, counts


def one_call_version(prompt: str = "v1") -> str:
    system, schema = PROMPTS[prompt]
    if prompt == "v1":
        return _hash(system, schema, MAX_EVIDENCE, MIN_QUOTE_CHARS)
    return _hash(system, schema, MAX_EVIDENCE, MIN_QUOTE_CHARS, TAX_RATE, RATIOS)


def _hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8]


def extract_version() -> str:
    """Short hash of the extract prompt and its settings, for naming cached replies."""
    return _hash(EXTRACT_SYSTEM, EXTRACT_SCHEMA, CHUNK_CHARS, MAX_QUOTES, MIN_QUOTE_CHARS)


def rate_version() -> str:
    return _hash(RATE_SYSTEM, RATE_SCHEMA, OPENING_CHARS)


def chat_json(url: str, model: str, messages: list[dict], schema: dict, think: bool = False,
              num_ctx: int = 8192, retries: int = 5) -> tuple[str, float]:
    """One Ollama chat call with the reply constrained to ``schema``; returns (raw reply,
    seconds). With ``think`` the model reasons first (not returned). Retries server errors
    (CUDA OOM on a shared laptop GPU) after a pause."""
    body = {"model": model, "stream": False, "think": think, "messages": messages, "format": schema,
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": 4096 if think else 2048}}
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{url}/api/chat", json=body, timeout=900)
            resp.raise_for_status()
            break
        except (requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(exc.response, "status_code", None)
            if attempt == retries or (status is not None and status < 500):
                raise
            logger.warning("%s (attempt %d); retrying in %ds", exc, attempt, 15 * attempt)
            time.sleep(15 * attempt)
    return resp.json()["message"]["content"], time.perf_counter() - start
