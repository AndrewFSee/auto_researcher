"""
Customer-supplier links from 10-Ks: a feasibility count before any customer-momentum test.

Reads the 10-Ks the sentiq project has stored (its lake's text, one 10-K per company filed in
``--year``), extracts each company's major customers with a local model
(``auto_researcher.llm.customers``), matches the named ones to listed companies (SEC's
company list), and counts how many S&P 500 suppliers have a named, listed customer with 10%
or more of revenue. No returns are computed.

* ``dev``: run the extraction on a few filings from another year and print what it found,
  for writing the prompt (the counted year is run once, after the prompt is fixed).
* ``extract``: every 10-K filed in ``--year``; replies cached in ``data/customers/runs/`` so a
  run can be stopped and resumed.
* ``xbrl``: the revenue concentrations each filing tags in inline XBRL (raw filings indexed by
  ``build_filing_qa.py``), the check on what the text extraction misses.
* ``report``: the counts, the XBRL check and a review sheet for the hand check.

Examples::

    python scripts/customer_links.py dev --year 2023 --n 15
    python scripts/customer_links.py extract --year 2024
    python scripts/customer_links.py xbrl --year 2024
    python scripts/customer_links.py report --year 2024 --out docs/results/customer_links_count.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.customers import (  # noqa: E402
    EXTRACT_SCHEMA,
    TickerIndex,
    _rank,
    candidate_windows,
    collect_customers,
    extract_messages,
    extract_version,
    flatten,
    group_windows,
    xbrl_concentrations,
)
from auto_researcher.llm.moat import chat_json  # noqa: E402

logger = logging.getLogger("customer_links")

DATA = ROOT / "data" / "customers"
SENTIQ = Path(r"C:\Users\andre\projects\sentiq\data")
LAKE = SENTIQ / "lake" / "items" / "src=edgar_periodic"
TICKERS = SENTIQ / "ref" / "company_tickers.json"  # SEC's list, fetched 2026-10-09
FILING_INDEX = ROOT / "data" / "filingqa" / "filing_index.json"
MIN_PERCENT = 10.0
FIRST_LAKE_YEAR = 2018
EARLY = DATA / "early_10k"  # 10-Ks filed 2009-2017, from scripts/download_early_10ks.py


def _iso(period: str | None) -> str | None:
    """The lake's period of report ("20221231") as an ISO date."""
    return f"{period[:4]}-{period[4:6]}-{period[6:8]}" if period and len(period) == 8 else period


def load_10ks(year: int) -> list[dict]:
    """10-Ks filed in ``year`` from sentiq's lake, one per company (the first filed). Before
    2018 (the lake's first year), the filings ``download_early_10ks.py`` stored the same way."""
    if year < FIRST_LAKE_YEAR:
        path = EARLY / f"{year}.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"{path} not found: run scripts/download_early_10ks.py first")
        rows = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                row = {k: r[k] for k in ("ticker", "cik", "accession", "company", "filed", "period")}
                row["text"] = flatten(r["text"])
                if row["cik"] not in rows or row["filed"] < rows[row["cik"]]["filed"]:
                    rows[row["cik"]] = row
        return sorted(rows.values(), key=lambda r: (r["filed"], r["ticker"]))
    table = ds.dataset(LAKE, format="parquet", partitioning="hive").to_table(
        columns=["tickers", "meta", "body_text"],
        filter=pc.starts_with(ds.field("ym"), str(year)) & pc.match_substring(ds.field("title"), " 10-K "))
    rows = {}
    for tickers, meta, body in zip(*(table[c].to_pylist() for c in ("tickers", "meta", "body_text"))):
        m = json.loads(meta)
        if m["form"] != "10-K":
            continue
        row = {"ticker": tickers[0], "cik": int(m["cik"]), "accession": m["accession"], "company": m["company"],
               "filed": m["filing_date"], "period": _iso(m["period"]), "text": flatten(body)}
        if row["cik"] not in rows or row["filed"] < rows[row["cik"]]["filed"]:
            rows[row["cik"]] = row
    return sorted(rows.values(), key=lambda r: (r["filed"], r["ticker"]))


def _parse(raw: str) -> dict | None:
    try:
        reply = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return reply if isinstance(reply, dict) else None


def run_path(year: int, model: str, think: bool = False) -> Path:
    think_tag = "__think" if think else ""
    return DATA / "runs" / f"{year}__{model.replace(':', '_')}__{extract_version()}{think_tag}.jsonl"


def _cached(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    return {(r["accession"], r["group"]): r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def extract(filings: list[dict], path: Path, args) -> None:
    done = _cached(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = [(f, k, g) for f in filings for k, g in enumerate(group_windows(f["text"], candidate_windows(f["text"])))
            if (f["accession"], k) not in done]
    logger.info("%d calls to make (%d cached) -> %s", len(todo), len(done), path.name)
    with path.open("a", encoding="utf-8") as fh:
        for n, (f, k, excerpts) in enumerate(todo, 1):
            raw, seconds = chat_json(args.url, args.model, extract_messages(excerpts), EXTRACT_SCHEMA,
                                    think=args.think)
            fh.write(json.dumps({"accession": f["accession"], "ticker": f["ticker"], "group": k,
                                 "excerpts": excerpts, "reply": raw, "seconds": round(seconds, 1)}) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                logger.info("%d / %d calls (%s, %.0f s)", n, len(todo), f["ticker"], seconds)


def results(filings: list[dict], path: Path) -> list[dict]:
    """Per filing: verified customers, drop counts, and whether a group's reply says no
    customer reached 10%."""
    by_acc: dict[str, list[dict]] = {}
    for r in _cached(path).values():
        by_acc.setdefault(r["accession"], []).append(r)
    out = []
    for f in filings:
        calls = sorted(by_acc.get(f["accession"], []), key=lambda r: r["group"])
        replies = [(c["excerpts"], _parse(c["reply"])) for c in calls]
        customers, counts = collect_customers(replies)
        out.append({**{k: f[k] for k in ("ticker", "cik", "accession", "company", "filed", "period")},
                    "calls": len(calls), "customers": customers, "counts": counts,
                    "no_major_customer": any(r and r.get("no_major_customer") for _, r in replies)})
    return out


def tickers() -> TickerIndex:
    """SEC's current company list, plus the names S&P 500 companies filed under since 2017."""
    index = TickerIndex.from_sec_json(json.loads(TICKERS.read_text(encoding="utf-8")))
    index.add_former_names((r["company"], int(r["cik"])) for r in json.loads(FILING_INDEX.read_text(encoding="utf-8")))
    return index


def cmd_dev(args) -> None:
    filings = [f for f in load_10ks(args.year) if candidate_windows(f["text"])]
    random.Random(args.seed).shuffle(filings)
    filings = sorted(filings[: args.n], key=lambda f: f["ticker"])
    path = DATA / "dev" / run_path(args.year, args.model, args.think).name
    extract(filings, path, args)
    index = tickers()
    for r in results(filings, path):
        print(f"\n== {r['ticker']} ({r['company']}) calls {r['calls']} {r['counts']} "
              f"no_major={r['no_major_customer']}")
        for c in r["customers"]:
            print(f"   {c['customer']!r} named={c['named']} {c['percent']}% {c['share_of']} {c['year']} "
                  f"-> {index.match(c['customer']) if c['named'] else ''}")
            print(f"      \"{c['quote'][:240]}\"")


def cmd_extract(args) -> None:
    filings = load_10ks(args.year)
    logger.info("%d 10-Ks filed in %d; %d with candidate windows", len(filings), args.year,
                sum(bool(candidate_windows(f["text"])) for f in filings))
    extract(filings, run_path(args.year, args.model, args.think), args)


def cmd_xbrl(args) -> None:
    out_path = DATA / f"xbrl_{args.year}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    docs = {r["accession"]: r["document"] for r in json.loads(FILING_INDEX.read_text(encoding="utf-8"))}
    filings = [f for f in load_10ks(args.year) if f["accession"] not in done]
    logger.info("%d filings to read", len(filings))
    for n, f in enumerate(filings, 1):
        doc = docs.get(f["accession"])
        if doc is None or not Path(doc).exists():
            done[f["accession"]] = None
            continue
        with gzip.open(doc, "rb") as fh:
            done[f["accession"]] = xbrl_concentrations(fh.read())
        if n % 50 == 0:
            logger.info("%d / %d", n, len(filings))
            out_path.write_text(json.dumps(done), encoding="utf-8")
    out_path.write_text(json.dumps(done), encoding="utf-8")


def _latest(concs: list[dict], period: str | None) -> list[dict]:
    """Tagged concentrations for the fiscal year the 10-K reports (ends within 20 days of its period)."""
    if not concs or not period:
        return []
    end = date.fromisoformat(period)
    return [c for c in concs if abs((date.fromisoformat(c["end"]) - end).days) <= 20]


def xbrl_pairs(res: list[dict], xbrl: dict, index: TickerIndex) -> dict[tuple[str, str], dict]:
    """(supplier, customer ticker) -> {"member", "percent", "accession"} for the customers a
    filing tags in inline XBRL at 10% or more of revenue, for the fiscal year it reports, whose
    member names a listed company other than the filer (the first tagged percentage)."""
    pairs: dict[tuple[str, str], dict] = {}
    for r in res:
        for c in _latest(xbrl.get(r["accession"]) or [], r["period"]):
            if c["percent"] < MIN_PERCENT:
                continue
            m = index.match_member(c["member"])
            if m and m[1] != r["cik"]:
                pairs.setdefault((r["ticker"], m[0]), {"member": c["member"], "percent": c["percent"],
                                                       "accession": r["accession"]})
    return pairs


def link_rows(res: list[dict], index: TickerIndex) -> list[dict]:
    """One row per customer, with its ticker match and whether it counts as a link: named,
    listed, not the filer itself, 10% or more of total revenue (or a 10% threshold), for the
    fiscal year the filing reports (or the year before, for fiscal years ending by June, which
    retailers name after the prior year); one row per supplier and customer ticker."""
    rows = []
    for r in res:
        fiscal = None
        if r["period"]:
            end = date.fromisoformat(r["period"])
            fiscal = end.year - (end.month <= 6)
        best: dict[str, dict] = {}
        for c in r["customers"]:
            m = index.match(c["customer"]) if c["named"] else None
            year = re.search(r"(?:19|20)\d\d", c["year"])
            current = fiscal is None or (year is not None and int(year.group()) >= fiscal)
            big = c["percent"] >= MIN_PERCENT or c["percent"] == 0
            ok = bool(m and m[1] != r["cik"] and big and current)
            row = {**c, "supplier": r["ticker"], "supplier_cik": r["cik"], "accession": r["accession"],
                   "match": m, "self": bool(m and m[1] == r["cik"]), "current": current,
                   "link": ok and c["share_of"] == "total revenue",
                   "segment_link": ok and c["share_of"] == "segment revenue"}
            if m is None:
                rows.append(row)
            elif m[0] not in best or _rank(row) > _rank(best[m[0]]):
                best[m[0]] = row
        rows.extend(best.values())
    return rows


VERDICT = {"ok": "correct", "company": "wrong company", "number": "wrong % or year", "notrev": "not a revenue customer"}


def blind_check(rows: list[dict], checks: dict) -> list[str]:
    """A blind hand check of a random sample of one year's links, made under the rules of
    ``checks["rules"]``: the checker confirmed the company and typed the share read in the
    passage without seeing the model's; a link is correct if the company is right and the
    share agrees within half a point. Then what the current rules do with the checked links."""
    marked = [c for c in checks["rows"] if c["kind"] == "link"]
    now = {(r["supplier"], r["customer"]): r for r in rows}
    fmt = lambda t: ", ".join(f"{VERDICT[v]} {n}" for v, n in t.most_common())  # noqa: E731
    total = Counter(c["verdict"] for c in marked)
    real = [c for c in marked if c["verdict"] == "ok" or (c["verdict"] == "number" and c["read_share"] in ("threshold", "figure")
                                                          and (c["read_share"] == "threshold" or float(c["read_figure"]) >= 10))]
    kept = [c for c in marked if now.get((c["supplier"], c["customer"]), {}).get("link")
            and now[(c["supplier"], c["customer"])]["match"][0] == c["ticker"]]
    gone = [c for c in marked if c not in kept]
    k_t = Counter(c["verdict"] for c in kept)
    lines = [
        "", "## Blind hand check", "",
        f"A {checks['sample']}, checked by hand on {checks['checked']} under the rules at {checks['rules']}, before any "
        "rule was tuned on this year. The checker saw each passage and the matched ticker but not the model's "
        "percentage; a link is correct if the company is right and the share read agrees within half a point.",
        "",
        f"* Under the rules at {checks['rules']}: {fmt(total)}; precision {total['ok']} of {len(marked)} "
        f"({total['ok'] / len(marked):.0%}). A real link (the listed company, 10% or more of total revenue, whatever the "
        f"exact figure): {len(real)} of {len(marked)}.",
        "* Wrong links and figures: " + "; ".join(
            f"{c['supplier']} → {c['ticker']} ({VERDICT[c['verdict']]}: {c['customer']}, model {c['percent']:g}%, read "
            f"{c['read_figure'] + '%' if c['read_share'] == 'figure' else c['read_share'] or 'n/a'})"
            for c in marked if c["verdict"] != "ok") + ".",
        f"* The rules added after this check (a group's share; names in `SEPARATE`; prefix matches need a capital) keep "
        f"{len(kept)} of the checked links ({fmt(k_t)}) and drop or rematch {len(gone)}: "
        + ("; ".join(f"{c['supplier']} → {c['ticker']} ({VERDICT[c['verdict']]})" for c in gone) or "none")
        + f". Precision of the kept links: {k_t['ok']} of {len(kept)} ({k_t['ok'] / max(len(kept), 1):.0%}), in-sample "
        "for those rules.",
    ]
    return lines


def hand_check(rows: list[dict], checks: dict) -> list[str]:
    """The hand check of an earlier report's links (keyed by supplier and customer as
    written) against the current rules: which checked links the rules keep or drop, and which
    names marked as listed companies now match."""
    marked = {(c["supplier"], c["customer"]): c for c in checks["rows"] if c["kind"] == "link"}
    named = {(c["supplier"], c["customer"]): c for c in checks["rows"] if c["kind"] == "unmatched"}
    now = {(r["supplier"], r["customer"]): r for r in rows}
    total = [k for k, c in marked.items() if not c["segment"]]  # segment-only links were not counted
    kept = [k for k in total if now.get(k, {}).get("link")]
    dropped = [k for k in total if k not in kept]
    tally = lambda keys: Counter(marked[k]["verdict"] for k in keys)  # noqa: E731
    fmt = lambda t: ", ".join(f"{VERDICT[v]} {n}" for v, n in t.most_common())  # noqa: E731
    k_t, d_t = tally(kept), tally(dropped)
    lines = [
        "", "## Hand check", "",
        f"The {len(total)} total-revenue links of the first report were checked by hand on {checks['checked']} against "
        f"the passages the model read: {fmt(tally(total))}. The rules below were then added and tuned on those marks, "
        "so the 2024 figures after them are in-sample; another year needs its own check.",
        "",
        "* Rules added after the check: a quote about purchases from suppliers is dropped; a share without a percentage needs "
        "one in the quote (\"10% or more\"); a table row under a heading for part of the company's revenue (product, segment, "
        "international) is a segment share; percentages in \"A and B … 18% and 21%, respectively\" are lined up with "
        "the names; an alias table maps subsidiaries, short names and delisted companies to tickers.",
        "* Rules added after the blind check of the 2020 10-Ks (`report_2020.md`): a share stated for a group of "
        "customers (\"our top five customers accounted for 27%\") is dropped; names that share a short form with a "
        "listed company but are another firm or none (Coca-Cola European Partners, Sunoco Logistics, \"Emerging "
        "Markets\") get the company meant or none (`SEPARATE`); a prefix match needs a name written with a capital.",
    ]
    # A wrong percentage the rules have corrected counts as correct.
    fixed = [k for k in kept if marked[k]["verdict"] == "number" and now[k]["percent"] != marked[k]["percent"]]
    good = k_t["ok"] + len(fixed)
    lines += [
        f"* Checked links the rules keep: {len(kept)} ({fmt(k_t)}); dropped: {len(dropped)} ({fmt(d_t) or 'none'}).",
        f"* Percentages the rules corrected: " + ("; ".join(
            f"{s} → {c} {marked[(s, c)]['percent']:g}% → {now[(s, c)]['percent']:g}%" for s, c in fixed) or "none") + ".",
        f"* Precision of the kept links, as marked, counting corrected percentages as correct: {good} of {len(kept)} "
        f"({good / max(len(kept), 1):.0%}).",
    ]
    confirm = [(k, marked[k]) for k in total if marked[k].get("to_confirm")]
    if confirm:
        lines.append("* Marks to check again: " + "; ".join(
            f"{s} → {c} (marked {VERDICT[m['verdict']]}): {m['to_confirm']}" for (s, c), m in confirm) + ".")
    wrong_kept = [k for k in kept if marked[k]["verdict"] != "ok" and k not in fixed and not marked[k].get("to_confirm")]
    if wrong_kept:
        lines.append("* Wrong links the rules still keep: " + "; ".join(
            f"{s} → {now[(s, c)]['match'][0]} ({VERDICT[marked[(s, c)]['verdict']]}: {marked[(s, c)]['note']})"
            for s, c in wrong_kept) + ".")

    listed = {k: c for k, c in named.items() if c["verdict"] == "listed"}
    new = [r for r in rows if r["link"] and (r["supplier"], r["customer"]) not in marked]
    agree = [r for r in new if (r["supplier"], r["customer"]) in listed]
    matched, dropped_named, unmatched = [], [], []
    for (s, cust), c in listed.items():
        r = now.get((s, cust))
        if r is None:  # folded into another entry for the same ticker, or dropped by a check
            same = any(x["supplier"] == s and x["match"] and c["listed_as"].upper().startswith(x["match"][0])
                       for x in rows)
            (matched if same else dropped_named).append(f"{s} → {c['listed_as']} ({cust})")
        else:
            (matched if r["match"] else unmatched).append(f"{s} → {c['listed_as']} ({cust})")
    lines += [
        f"* Unmatched names marked as listed companies: {len(listed)} of {len(named)}; now matched {len(matched)}, "
        f"dropped by the new checks {len(dropped_named)} ({'; '.join(dropped_named) or 'none'}), still unmatched "
        f"{len(unmatched)}.",
        f"* New links since the check: {len(new)}, of them {len(agree)} from names marked as listed (ticker agrees in "
        f"{sum(listed[(r['supplier'], r['customer'])]['listed_as'].upper().startswith(r['match'][0]) for r in agree)}"
        f"; their shares are not yet checked); other new links: " + (", ".join(f"{r['supplier']} → {r['match'][0]}" for r in new if r not in agree)
                               or "none") + ".",
        "* Names marked as listed that still match no ticker (foreign listings, the filer's own subsidiaries, "
        "suppliers, a group of three, a company off SEC's current list): " + "; ".join(unmatched) + ".",
    ]
    return lines


def cmd_report(args) -> None:
    filings = load_10ks(args.year)
    path = run_path(args.year, args.model, args.think)
    res = results(filings, path)
    index = tickers()
    rows = link_rows(res, index)
    xbrl_path = DATA / f"xbrl_{args.year}.json"
    xbrl = json.loads(xbrl_path.read_text(encoding="utf-8")) if xbrl_path.exists() else {}
    with_windows = sum(r["calls"] > 0 for r in res)
    counts = Counter()
    for r in res:
        counts.update(r["counts"])
    seconds = sum(c["seconds"] for c in _cached(path).values())

    suppliers = {r["supplier"] for r in rows if r["link"]}
    seg_only = {r["supplier"] for r in rows if r["segment_link"]} - suppliers
    any_customer = {r["ticker"] for r in res if any(c["share_of"] != "other" for c in r["customers"])}
    named = [r for r in rows if r["named"] and r["share_of"] != "other"]
    unmatched = Counter(r["customer"] for r in named if r["match"] is None)

    lines = [
        f"# Customer-supplier links: feasibility count ({args.year} 10-Ks)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/customer_links.py`, model `{args.model}`, prompt "
        f"`{extract_version()}`. No returns were computed.",
        "",
        f"* 10-Ks filed in {args.year} (one per company, sentiq's lake): {len(res)}; with text near a customer "
        f"mention and a percentage of revenue or sales: {with_windows}; model calls {sum(r['calls'] for r in res)} "
        f"({seconds / 3600:.1f} h).",
        f"* Customers returned {counts['returned']}; dropped: quote not in the text {counts['not_in_text']}, named "
        f"customer not in its quote {counts['name_not_in_quote']}, percentage not in the quote "
        f"{counts['percent_not_in_quote']}, no percentage in the quote "
        f"{counts['no_share_in_quote']}, quote without revenue or sales {counts['no_revenue_in_quote']}, "
        f"purchases from suppliers {counts['purchases']}, a group's share {counts['group_share']}; unparseable replies {counts['unparsed']}. Percentages "
        f"lined up with their names in \"…, respectively\" lists: {counts['realigned']}.",
        f"* 10-Ks stating no customer reached 10%: {sum(r['no_major_customer'] for r in res)}.",
        "",
        "## Count",
        "",
        "| | Companies |",
        "| --- | ---: |",
        f"| Any customer with a share of total or segment revenue | {len(any_customer)} |",
        f"| A *named* customer (total or segment revenue) | {len({r['supplier'] for r in named})} |",
        f"| **A named, listed customer with 10%+ of total revenue (a link)** | **{len(suppliers)}** |",
        f"| Only a segment-level link | {len(seg_only)} |",
        "",
        f"Links: {sum(r['link'] for r in rows)} (exact name match "
        f"{sum(r['link'] and r['match'][2] == 'exact' for r in rows)}, alias "
        f"{sum(r['link'] and r['match'][2] == 'alias' for r in rows)}, prefix "
        f"{sum(r['link'] and r['match'][2] == 'prefix' for r in rows)}); "
        f"distinct customers {len({r['match'][0] for r in rows if r['link']})}.",
        "",
        "Customers by kind (total or segment revenue):",
        "",
        f"* named and matched to a listed company: {sum(r['match'] is not None and not r['self'] for r in named)}",
        f"* named, not matched (private, government, foreign, or a name the matcher misses): "
        f"{sum(r['match'] is None for r in named)}",
        f"* not named (\"Customer A\", \"one customer\"): "
        f"{sum(not r['named'] and r['share_of'] != 'other' for r in rows)}",
        "",
        "Most common unmatched names: " + "; ".join(f"{n} ({k})" for n, k in unmatched.most_common(25)),
    ]

    if xbrl:
        tagged = {r["accession"]: _latest(xbrl.get(r["accession"]) or [], r["period"]) for r in res}
        tagged = {a: [c for c in cs if c["percent"] >= MIN_PERCENT] for a, cs in tagged.items()}
        has = [r for r in res if tagged.get(r["accession"])]
        found = [r for r in has if any(c["share_of"] != "other" for c in r["customers"])]
        n_tags = sum(len({c["member"] for c in tagged[r["accession"]]}) for r in has)
        matched_tags = 0
        for r in has:
            pcts = [c["percent"] for c in r["customers"] if c["share_of"] != "other"]
            members = {}
            for c in tagged[r["accession"]]:
                members.setdefault(c["member"], c["percent"])
            matched_tags += sum(any(abs(p - v) <= 1 for p in pcts) for v in members.values())
        missed = [r["ticker"] for r in has if r not in found]
        # Named members that match a listed company: the links XBRL alone gives.
        links = {(r["supplier"], r["match"][0]) for r in rows if r["link"] or r["segment_link"]}
        xbrl_links = {k: v["member"] for k, v in xbrl_pairs(res, xbrl, index).items()}
        unmatched_members = Counter(c["member"].split(":")[-1] for r in has for c in tagged[r["accession"]]
                                    if index.match_member(c["member"]) is None)
        both = [k for k in xbrl_links if k in links]
        lines += [
            "",
            "## Check against inline XBRL",
            "",
            f"{sum(v is not None for v in xbrl.values())} filings read; {len(has)} tag a customer with 10% or more of "
            f"revenue for the year reported ({n_tags} customer members).",
            "",
            f"* Filings where the text extraction found a customer: {len(found)} of {len(has)}.",
            f"* Tagged customers whose percentage (within 1 point) the extraction found: {matched_tags} of {n_tags}.",
            f"* Missed filings: {', '.join(sorted(missed)) or 'none'}.",
            f"* Tagged members naming a listed company (supplier, customer pairs): {len(xbrl_links)}; found as links "
            f"in the text: {len(both)}. Not found: "
            + (", ".join(f"{s} → {t} ({xbrl_links[(s, t)].split(':')[-1]})" for s, t in sorted(set(xbrl_links) - set(both)))
               or "none") + ".",
            f"* Text links in filings that tag customers but not that one: "
            + (", ".join(f"{s} → {t}" for s, t in sorted(links - set(xbrl_links))
                         if any(r["ticker"] == s for r in has)) or "none") + ".",
            "* Tagged members not matched to a listed company (most are unnamed: Customer A, one customer): "
            + "; ".join(f"{m} ({k})" for m, k in unmatched_members.most_common(30)) + ".",
        ]

    check_path = DATA / f"hand_check_{args.year}.json"
    checks = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else None
    if checks:
        lines += blind_check(rows, checks) if checks.get("blind") else hand_check(rows, checks)

    verdict = {(c["supplier"], c["customer"]): c["verdict"] for c in (checks or {}).get("rows", []) if c["kind"] == "link"}
    lines += ["", "## Links", "", "| Supplier | Customer (as written) | Ticker | Match | % | Share of | Year | Hand check |",
              "| --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for r in sorted((r for r in rows if r["link"] or r["segment_link"]), key=lambda r: r["supplier"]):
        lines.append(f"| {r['supplier']} | {r['customer']} | {r['match'][0]} | {r['match'][2]} | "
                     f"{r['percent']:g} | {r['share_of']} | {r['year']} | "
                     f"{VERDICT.get(verdict.get((r['supplier'], r['customer'])), 'not checked')} |")
    out = Path(args.out) if args.out else DATA / f"report_{args.year}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATA / f"customers_{args.year}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    logger.info("wrote %s (%d suppliers with a link)", out, len(suppliers))


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("dev", "extract", "xbrl", "report"):
        p = sub.add_parser(name)
        p.add_argument("--year", type=int, default=2024)
        p.add_argument("--model", default="qwen3:8b")
        p.add_argument("--url", default="http://localhost:11434")
        p.add_argument("--think", action="store_true", help="let the model reason first (slower)")
    sub.choices["dev"].add_argument("--n", type=int, default=15)
    sub.choices["dev"].add_argument("--seed", type=int, default=0)
    sub.choices["report"].add_argument("--out")
    args = parser.parse_args()
    {"dev": cmd_dev, "extract": cmd_extract, "xbrl": cmd_xbrl, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
