"""
Moat tagger pilot: tag the sources of an economic moat in 50 companies' 10-Ks, for a
hand check before the tags are used in any test.

``sample`` draws the companies from the 10-Ks the sentiq project has stored (S&P 500
members, filed in ``--year``; Item 1 located by sentiq's ``locate_sections`` and at least
``MIN_ITEM1_CHARS`` long), stratified by GICS sector in proportion to the sector's share,
one filing per company. It also draws a separate ``dev`` set of five companies for
checking the prompts, so the 50 are tagged only once, with prompts fixed beforehand.

``tag`` runs ``auto_researcher.llm.moat``: with an Ollama model, two passes (extract
verified quotes per chunk of Item 1, then rate per company); with a Claude model
(``--model claude-sonnet-5-5``), one call per company that does both. Every reply is
cached in ``data/moat/runs/`` so a run can be resumed and the report rebuilt for free. ``report`` writes a summary and a review
sheet: per company, each source's rating, the model's reason and the quotes it cites.

Examples::

    python scripts/moat_pilot.py sample
    python scripts/moat_pilot.py tag --set dev
    python scripts/moat_pilot.py report --set dev --out <scratch>/moat_dev.md
    python scripts/moat_pilot.py tag --set sample --model claude-sonnet-5-5 --prompt v1
    python scripts/moat_pilot.py report --set sample --model claude-sonnet-5-5 --prompt v1

After the hand check of those tags, prompt ``v2`` adds each company's financial history
(``company_facts`` and ``financial_history``: SEC XBRL facts as filed up to the 10-K). It
was revised on ``sample`` against the verdicts, then frozen and run once on a fresh set
of companies in no earlier set::

    python scripts/moat_pilot.py tag --set sample --model claude-sonnet-5-5
    python scripts/moat_pilot.py report --set sample --model claude-sonnet-5-5 --out docs/results/moat_v2_sample.md
    python scripts/moat_pilot.py sample --fresh fresh --seed 1
    python scripts/moat_pilot.py tag --set fresh --model claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.moat import (  # noqa: E402
    EXTRACT_SCHEMA,
    KINDS,
    MOATS,
    RATE_SCHEMA,
    SOURCES,
    anonymize,
    chat_json,
    check_rating,
    chunk_lines,
    collect_evidence,
    extract_messages,
    number_quotes,
    extract_version,
    financial_ratios,
    financial_table,
    item1_problem,
    one_call_params,
    one_call_version,
    rate_version,
    verify_evidence,
    rate_messages,
)
from auto_researcher.data.sec_fundamentals import FACTS_URL, ITEMS, build_pit_table, extract_facts  # noqa: E402

logger = logging.getLogger("moat_pilot")

DATA = ROOT / "data" / "moat"
SENTIQ = Path(r"C:\Users\andre\projects\sentiq\data")
LAKE = SENTIQ / "lake" / "items" / "src=edgar_periodic"
SECTORS = SENTIQ / "ref" / "universe" / "sp500_sectors.parquet"
FACTS = DATA / "facts"  # SEC company facts by CIK (filings' own CIKs; tickers change)
SEC_ENV = SENTIQ.parent.parent / "auto_researcher" / ".env"  # SEC_API_USER_AGENT
HISTORY_YEARS = 5
MIN_ITEM1_CHARS = 5000
MAX_CHUNKS = 16  # 192k characters; Item 1 is longer in about 1% of 10-Ks (the rest is not read)
# USD per million tokens (platform.claude.com pricing, as in sentiq's config, 2026-10-09)
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
}
FALLBACK_MODELS = {"claude-sonnet-5-5"}

NOTES = [
    "**How the model was chosen (on the 5 `dev` companies only).** `qwen3:8b` (local) in three prompt "
    "rounds quoted well (2-3% of quotes not in the text) but rated badly: credit for network effects its own "
    "reasons denied, a cost advantage from 'sustainable technologies', a strong brand from 'a well-known "
    "portfolio of brands'; thinking did not fix it. Claude Haiku 5.5 and Sonnet 5.5 then ran the one-call "
    "prompt on the same 5; Sonnet weighed the evidence against a moat and found RTX's installed-base "
    "switching costs, Haiku rated more coarsely. The 50 were tagged once, with Sonnet 5.5 and the prompt "
    "fixed beforehand.",
    "**Deviation: the quote check was loosened after the run, before the hand review.** The first check "
    "rejected 21 of 488 quotes (4.3%), nearly all real: the text extracted from 10-K HTML has spaces around "
    "'®' and dashes ('Tinder ®', 'PSE&G —A') and one table row was quoted with bars. The check now ignores "
    "white space, trademark signs and table bars; it rejects 1 quote, a genuine paraphrase (MTCH: 'Pairs is a "
    "leading provider of online dating services in Japan'; the text says 'Pairs is a dating platform ...'). "
    "The model's replies were not rerun; only the citations it dropped changed (14 fixes before, 1 after).",
    "**Cost** $2.43 for the 50 (Claude API, sentiq's workspace: auto_researcher's own key has no credit).",
    "**Name hiding is partial.** Name, short name, self-defined aliases, tickers and web addresses are "
    "replaced, but brands and details (Tinder, F-35, United's hubs) identify many companies, and Claude "
    "knows how they fared after these 10-Ks (filed in 2025, before its training cutoff). The tags are fit "
    "for a hand check and for describing companies; a return test needs filings after the model's cutoff "
    "or a check that hiding more of the name does not change the tags.",
    "**Only Item 1** (business) is read, from one 10-K per company; moat evidence in MD&A or risk factors "
    "is missed, and 'against' quotes come from the competition sections Item 1 usually has.",
]
# Tickers whose "Item 1" from sentiq's section offsets is another part of the 10-K (found in the hand check).
WRONG_TEXT = {
    "DGX": "MD&A and the financial statements",
    "TTWO": "MD&A: impairments, critical accounting policies, liquidity",
    "O": "the financial statements, with the business description only at the end",
}
HAND_CHECK_NOTES = [
    "**One rater, with hindsight.** The verdicts are the user's, made with the company names shown and "
    "knowing how the companies fared since; they measure whether the tags look right to an informed reader, "
    "not a ground truth. Several are marked low confidence (CHRW, EOG, O).",
    "**The tagger errs low.** Of the overall calls the reviewer disputed on the right text, all but one say "
    "the moat should be higher, as do 21 of the 22 disputed source ratings, mostly 0s. The reasons given "
    "lean on returns: IDXX's returns on capital, AME's margins, CMG's unit economics, and the one call "
    "disputed as too high, CCL, because cruising rarely earns its cost of capital. Item 1 has no numbers, "
    "so the tagger cannot see returns; NTRS was held to a filing that never describes its custody moat.",
    "**A source rated 1 does not force a moat, and that is mostly right.** 18 companies were called none "
    "despite a source rated 1; the reviewer agreed on 16 of them.",
    "**Wrong text from the section offsets.** In a scan of all 480 10-Ks filed in 2025 that have an Item 1 "
    "span, 11 start like MD&A or financial statements in their first 3,000 characters (CL, ZTS, PLD, BALL, "
    "DGX, BG, CDNS, LH, O, AFL, MCHP); TTWO's span is wrong too but starts mid-sentence, so the scan "
    "misses it. Others run on into Item 1A (REGN, NTAP), which adds risk factors but keeps Item 1.",
]
V2_NOTES = [
    "**What changed from v1.** The prompt adds the company's financial history and a step judging returns "
    "on capital before the overall call (`ONE_CALL_SYSTEM_V2`); sources are still rated from quotes only. "
    "Item 1 texts that start mid-sentence or hold MD&A or financial statement headings early are not tagged "
    "(`item1_problem`).",
    "**Tuned on the hand-checked set, then frozen.** Against the targets the 47 v1 verdicts imply, v2 "
    "(fixed rules: high returns plus a source rated 1 meant narrow, one 8-10% cost of capital) put 34 overall "
    "calls on target and moved weak resellers up and utilities down; v3 (returns corroborate the text, a lower "
    "cost of capital for utilities and REITs) put 39. A second run of the unchanged v1 prompt put 36, and 4 "
    "of 47 calls changed between the two identical v1 runs, so v3's lead is small. v3 was committed (bfcd03b) "
    "before the fresh set was tagged once.",
    "**The fresh set** is 50 companies in no earlier set, drawn the same way (sector-stratified, seed 1) from "
    "the 2025 10-Ks that pass the Item 1 check; its tags await the same hand check.",
    "**The financial history is point-in-time:** SEC XBRL company facts filed on or before the 10-K's filing "
    "date (the 10-K's own figures included), the latest value known then for each fiscal year "
    "(`data.sec_fundamentals.build_pit_table`). ROIC is operating income after a 21% tax over average equity "
    "+ debt - cash; ratios on negative or near-zero capital are shown as n/a.",
]
SHORT = {"intangibles": "intang", "switching_costs": "switch", "network_effects": "network",
         "cost_advantage": "cost", "efficient_scale": "eff.scale"}


def load_10ks(year: int) -> pd.DataFrame:
    """10-Ks filed in ``year`` from sentiq's lake, with Item 1 located, one per company."""
    table = ds.dataset(LAKE, format="parquet", partitioning="hive").to_table(
        columns=["tickers", "meta", "body_text"],
        filter=pc.starts_with(ds.field("ym"), str(year)) & pc.match_substring(ds.field("title"), " 10-K "))
    rows = []
    for tickers, meta, body in zip(*(table[c].to_pylist() for c in ("tickers", "meta", "body_text"))):
        m = json.loads(meta)
        span = m["sections"].get("1")
        if m["form"] != "10-K" or not span or span[1] - span[0] < MIN_ITEM1_CHARS:
            continue
        rows.append({"ticker": tickers[0], "tickers": tickers, "cik": m["cik"], "accession": m["accession"],
                     "company": m["company"], "filed": m["filing_date"], "period": m["period"],
                     "item1": body[span[0]:span[1]]})
    df = pd.DataFrame(rows).sort_values("filed")
    return df.drop_duplicates("cik").reset_index(drop=True)


def attach_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """GICS sector as of the filing month (sentiq's monthly snapshots of the Wikipedia list)."""
    sec = pd.read_parquet(SECTORS)
    sec["sector"] = sec["sector"].replace({"Communications Services": "Communication Services"})
    sec["valid_from"] = pd.to_datetime(sec["valid_from"])
    out = []
    for r in df.itertuples():
        s = sec[(sec["ticker"].isin(r.tickers)) & (sec["valid_from"] <= pd.Timestamp(r.filed))]
        out.append(s.sort_values("valid_from")["sector"].iloc[-1] if len(s) else None)
    return df.assign(sector=out).dropna(subset=["sector"])


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows, each sector getting its share (largest remainder), drawn at random."""
    share = df["sector"].value_counts(normalize=True).sort_index() * n
    counts = np.floor(share).astype(int)
    for s in (share - counts).sort_values(ascending=False).index[: n - counts.sum()]:
        counts[s] += 1
    rng = np.random.default_rng(seed)
    picks = [g.iloc[rng.choice(len(g), counts[s], replace=False)] for s, g in df.groupby("sector")]
    return pd.concat(picks).sort_values(["sector", "ticker"]).reset_index(drop=True)


def cmd_sample(args) -> None:
    df = attach_sectors(load_10ks(args.year))
    logger.info("%d 10-Ks filed in %d with Item 1 and a sector", len(df), args.year)
    if args.fresh:
        # A new set for testing the tagger after it was revised on "sample": companies in
        # no earlier set, and only Item 1 texts that pass the wrong-section check.
        used = {c["cik"] for name in ("sample", "dev") for c in json.loads((DATA / f"{name}.json").read_text())}
        bad = df["item1"].map(item1_problem)
        logger.info("dropped %d already used, %d failing the Item 1 check", df["cik"].isin(used).sum(),
                    (bad.notna() & ~df["cik"].isin(used)).sum())
        parts = [(args.fresh, stratified(df[~df["cik"].isin(used) & bad.isna()], args.n, seed=args.seed))]
    else:
        sample = stratified(df, args.n, seed=args.seed)
        rest = df[~df["cik"].isin(sample["cik"])]
        dev = rest.iloc[np.random.default_rng(args.seed + 1).choice(len(rest), args.dev, replace=False)]
        parts = [("sample", sample), ("dev", dev.sort_values(["sector", "ticker"]))]
    DATA.mkdir(parents=True, exist_ok=True)
    for name, part in parts:
        recs = [{**{k: r[k] for k in ("ticker", "tickers", "cik", "accession", "company", "sector", "filed",
                                      "period")},
                 "item1_chars": len(r["item1"]), "item1": anonymize(r["item1"], r["company"], r["tickers"])}
                for _, r in part.iterrows()]
        (DATA / f"{name}.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
        logger.info("%s: %d companies; %s", name, len(recs), dict(Counter(r["sector"] for r in recs)))


def company_facts(cik: int, session=None) -> pd.DataFrame:
    """All XBRL facts a company has filed (cached in ``FACTS``; filter by ``filed`` before use)."""
    path = FACTS / f"{int(cik)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import requests
    from dotenv import dotenv_values

    ua = (dotenv_values(SEC_ENV).get("SEC_API_USER_AGENT") or "").strip().strip('"')
    resp = (session or requests).get(FACTS_URL.format(cik=int(cik)), headers={"User-Agent": ua}, timeout=60)
    time.sleep(0.15)  # SEC allows 10 requests a second
    if resp.status_code == 404:
        facts = pd.DataFrame(columns=["symbol", "item", "concept", "priority", "start", "end", "value", "filed",
                                      "accn"])
    else:
        resp.raise_for_status()
        facts = extract_facts(resp.json(), str(int(cik)))
    FACTS.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(path)
    return facts


def financial_history(cik: int, filed: str, period: str, years: int = HISTORY_YEARS) -> dict:
    """Ratios for the ``years`` fiscal years up to ``period``, from facts filed on or before
    ``filed`` (the 10-K's own figures included), the latest value known then per period."""
    facts = company_facts(cik)
    facts = facts[facts["filed"] <= pd.Timestamp(filed)]
    if facts.empty:
        return {}
    table = build_pit_table(facts, mode="last")
    table = table[table["item"].map(lambda i: ITEMS[i].unit == "USD")]  # not the cover-page share counts
    fy_end = pd.Timestamp(period)
    out = {}
    for k in range(years, -1, -1):
        target = fy_end - pd.DateOffset(years=k)
        gap = (table["period_end"] - target).abs()
        if not (gap <= pd.Timedelta(days=20)).any():
            continue
        end = table.loc[gap.idxmin(), "period_end"]
        rows = table[table["period_end"] == end]
        out[f"{end:%Y-%m-%d}"] = {item: float(v) for item, v in zip(rows["item"], rows["value"])}
    return financial_ratios(out)


def _cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _parse(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_tags(args) -> tuple[str, str]:
    """Names for the extract replies and for everything after them (ratings, results). A
    Claude model runs both steps in one call, so both names are the same."""
    if args.model.startswith("claude-"):
        tag = f"{args.model}__c{one_call_version(args.prompt)}__{args.effort}"
        tag += f"__run{args.run}" if args.run else ""
        return tag, tag
    ext = f"{args.model.replace(':', '_')}__x{extract_version()}"
    return ext, f"{ext}__r{rate_version()}{'__think' if args.think else ''}"


def cmd_tag(args) -> None:
    if args.model.startswith("claude-"):
        return cmd_tag_claude(args)
    companies = json.loads((DATA / f"{args.set}.json").read_text(encoding="utf-8"))
    runs = DATA / "runs"
    runs.mkdir(exist_ok=True)
    ext_tag, tag = run_tags(args)
    ext_path, rate_path = runs / f"{args.set}__{ext_tag}__extract.jsonl", runs / f"{args.set}__{tag}__rate.jsonl"
    ext_done, rate_done = _cached(ext_path), _cached(rate_path)
    results = []
    with ext_path.open("a", encoding="utf-8") as ext_fh, rate_path.open("a", encoding="utf-8") as rate_fh:
        for n, c in enumerate(companies, 1):
            chunks = chunk_lines(c["item1"])
            replies = []
            for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
                key = f"{c['accession']}:{i}"
                if key not in ext_done:
                    raw, sec = chat_json(args.url, args.model, extract_messages(chunk), EXTRACT_SCHEMA)
                    ext_done[key] = {"key": key, "raw": raw, "seconds": sec}
                    ext_fh.write(json.dumps(ext_done[key]) + "\n")
                    ext_fh.flush()
                replies.append((chunk, _parse(ext_done[key]["raw"]) or {}))
            by_kind, counts = collect_evidence(replies)
            counts["bad_json"] = sum(_parse(ext_done[f"{c['accession']}:{i}"]["raw"]) is None
                                     for i in range(len(replies)))
            numbered = number_quotes(by_kind)
            key = c["accession"]
            if key not in rate_done:
                raw, sec = chat_json(args.url, args.model, rate_messages(c["item1"], numbered), RATE_SCHEMA,
                                     think=args.think)
                rate_done[key] = {"key": key, "raw": raw, "seconds": sec}
                rate_fh.write(json.dumps(rate_done[key]) + "\n")
                rate_fh.flush()
            rating = _parse(rate_done[key]["raw"])
            seconds = sum(ext_done[f"{key}:{i}"]["seconds"] for i in range(len(replies))) + rate_done[key]["seconds"]
            results.append({**{k: c[k] for k in ("ticker", "company", "sector", "filed", "accession", "item1_chars")},
                            "chunks": len(chunks), "chunks_read": len(replies), "counts": counts,
                            "numbered": numbered, "rating": check_rating(rating, numbered) if rating else None,
                            "opening": c["item1"][:600], "seconds": seconds})
            r = results[-1]["rating"]
            logger.info("%2d/%d %-6s %-24s %s  %s", n, len(companies), c["ticker"], c["sector"][:24],
                        r and r["overall"]["moat"], r and " ".join(f"{SHORT[k]}={r[k]['rating']}" for k in SOURCES))
    out = DATA / f"{args.set}__{tag}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    logger.info("wrote %s", out)


def call_cost(model: str, usage: dict[str, int]) -> float:
    """USD for one call (``PRICES``); thinking tokens are billed as output."""
    p = PRICES[model]
    return (usage["input_tokens"] * p["input"] + usage["cache_creation_input_tokens"] * p["cache_write"]
            + usage["cache_read_input_tokens"] * p["cache_read"] + usage["output_tokens"] * p["output"]) / 1e6


def claude_call(client, params: dict) -> dict:
    """One Messages call; Sonnet 5.5 with the server-side refusal fallback (another model
    finishes a declined request; the serving model is recorded). The SDK retries 429/5xx."""
    if params["model"] in FALLBACK_MODELS:
        msg = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        msg = client.messages.create(**params)
    usage = {k: int(getattr(msg.usage, k, 0) or 0) for k in
             ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    text = next((b.text for b in msg.content if b.type == "text"), None)
    return {"stop_reason": msg.stop_reason, "served_by": msg.model, "usage": usage,
            "cost": call_cost(params["model"], usage), "raw": text, "request_id": msg._request_id}


def cmd_tag_claude(args) -> None:
    import anthropic
    from dotenv import dotenv_values

    key = dotenv_values(args.env_file).get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(f"no ANTHROPIC_API_KEY in {args.env_file}")
    client = anthropic.Anthropic(api_key=key, max_retries=4)
    companies = json.loads((DATA / f"{args.set}.json").read_text(encoding="utf-8"))
    runs = DATA / "runs"
    runs.mkdir(exist_ok=True)
    tag = run_tags(args)[1]
    path = runs / f"{args.set}__{tag}.jsonl"
    done = _cached(path)
    results = []
    with path.open("a", encoding="utf-8") as fh:
        for n, c in enumerate(companies, 1):
            key = c["accession"]
            base = {k: c[k] for k in ("ticker", "company", "sector", "filed", "accession", "item1_chars")}
            problem = item1_problem(c["item1"]) if args.prompt != "v1" else None
            if problem:
                logger.info("%2d/%d %-6s skipped: Item 1 %s", n, len(companies), c["ticker"], problem)
                results.append({**base, "chunks": 0, "chunks_read": 0, "counts": {}, "numbered": [],
                                "rating": None, "problem": problem, "opening": c["item1"][:600],
                                "seconds": 0.0, "cost": 0.0, "served_by": args.model, "stop_reason": None})
                continue
            ratios = financial_history(c["cik"], c["filed"], c["period"]) if args.prompt != "v1" else {}
            if key not in done:
                start = time.perf_counter()
                params = one_call_params(args.model, c["item1"], args.effort, prompt=args.prompt,
                                         financials=financial_table(ratios))
                done[key] = {"key": key, **claude_call(client, params), "seconds": time.perf_counter() - start}
                fh.write(json.dumps(done[key]) + "\n")
                fh.flush()
            d = done[key]
            reply = _parse(d["raw"]) if d["raw"] and d["stop_reason"] == "end_turn" else None
            numbered, counts = verify_evidence(reply["evidence"], c["item1"]) if reply else ([], Counter())
            counts = {**counts, "bad_json": int(reply is None)}
            results.append({**base, "chunks": 1, "chunks_read": 1, "counts": counts, "numbered": numbered,
                            "rating": check_rating(reply, numbered) if reply else None, "ratios": ratios,
                            "opening": c["item1"][:600], "seconds": d["seconds"], "cost": d["cost"],
                            "served_by": d["served_by"], "stop_reason": d["stop_reason"]})
            r = results[-1]["rating"]
            logger.info("%2d/%d %-6s %-24s %s  %s  $%.3f", n, len(companies), c["ticker"], c["sector"][:24],
                        r and r["overall"]["moat"], r and " ".join(f"{SHORT[k]}={r[k]['rating']}" for k in SOURCES),
                        d["cost"])
    out = DATA / f"{args.set}__{tag}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    logger.info("wrote %s; cost $%.2f", out, sum(r["cost"] for r in results))


def _excerpt(text: str, chars: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + " ..."


def hand_check(results: list[dict], verdicts: dict) -> list[str]:
    """Score the tags against the reviewer's verdicts (agree / too high / too low)."""
    pct = "{:.0%}".format
    rows = [(r["ticker"], r["rating"], verdicts[r["ticker"]]) for r in results
            if r["rating"] and r["ticker"] in verdicts]
    lines = ["", "## Hand check", "",
             f"{len(rows)} companies reviewed on the review page (verdicts in `data/moat/verdicts.json`). "
             f"Tickers with the wrong text: {', '.join(f'{t} ({w})' for t, w in WRONG_TEXT.items())}.", "",
             "| Companies | Overall agreed | Too high | Too low | Source ratings agreed | Too high | Too low |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, keep in [("all", lambda t: True), ("right text", lambda t: t not in WRONG_TEXT)]:
        sub = [(t, g, v) for t, g, v in rows if keep(t)]
        ov = Counter(v["overall"] for _, _, v in sub)
        src = Counter(v["sources"][k] for _, _, v in sub for k in SOURCES)
        lines.append(f"| {label} ({len(sub)}) | {ov['agree']} ({pct(ov['agree'] / len(sub))}) | {ov['high']} | "
                     f"{ov['low']} | {src['agree']} of {sum(src.values())} ({pct(src['agree'] / sum(src.values()))})"
                     f" | {src['high']} | {src['low']} |")
    lines += ["", "| Tag | Reviewed (right text) | Agreed |", "| --- | ---: | ---: |"]
    for m in MOATS:
        sub = [v for t, g, v in rows if g["overall"]["moat"] == m and t not in WRONG_TEXT]
        lines.append(f"| {m} | {len(sub)} | {sum(v['overall'] == 'agree' for v in sub)} |")
    lines += ["", "Disputed overall calls (\"too low\" = the moat should be higher):", ""]
    for t, g, v in rows:
        if v["overall"] != "agree":
            lines.append(f"* **{t}**: tagged {g['overall']['moat']}, too {v['overall']}"
                         f"{' (wrong text)' if t in WRONG_TEXT else ''}. {v['note']}")
    lines += ["", "Disputed source ratings:", ""]
    for t, g, v in rows:
        bad = [f"{k} {g[k]['rating']} too {v['sources'][k]}" for k in SOURCES if v["sources"][k] != "agree"]
        if bad:
            lines.append(f"* {t}: {'; '.join(bad)}")
    return lines + ["", "### What the hand check shows", ""] + [f"* {n}" for n in HAND_CHECK_NOTES]


def _target(old: int, verdict: str, levels: int) -> set[int]:
    """The ratings a verdict on ``old`` accepts: the same one, or any higher or lower."""
    if verdict == "low" and old < levels - 1:
        return set(range(old + 1, levels))
    if verdict == "high" and old > 0:
        return set(range(old))
    return {old}


def against_verdicts(results: list[dict], v1: list[dict], verdicts: dict, prompt: str,
                     baselines: list[tuple[str, list[dict]]] = ()) -> list[str]:
    """Score new tags against the hand check of the ``v1`` tags: a verdict of "too low" on
    a v1 call is met by any higher call, "too high" by any lower one, "agree" by the same."""
    old = {r["ticker"]: r["rating"] for r in v1 if r["rating"]}
    rows = [(r["ticker"], old[r["ticker"]], r["rating"], verdicts[r["ticker"]]) for r in results
            if r["rating"] and r["ticker"] in old and r["ticker"] in verdicts and r["ticker"] not in WRONG_TEXT]
    lvl = {m: i for i, m in enumerate(MOATS)}
    pct = "{:.0%}".format

    def hit_overall(o, n, v):
        return lvl[n["overall"]["moat"]] in _target(lvl[o["overall"]["moat"]], v["overall"], len(MOATS))

    def hits_sources(o, n, v):
        return sum(n[k]["rating"] in _target(o[k]["rating"], v["sources"][k], 3) for k in SOURCES)

    ns = len(rows) * len(SOURCES)

    def row(label, res):
        new = {r["ticker"]: r["rating"] for r in res if r["rating"]}
        sub = [(o, new[t], v) for t, o, _, v in rows if t in new]
        ov = sum(hit_overall(o, n, v) for o, n, v in sub)
        sc = sum(hits_sources(o, n, v) for o, n, v in sub)
        n_src = len(sub) * len(SOURCES)
        return f"| {label} | {ov} ({pct(ov / len(sub))}) | {sc} of {n_src} ({pct(sc / n_src)}) |"

    ov1 = sum(v["overall"] == "agree" for _, _, _, v in rows)
    sc1 = sum(v["sources"][k] == "agree" for _, _, _, v in rows for k in SOURCES)
    lines = ["", "## Against the hand check of v1", "",
             f"The reviewer judged the v1 tags; each verdict implies a target for the new tags (\"too low\" is "
             f"met by any higher call, \"too high\" by any lower one, \"agree\" by the same call). "
             f"{len(rows)} companies with the right Item 1 text in both runs. The reviewed v1 run is favoured by "
             "construction (the verdicts were written about its exact calls); a second run of the same v1 prompt "
             "shows how many targets a run misses by chance.", "",
             "| Tags | Overall calls on target | Source ratings on target |", "| --- | ---: | ---: |",
             f"| v1, the reviewed run | {ov1} ({pct(ov1 / len(rows))}) | {sc1} of {ns} ({pct(sc1 / ns)}) |",
             *(row(label, res) for label, res in baselines),
             row(f"{prompt} (with financials)", results),
             "",
             "Companies whose overall call changed or misses its target:", "",
             f"| Ticker | v1 | Verdict on v1 | {prompt} | Returns ({prompt}) | On target |",
             "| --- | --- | --- | --- | --- | --- |"]
    for t, o, n, v in rows:
        changed = o["overall"]["moat"] != n["overall"]["moat"]
        hit = hit_overall(o, n, v)
        if changed or not hit:
            lines.append(f"| {t} | {o['overall']['moat']} | {v['overall']} | {n['overall']['moat']} | "
                         f"{n.get('returns', {}).get('verdict', '')} | {'yes' if hit else 'no'} |")
    return lines


def cmd_report(args) -> None:
    tag = run_tags(args)[1]
    results = json.loads((DATA / f"{args.set}__{tag}.json").read_text(encoding="utf-8"))
    rated = [r for r in results if r["rating"]]
    tot = Counter()
    for r in results:
        tot.update(r["counts"])
    quoted = tot["kept"] + tot["not_in_text"] + tot["duplicate"] + tot["over_cap"]
    kinds = Counter(k for r in results for _, k, _ in r["numbered"])
    fixes = sum(len(r["rating"]["fixes"]) for r in rated)
    claude = args.model.startswith("claude-")
    how = (f"One call per company over Item 1 (business){' and five years of financials' if args.prompt != 'v1' else ''}"
           f", prompt {args.prompt}: quotes checked word for word against the text, each moat source rated from "
           "the verified quotes only" if claude else
           "Two passes over each 10-K's Item 1 (business): extract quotes per chunk and check them word for word "
           "against the text, then rate each moat source from the verified quotes only")
    lines = [
        "# Moat tagger pilot" if args.prompt == "v1" else f"# Moat tagger {args.prompt}: set `{args.set}`", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/moat_pilot.py`, set `{args.set}`, model "
        f"`{args.model}`, prompts `{tag}`. {how} (`auto_researcher.llm.moat`).", "",
        f"* Companies: {len(results)} ({len(rated)} rated); chunks read {sum(r['chunks_read'] for r in results)} "
        f"(Item 1 cut at {MAX_CHUNKS} chunks for {sum(r['chunks'] > r['chunks_read'] for r in results)}).",
        f"* Quotes returned: {quoted}; kept {tot['kept']}; not found in the text {tot['not_in_text']} "
        f"({tot['not_in_text'] / max(quoted, 1):.1%}); duplicates {tot['duplicate']}; over the cap "
        f"{tot['over_cap']}; unparseable replies {tot['bad_json']}.",
        "* Kept quotes by kind: " + ", ".join(f"{k} {kinds[k]}" for k in KINDS) + ".",
        f"* Rating fixes (citations of missing or \"against\" quotes dropped, uncited ratings lowered to 0): "
        f"{fixes}.",
        f"* Model time: {sum(r['seconds'] for r in results) / 60:.0f} min "
        f"({np.mean([r['seconds'] for r in results]):.0f} s per company)"
        + (f"; API cost ${sum(r['cost'] for r in results):.2f}" if "cost" in results[0] else "")
        + (f"; served by another model {sum(r['served_by'] != args.model for r in results)}"
           if "served_by" in results[0] else "") + ".", "",
        "## Ratings", "",
        "0 = none or generic, 1 = specific but modest, 2 = strong and durable.", "",
        "| Ticker | Sector | Overall | " + " | ".join(SHORT[k] for k in SOURCES) + " | Against quotes |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in SOURCES) + " | ---: |",
    ]
    with_returns = any(r["rating"] and "returns" in r["rating"] for r in results)
    if with_returns:
        lines[-2:] = [lines[-2] + " Returns |", lines[-1] + " --- |"]
    for r in results:
        g = r["rating"]
        cells = [g["overall"]["moat"], *(str(g[k]["rating"]) for k in SOURCES)] if g else ["?"] * 6
        against = sum(k == "against" for _, k, _ in r["numbered"])
        lines.append(f"| {r['ticker']} | {r['sector']} | " + " | ".join(cells) + f" | {against} |"
                     + (f" {g['returns']['verdict'] if g else '?'} |" if with_returns else ""))
    skipped = [r for r in results if r.get("problem")]
    if skipped:
        lines += ["", "Not tagged, because the located Item 1 is another part of the 10-K: "
                  + "; ".join(f"{r['ticker']} ({r['problem']})" for r in skipped) + "."]
    lines += ["", "| Sector | " + " | ".join(MOATS) + " |", "| --- | " + " | ".join("---:" for _ in MOATS) + " |"]
    by_sector = Counter((r["sector"], r["rating"]["overall"]["moat"]) for r in rated)
    for s in sorted({r["sector"] for r in rated}):
        lines.append(f"| {s} | " + " | ".join(str(by_sector[(s, m)]) for m in MOATS) + " |")
    verdicts = DATA / "verdicts.json"
    if args.set == "sample" and args.prompt == "v1":
        lines += ["", "## Notes", ""] + [f"* {n}" for n in NOTES]
        if verdicts.exists():
            lines += hand_check(results, json.loads(verdicts.read_text(encoding="utf-8")))
    elif args.set == "sample" and verdicts.exists():
        v1_tag = f"sample__{args.model}__c{one_call_version('v1')}__{args.effort}"
        v1 = json.loads((DATA / f"{v1_tag}.json").read_text(encoding="utf-8"))
        baselines = [(f"v1, run {p.stem.rsplit('run', 1)[1]} (same prompt)", json.loads(p.read_text(encoding="utf-8")))
                     for p in sorted(DATA.glob(f"{v1_tag}__run*.json"))]
        lines += against_verdicts(results, v1, json.loads(verdicts.read_text(encoding="utf-8")), args.prompt,
                                  baselines)
    if args.prompt != "v1":
        lines += ["", "## Notes", ""] + [f"* {n}" for n in V2_NOTES]
    lines += ["", "## Review sheet", "",
              "For each company: what it does (the opening of Item 1, name hidden), the overall call, then each "
              "source rated above 0 with the model's reason and the quotes it cites, and the quotes against a "
              "moat. Sources rated 0 are listed by name only.", ""]
    for r in results:
        g, quotes = r["rating"], {i: q for i, _, q in r["numbered"]}
        lines += [f"### {r['ticker']}: {r['company']} ({r['sector']}, 10-K filed {r['filed']})", "",
                  f"> {_excerpt(r['opening'])}", ""]
        if not g:
            lines += [f"Not tagged: the located Item 1 {r['problem']}." if r.get("problem")
                      else "No valid rating reply.", ""]
            continue
        lines += [f"**Overall: {g['overall']['moat']}.** {g['overall']['reason']}", ""]
        if "returns" in g:
            lines += [f"**Returns: {g['returns']['verdict']}.** {g['returns']['reason']}", ""]
            table = financial_table(r.get("ratios") or {})
            lines += [table.split("\n\n")[0], ""] if table else ["No financial history.", ""]
        for k in SOURCES:
            if g[k]["rating"] > 0:
                lines.append(f"* **{k} = {g[k]['rating']}.** {g[k]['reason']}")
                lines += [f"  * [{i}] \"{quotes[i]}\"" for i in g[k]["quotes"]]
        zeros = [k for k in SOURCES if g[k]["rating"] == 0]
        if zeros:
            lines.append(f"* Rated 0: {', '.join(zeros)}.")
        against = [(i, q) for i, k, q in r["numbered"] if k == "against"]
        if against:
            lines.append("* Against a moat:")
            lines += [f"  * [{i}] \"{q}\"" for i, q in against]
        if g["fixes"]:
            lines.append(f"* Fixes: {'; '.join(g['fixes'])}.")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", args.out)


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--year", type=int, default=2025)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--dev", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--fresh", metavar="NAME", help="draw a new set NAME from companies in no earlier set")
    for name in ("tag", "report"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="sample")
        p.add_argument("--model", default="qwen3:8b")
        p.add_argument("--url", default="http://localhost:11434")
        p.add_argument("--think", action="store_true", help="let the model think before rating (Ollama)")
        p.add_argument("--effort", default="high", help="Claude models: thinking effort")
        p.add_argument("--prompt", default="v3", choices=["v1", "v2", "v3"],
                       help="Claude models: v1 = Item 1 only (the first hand check); v2 and v3 add financials")
        p.add_argument("--run", type=int, default=0,
                       help="Claude models: repeat number, to measure how much the tags vary between runs")
        # auto_researcher's own key has no credit (2026-10-31); spend goes to sentiq's workspace
        p.add_argument("--env-file", type=Path, default=SENTIQ.parent / ".env",
                       help="file with ANTHROPIC_API_KEY (Claude models)")
    sub.choices["report"].add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "moat_pilot.md")
    args = ap.parse_args()
    {"sample": cmd_sample, "tag": cmd_tag, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
