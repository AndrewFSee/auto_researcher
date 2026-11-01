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
    python scripts/moat_pilot.py tag --set sample --model claude-sonnet-5-5
    python scripts/moat_pilot.py report --set sample --model claude-sonnet-5-5
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
    one_call_params,
    one_call_version,
    rate_version,
    verify_evidence,
    rate_messages,
)

logger = logging.getLogger("moat_pilot")

DATA = ROOT / "data" / "moat"
SENTIQ = Path(r"C:\Users\andre\projects\sentiq\data")
LAKE = SENTIQ / "lake" / "items" / "src=edgar_periodic"
SECTORS = SENTIQ / "ref" / "universe" / "sp500_sectors.parquet"
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
    sample = stratified(df, args.n, seed=args.seed)
    rest = df[~df["cik"].isin(sample["cik"])]
    dev = rest.iloc[np.random.default_rng(args.seed + 1).choice(len(rest), args.dev, replace=False)]
    DATA.mkdir(parents=True, exist_ok=True)
    for name, part in (("sample", sample), ("dev", dev.sort_values(["sector", "ticker"]))):
        recs = [{**{k: r[k] for k in ("ticker", "tickers", "cik", "accession", "company", "sector", "filed",
                                      "period")},
                 "item1_chars": len(r["item1"]), "item1": anonymize(r["item1"], r["company"], r["tickers"])}
                for _, r in part.iterrows()]
        (DATA / f"{name}.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
        logger.info("%s: %d companies; %s", name, len(recs), dict(Counter(r["sector"] for r in recs)))


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
        tag = f"{args.model}__c{one_call_version()}__{args.effort}"
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
            if key not in done:
                start = time.perf_counter()
                done[key] = {"key": key, **claude_call(client, one_call_params(args.model, c["item1"], args.effort)),
                             "seconds": time.perf_counter() - start}
                fh.write(json.dumps(done[key]) + "\n")
                fh.flush()
            d = done[key]
            reply = _parse(d["raw"]) if d["raw"] and d["stop_reason"] == "end_turn" else None
            numbered, counts = verify_evidence(reply["evidence"], c["item1"]) if reply else ([], Counter())
            counts = {**counts, "bad_json": int(reply is None)}
            results.append({**{k: c[k] for k in ("ticker", "company", "sector", "filed", "accession", "item1_chars")},
                            "chunks": 1, "chunks_read": 1, "counts": counts, "numbered": numbered,
                            "rating": check_rating(reply, numbered) if reply else None,
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
    lines = [
        "# Moat tagger pilot", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/moat_pilot.py`, set `{args.set}`, model "
        f"`{args.model}`, prompts `{tag}`. Two passes over each 10-K's Item 1 (business): "
        "extract quotes per chunk and check them word for word against the text, then rate each moat source "
        "from the verified quotes only (`auto_researcher.llm.moat`).", "",
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
    for r in results:
        g = r["rating"]
        cells = [g["overall"]["moat"], *(str(g[k]["rating"]) for k in SOURCES)] if g else ["?"] * 6
        against = sum(k == "against" for _, k, _ in r["numbered"])
        lines.append(f"| {r['ticker']} | {r['sector']} | " + " | ".join(cells) + f" | {against} |")
    lines += ["", "| Sector | " + " | ".join(MOATS) + " |", "| --- | " + " | ".join("---:" for _ in MOATS) + " |"]
    by_sector = Counter((r["sector"], r["rating"]["overall"]["moat"]) for r in rated)
    for s in sorted({r["sector"] for r in rated}):
        lines.append(f"| {s} | " + " | ".join(str(by_sector[(s, m)]) for m in MOATS) + " |")
    if args.set == "sample":
        lines += ["", "## Notes", ""] + [f"* {n}" for n in NOTES]
    lines += ["", "## Review sheet", "",
              "For each company: what it does (the opening of Item 1, name hidden), the overall call, then each "
              "source rated above 0 with the model's reason and the quotes it cites, and the quotes against a "
              "moat. Sources rated 0 are listed by name only.", ""]
    for r in results:
        g, quotes = r["rating"], {i: q for i, _, q in r["numbered"]}
        lines += [f"### {r['ticker']}: {r['company']} ({r['sector']}, 10-K filed {r['filed']})", "",
                  f"> {_excerpt(r['opening'])}", ""]
        if not g:
            lines += ["No valid rating reply.", ""]
            continue
        lines += [f"**Overall: {g['overall']['moat']}.** {g['overall']['reason']}", ""]
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
    for name in ("tag", "report"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="sample")
        p.add_argument("--model", default="qwen3:8b")
        p.add_argument("--url", default="http://localhost:11434")
        p.add_argument("--think", action="store_true", help="let the model think before rating (Ollama)")
        p.add_argument("--effort", default="high", help="Claude models: thinking effort")
        # auto_researcher's own key has no credit (2026-10-31); spend goes to sentiq's workspace
        p.add_argument("--env-file", type=Path, default=SENTIQ.parent / ".env",
                       help="file with ANTHROPIC_API_KEY (Claude models)")
    sub.choices["report"].add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "moat_pilot.md")
    args = ap.parse_args()
    {"sample": cmd_sample, "tag": cmd_tag, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
