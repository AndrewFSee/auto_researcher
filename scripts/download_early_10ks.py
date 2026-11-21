"""
Download the 10-Ks of S&P 500 companies filed 2009-2017 from EDGAR, for the early-sample
customer momentum test (``docs/results/customer_momentum_2010_protocol.md``).

sentiq's lake starts in 2018, so this fetches the earlier filings itself and stores them in
the shape ``customer_links.load_10ks`` reads: one 10-K per company per calendar year (the
first original 10-K filed), primary document only, turned into text with sentiq's own
``html_to_text`` (loaded from its source, read-only) and not truncated. Companies are those
in sentiq's ``ref/universe/sp500_members.parquet`` (every member from 2018 on).

Writes ``data/customers/early_10k/<year>.jsonl.gz`` (resumable: a filing already stored is not
fetched again) and ``data/customers/early_10k/index.json`` (per filing: whether its EDGAR
folder holds an Exhibit 13, which this does not read).

    python scripts/download_early_10ks.py              # all years
    python scripts/download_early_10ks.py --limit 5    # a few companies, to check
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402

logger = logging.getLogger("download_early_10ks")

SENTIQ = Path(r"C:\Users\andre\projects\sentiq")
MEMBERS = SENTIQ / "data" / "ref" / "universe" / "sp500_members.parquet"
TICKERS = SENTIQ / "data" / "ref" / "company_tickers.json"
OUT = ROOT / "data" / "customers" / "early_10k"
YEARS = range(2009, 2018)
SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"
PAUSE = 0.15  # SEC allows 10 requests a second
_EX13 = re.compile(r"ex-?13(?![0-9])", re.I)


def html_to_text():
    """sentiq's ``newsfeed.textutil.html_to_text``, loaded from its file (no sentiq imports run)."""
    spec = importlib.util.spec_from_file_location("sentiq_textutil", SENTIQ / "newsfeed" / "textutil.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.html_to_text


class Edgar:
    def __init__(self) -> None:
        ua = (dotenv_values(ROOT / ".env").get("SEC_API_USER_AGENT") or "").strip().strip('"')
        if not ua:
            raise SystemExit("Set SEC_API_USER_AGENT in .env (SEC requires it)")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = ua

    def get(self, url: str) -> requests.Response | None:
        for attempt in range(5):
            time.sleep(PAUSE)
            try:
                r = self.s.get(url, timeout=60)
            except requests.RequestException as e:
                logger.warning("%s: %s (retry %d)", url, e, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"giving up on {url}")


def filings_of(edgar: Edgar, cik: int) -> list[dict]:
    """Every original 10-K the company filed 2009-2017 (the submissions file and its older pages)."""
    first = edgar.get(SUBMISSIONS.format(name=f"CIK{cik:010d}.json"))
    if first is None:
        return []
    sub = first.json()
    pages = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        r = edgar.get(SUBMISSIONS.format(name=f["name"]))
        if r is not None:
            pages.append(r.json())
    out = []
    for p in pages:
        for acc, form, filed, report, doc in zip(p["accessionNumber"], p["form"], p["filingDate"],
                                                 p["reportDate"], p["primaryDocument"]):
            if form == "10-K" and str(YEARS.start) <= filed[:4] <= str(YEARS.stop - 1) and doc:
                out.append({"accession": acc, "filed": filed, "period": report or None, "document": doc,
                            "company": sub.get("name", "")})
    return out


def has_ex13(edgar: Edgar, cik: int, acc: str) -> bool | None:
    r = edgar.get(ARCHIVE.format(cik=cik, acc=acc.replace("-", "")) + "/index.json")
    if r is None:
        return None
    return any(_EX13.search(i.get("name", "")) for i in r.json().get("directory", {}).get("item", []))


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, help="only the first N companies (a check run)")
    args = ap.parse_args()

    to_text = html_to_text()
    members = pd.read_parquet(MEMBERS)
    sec = json.loads(TICKERS.read_text(encoding="utf-8"))
    current: dict[int, str] = {}
    for r in sec.get("data", sec).values():
        current.setdefault(int(r["cik_str"]), r["ticker"])
    companies = members.assign(cik=members["cik"].astype(int)).drop_duplicates("cik").sort_values("cik")
    if args.limit:
        companies = companies.head(args.limit)

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    stored = {y: set() for y in YEARS}
    for y in YEARS:
        p = OUT / f"{y}.jsonl.gz"
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                stored[y] = {json.loads(line)["accession"] for line in fh}
    edgar = Edgar()
    for n, (cik, ticker) in enumerate(zip(companies["cik"], companies["ticker"]), 1):
        firsts: dict[int, dict] = {}
        for f in filings_of(edgar, int(cik)):
            y = int(f["filed"][:4])
            if y not in firsts or f["filed"] < firsts[y]["filed"]:
                firsts[y] = f
        for y, f in sorted(firsts.items()):
            if f["accession"] in stored[y]:
                continue
            url = ARCHIVE.format(cik=int(cik), acc=f["accession"].replace("-", "")) + "/" + f["document"]
            r = edgar.get(url)
            if r is None:
                logger.warning("%s %s: primary document not found", ticker, f["accession"])
                continue
            row = {"ticker": current.get(int(cik), ticker), "cik": int(cik), "accession": f["accession"],
                   "company": f["company"], "filed": f["filed"], "period": f["period"], "document": f["document"],
                   "text": to_text(r.content)}
            with gzip.open(OUT / f"{y}.jsonl.gz", "at", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            stored[y].add(f["accession"])
            index[f["accession"]] = {"cik": int(cik), "year": y, "ex13": has_ex13(edgar, int(cik), f["accession"])}
        if n % 10 == 0 or n == len(companies):
            index_path.write_text(json.dumps(index), encoding="utf-8")
            logger.info("%d / %d companies; filings stored %s", n, len(companies),
                        {y: len(s) for y, s in stored.items()})
    index_path.write_text(json.dumps(index), encoding="utf-8")


if __name__ == "__main__":
    main()
