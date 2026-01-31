"""Debug SEC EDGAR URLs."""
import requests

headers = {"User-Agent": "AutoResearcher research@example.com"}

# Get NVDA CIK
url = "https://www.sec.gov/files/company_tickers.json"
resp = requests.get(url, headers=headers, timeout=10)
data = resp.json()

cik = None
for entry in data.values():
    if entry.get("ticker") == "NVDA":
        cik = str(entry["cik_str"]).zfill(10)
        print(f"NVDA CIK: {cik}")
        break

# Get filings
url = f"https://data.sec.gov/submissions/CIK{cik}.json"
print(f"Fetching: {url}")
resp = requests.get(url, headers=headers, timeout=15)
print(f"Status: {resp.status_code}")
data = resp.json()

filings = data.get("filings", {}).get("recent", {})
forms = filings.get("form", [])
accessions = filings.get("accessionNumber", [])

for i, form in enumerate(forms[:20]):
    if form == "10-K":
        acc = accessions[i]
        print(f"10-K accession: {acc}")
        acc_clean = acc.replace("-", "")
        
        # Try different URL patterns
        cik_short = cik.lstrip("0")
        
        urls_to_try = [
            f"https://data.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/index.json",
            f"https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/index.json",
            f"https://data.sec.gov/Archives/edgar/data/{cik_short}/{acc}/index.json",
        ]
        
        for test_url in urls_to_try:
            print(f"\nTrying: {test_url}")
            resp2 = requests.get(test_url, headers=headers, timeout=10)
            print(f"Status: {resp2.status_code}")
            if resp2.status_code == 200:
                idx = resp2.json()
                items = idx.get("directory", {}).get("item", [])
                print(f"Found {len(items)} files")
                for item in items[:5]:
                    print(f"  - {item.get('name')}")
                break
        
        break
