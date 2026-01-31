"""Debug 10-K document structure."""
import requests
import re

headers = {"User-Agent": "Test test@test.com"}

# Get NVDA 10-K
acc = "000104581025000023"
cik = "1045810"
index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"

resp = requests.get(index_url, headers=headers, timeout=10)
data = resp.json()

items = data.get("directory", {}).get("item", [])
print(f"Found {len(items)} files:")
for item in items[:10]:
    print(f"  - {item.get('name')}")

# Find main document - NVDA uses nvda-YYYYMMDD.htm format
main_doc = None
for item in items:
    name = item.get("name", "")
    # Look for main filing document - ticker-date.htm pattern or 10-k in name
    if name.endswith(".htm"):
        if "10-k" in name.lower() or "10k" in name.lower():
            main_doc = name
            break
        # Also check for ticker-date pattern (common format)
        if name.startswith("nvda-") and name.endswith(".htm"):
            main_doc = name
            break

if not main_doc:
    # Try to find largest .htm file from items
    for item in items:
        name = item.get("name", "")
        size = item.get("size", 0)
        if name.endswith(".htm") and not name.startswith("0001"):
            print(f"Potential: {name} (size: {size})")
            if size and int(size) > 100000:  # Larger than 100KB
                main_doc = name
                break

if main_doc:
    print(f"\nMain doc: {main_doc}")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{main_doc}"
    resp2 = requests.get(doc_url, headers=headers, timeout=30)
    print(f"Doc length: {len(resp2.text):,} chars")
    
    # Test section extraction
    text = re.sub(r'<[^>]+>', ' ', resp2.text)
    text = re.sub(r'\s+', ' ', text)
    print(f"Cleaned text length: {len(text):,} chars")
    
    # Look for ITEM 1
    if "ITEM 1" in text.upper():
        print("\nFound ITEM 1 in text!")
        # Find positions
        for m in re.finditer(r'ITEM\s*1', text, re.IGNORECASE):
            pos = m.start()
            context = text[pos:pos+100]
            print(f"  Position {pos}: {context}...")
            if pos > 500:
                break
    else:
        print("ITEM 1 not found in text!")
        print(f"First 500 chars: {text[:500]}")
else:
    print("Could not find main document!")
