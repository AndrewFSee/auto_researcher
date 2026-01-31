"""Debug earnings call analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.models.earnings_tech_signal import EarningsCallTechModel

model = EarningsCallTechModel()

# Check raw transcripts first
print("Checking raw transcript data...")
transcripts = model.client.get_transcripts("NVDA", limit=2)
print(f"Got {len(transcripts)} transcripts")

if transcripts:
    t = transcripts[0]
    print(f"Date: {t['date']}")
    print(f"Quarter: {t['quarter']}")
    print(f"Year: {t['year']}")
    print(f"Content length: {len(t['content'])} chars")
    print(f"Content preview: {t['content'][:500]}...")

print()
print("Running full analysis...")
signal = model.analyze_company("NVDA", quarters=2)

print(f"Current analysis: {signal.current}")
if signal.current:
    print(f"Date: {signal.current.date}")
    print(f"Quarter: {signal.current.quarter}")
    print(f"Tech intensity: {signal.current.tech_intensity}")
    print(f"Mentions: {signal.current.mentions_by_area}")
else:
    print("No current analysis")
