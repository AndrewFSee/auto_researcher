"""
Test the Early Adopter Model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_researcher.models.early_adopter import (
    EarlyAdopterModel,
    find_early_adopters,
    get_genai_pioneers,
)


def test_model():
    """Test the Early Adopter Model."""
    
    print("=" * 70)
    print("EARLY ADOPTER MODEL TEST")
    print("=" * 70)
    
    # Initialize model
    model = EarlyAdopterModel()
    
    # Define test universe
    tickers = [
        # Big Tech
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
        # Semiconductors
        "AMD", "INTC", "AVGO", "QCOM",
        # Software
        "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
        # Enterprise
        "IBM", "CSCO", "DELL",
        # Non-tech
        "JPM", "WMT", "KO",
    ]
    
    print(f"\nBuilding industry timelines for {len(tickers)} tickers...")
    timelines = model.build_industry_timelines(tickers)
    
    print(f"\nFound {len(timelines)} technologies with adoption data:")
    for tech_id, timeline in sorted(timelines.items(), key=lambda x: x[1].first_date):
        print(f"  {tech_id}: First={timeline.first_adopter} ({timeline.first_date.strftime('%Y-%m')}), "
              f"Median={timeline.median_date.strftime('%Y-%m')}, Adopters={timeline.adoption_count}")
    
    # Rank all pioneers
    print("\n" + "=" * 70)
    print("PIONEER RANKINGS")
    print("=" * 70)
    
    signals = model.rank_pioneers(tickers)
    
    print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Techs':<8} {'Early':<8} {'Avg Lead':<12} {'Signal'}")
    print("-" * 70)
    
    for i, s in enumerate(signals[:15], 1):
        print(f"{i:<5} {s.ticker:<8} {s.pioneer_score:.2f}    {s.total_techs_adopted:<8} "
              f"{s.techs_adopted_early:<8} {s.avg_lead_months:>+8.1f} mo   {s.signal}")
    
    # GenAI Pioneers
    print("\n" + "=" * 70)
    print("GENAI PIONEERS")
    print("=" * 70)
    
    genai_signals = [s for s in signals if s.genai_adopted]
    genai_signals.sort(key=lambda x: -x.genai_score)
    
    print(f"\n{'Ticker':<8} {'GenAI Score':<15} {'Terms Adopted'}")
    print("-" * 60)
    
    for s in genai_signals[:10]:
        terms = ", ".join(s.genai_adopted[:3])
        print(f"{s.ticker:<8} {s.genai_score:>+10.1f} mo    {terms}")
    
    # Detailed analysis of top pioneer
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: TOP PIONEER")
    print("=" * 70)
    
    top = signals[0]
    print(model.get_summary(top))
    
    # First movers for key technologies
    print("\n" + "=" * 70)
    print("FIRST MOVERS BY TECHNOLOGY")
    print("=" * 70)
    
    key_techs = ["large_language_model", "generative_ai", "copilot", "inference"]
    
    for tech_id in key_techs:
        if tech_id in timelines:
            print(f"\n{tech_id.replace('_', ' ').upper()}:")
            movers = model.get_first_movers(tech_id)
            for ticker, date, lead in movers[:5]:
                print(f"  {ticker}: {date.strftime('%Y-%m')} ({lead:+.0f} mo vs median)")
    
    # Currently emerging technologies
    print("\n" + "=" * 70)
    print("CURRENTLY EMERGING TECHNOLOGIES (Maturity=1)")
    print("=" * 70)
    
    emerging = model.get_emerging_now()
    print(f"\nThese are the most relevant for detecting early adopters NOW:")
    for tech_id in emerging:
        config = model.lexicon[tech_id]
        print(f"  {tech_id}: {config['category']} (emerged {config['emerged']})")
    
    # Convenience function tests
    print("\n" + "=" * 70)
    print("CONVENIENCE FUNCTION: find_early_adopters()")
    print("=" * 70)
    
    early_adopters = find_early_adopters(tickers, min_pioneer_score=0.6)
    print(f"\nCompanies with Pioneer Score >= 0.6:")
    for s in early_adopters:
        print(f"  {s.ticker}: {s.pioneer_score:.2f} ({s.signal})")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
