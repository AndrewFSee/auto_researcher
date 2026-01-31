#!/usr/bin/env python3
"""Test the thematic agent with early adopter model integration."""

import sys
sys.path.insert(0, "c:/Users/Andrew/projects/auto_researcher/src")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from auto_researcher.agents.thematic_agent import ThematicAnalysisAgent

def main():
    print("=" * 70)
    print("THEMATIC AGENT WITH EARLY ADOPTER MODEL TEST")
    print("=" * 70)
    
    # Initialize the agent with early adopter model enabled
    agent = ThematicAnalysisAgent(
        use_sector_momentum_model=True,
        use_early_adopter_model=True
    )
    
    # Test with a known tech pioneer
    test_tickers = ["NVDA", "MSFT"]
    
    for ticker in test_tickers:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {ticker}")
        print("=" * 70)
        
        result = agent.analyze_ticker(ticker)
        
        print(f"\n📊 BASIC INFO:")
        print(f"   Sector: {result.sector}")
        print(f"   Primary Themes: {result.primary_themes}")
        
        print(f"\n📈 FORWARD OUTLOOK:")
        print(f"   Forward Score: {result.forward_score:.2f}")
        print(f"   Forward Signal: {result.forward_signal}")
        
        print(f"\n🏭 SECTOR MOMENTUM:")
        print(f"   {result.sm_summary or 'N/A'}")
        
        print(f"\n🚀 EARLY ADOPTER SIGNAL:")
        print(f"   Pioneer Score: {result.ea_pioneer_score or 'N/A'}")
        print(f"   Techs Adopted: {result.ea_techs_adopted or 'N/A'}")
        print(f"   Techs Early: {result.ea_techs_early or 'N/A'}")
        print(f"   Avg Lead (months): {result.ea_avg_lead_months or 'N/A'}")
        print(f"   GenAI Score: {result.ea_genai_score or 'N/A'}")
        print(f"   Signal: {result.ea_signal or 'N/A'}")
        print(f"   Top Adoptions: {result.ea_top_adoptions or 'N/A'}")
        print(f"\n   Summary: {result.ea_summary or 'N/A'}")
        
        print(f"\n💡 OVERALL SUMMARY:")
        print(f"   {result.summary}")
        
        print(f"\n   Catalysts: {result.key_catalysts}")
        print(f"   Risks: {result.key_risks}")
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
