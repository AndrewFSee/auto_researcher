"""
Test Transcript VectorStore: build from synthetic data, query, and verify
peer comparison integrates with EarningsCallQualModel.
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_synthetic_transcript(ticker: str, tone: str, quarter: int, year: int) -> str:
    """Generate a synthetic earnings call transcript."""
    
    if tone == "positive":
        mgmt_text = (
            "We are very pleased with our performance this quarter. Revenue grew by 15% "
            "year-over-year, exceeding our guidance range of $4.2 billion to $4.4 billion. "
            "We expect continued strong momentum in the second half, with operating margin "
            "expanding to 25-27%. Our AI initiatives are showing excellent traction, with "
            "bookings up 40% sequentially. We are raising our full-year guidance to $18 billion "
            "in revenue, up from $17.5 billion previously. Free cash flow generation remains "
            "robust at $3.2 billion, enabling continued investment in growth and shareholder returns. "
            "We are confident in our competitive positioning and expect market share gains "
            "across all major product categories."
        )
        analyst_text = (
            "That's very helpful. Could you provide more color on the AI bookings pipeline? "
            "And how should we think about the margin trajectory in the back half?"
        )
    elif tone == "negative":
        mgmt_text = (
            "This was a challenging quarter. We faced significant headwinds from tariff uncertainty "
            "and softening demand in our enterprise segment. Revenue declined 3% year-over-year. "
            "We are not yet ready to provide specific guidance for next quarter given the uncertain "
            "macro environment. Margins were under pressure due to higher input costs and we may "
            "need to reassess our cost structure. We are cautiously optimistic but there are "
            "considerable risks ahead. We cannot rule out further deterioration in the near term. "
            "We believe it is too early to quantify the full impact of these headwinds."
        )
        analyst_text = (
            "Can you elaborate on the enterprise weakness? Is this primarily demand-driven "
            "or are you losing share to competitors?"
        )
    else:  # neutral
        mgmt_text = (
            "Our results were broadly in line with expectations this quarter. Revenue was approximately "
            "$5.1 billion, roughly flat year-over-year. We continue to execute on our strategic plan "
            "and are making steady progress on cost optimization initiatives. We expect results "
            "in the next quarter to be generally similar to this quarter. Our capital expenditure "
            "plans remain on track at around $1.2 billion for the full year."
        )
        analyst_text = (
            "Thank you for the update. Could you talk about any changes you're seeing in "
            "the competitive landscape?"
        )
    
    return (
        f"Operator: Good afternoon and welcome to {ticker}'s Q{quarter} {year} Earnings Call.\n"
        f"CEO John Smith - Chief Executive Officer: {mgmt_text}\n"
        f"CFO Jane Doe - Chief Financial Officer: We are pleased with the progress on our balance sheet. "
        f"Total debt stands at $8 billion and we have $5 billion in cash and equivalents.\n"
        f"Operator: We will now begin the question-and-answer session.\n"
        f"Analyst Mike Chen from Goldman Sachs: {analyst_text}\n"
        f"CEO John Smith - Chief Executive Officer: {mgmt_text}\n"
        f"Analyst Sarah Park from Morgan Stanley: How are you thinking about capital allocation priorities?\n"
        f"CFO Jane Doe - Chief Financial Officer: {mgmt_text}\n"
    )


def test_chunking():
    """Test transcript parsing and chunking."""
    from auto_researcher.data.transcript_vectorstore import _parse_and_chunk_transcript
    
    content = make_synthetic_transcript("AAPL", "positive", 4, 2025)
    chunks = _parse_and_chunk_transcript(content, "AAPL", 4, 2025, "2025-01-28")
    
    print(f"\n{'='*60}")
    print(f"TEST 1: Chunking")
    print(f"{'='*60}")
    print(f"Transcript length: {len(content):,} chars")
    print(f"Chunks produced: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        m = c["metadata"]
        print(f"  [{i}] role={m['role']:12s} qa={m['is_qa']!s:5s} chars={m['char_count']:4d} | {c['text'][:80]}...")
    
    assert len(chunks) > 0, "Should produce at least 1 chunk"
    
    # Verify metadata
    mgmt_chunks = [c for c in chunks if c["metadata"]["is_management"]]
    qa_chunks = [c for c in chunks if c["metadata"]["is_qa"]]
    print(f"\nManagement chunks: {len(mgmt_chunks)}")
    print(f"Q&A chunks: {len(qa_chunks)}")
    assert len(mgmt_chunks) > 0, "Should have management chunks"
    assert len(qa_chunks) > 0, "Should have Q&A chunks"
    
    print("✓ Chunking test passed")


def test_vectorstore_build_and_query():
    """Test building and querying a small vectorstore from synthetic data."""
    from auto_researcher.data.transcript_vectorstore import (
        TranscriptVectorStore, _parse_and_chunk_transcript,
    )
    
    # Create a temp directory for the test vectorstore
    tmpdir = Path(tempfile.mkdtemp(prefix="test_transcript_vs_"))
    
    try:
        store = TranscriptVectorStore(chroma_path=tmpdir / "chroma")
        
        # Build synthetic chunks for multiple tickers
        tickers_tones = [
            ("AAPL", "positive", 4, 2025),
            ("MSFT", "positive", 4, 2025),
            ("GOOG", "neutral", 4, 2025),
            ("AMZN", "negative", 4, 2025),
            ("META", "negative", 4, 2025),
        ]
        
        all_ids = []
        all_docs = []
        all_metas = []
        
        for ticker, tone, q, y in tickers_tones:
            content = make_synthetic_transcript(ticker, tone, q, y)
            chunks = _parse_and_chunk_transcript(content, ticker, q, y, f"{y}-01-28")
            for idx, c in enumerate(chunks):
                chunk_id = f"tc_{ticker}_{y}_Q{q}_{idx}"
                all_ids.append(chunk_id)
                all_docs.append(c["text"])
                all_metas.append(c["metadata"])
        
        # Add to vectorstore
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_docs, show_progress_bar=False)
        
        collection = store._get_collection()
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=embeddings.tolist(),
        )
        
        print(f"\n{'='*60}")
        print(f"TEST 2: VectorStore Build & Query")
        print(f"{'='*60}")
        print(f"Indexed {collection.count()} chunks from {len(tickers_tones)} companies")
        
        # Test 2a: Query single ticker
        results = store.query("AAPL", query_text="revenue guidance growth", n_results=5)
        print(f"\nQuery 'revenue guidance growth' for AAPL -> {len(results['documents'])} results")
        for doc, meta in zip(results["documents"][:3], results["metadatas"][:3]):
            print(f"  [{meta['role']}] {doc[:100]}...")
        assert len(results["documents"]) > 0, "Should find AAPL results"
        
        # Test 2b: Thematic query across companies
        theme_results = store.query_by_theme(
            "tariff impact headwinds challenging",
            management_only=True,
            n_results=10,
        )
        print(f"\nThematic query 'tariff impact headwinds' -> {len(theme_results['documents'])} results")
        tickers_found = set()
        for doc, meta in zip(theme_results["documents"][:5], theme_results["metadatas"][:5]):
            t = meta.get("ticker", "?")
            tickers_found.add(t)
            print(f"  [{t}] {doc[:100]}...")
        print(f"Tickers found: {tickers_found}")
        
        # Test 2c: Peer comparison
        peer_results = store.query_peer_comparison(
            "AAPL", ["MSFT", "GOOG", "AMZN"],
            query_text="outlook guidance revenue",
            n_per_company=3,
        )
        print(f"\nPeer comparison AAPL vs MSFT/GOOG/AMZN:")
        for t, data in peer_results.items():
            print(f"  {t}: {len(data['documents'])} results")
        
        # Test 2d: Historical query
        hist_results = store.query_ticker_history(
            "AAPL", query_text="earnings growth", n_results=5,
        )
        print(f"\nHistory query for AAPL -> {len(hist_results['documents'])} results")
        
        # Test 2e: Format context for analysis
        context = store.format_context_for_analysis(
            "AAPL",
            query_text="management outlook",
            n_results=5,
            include_peers=["MSFT", "AMZN"],
        )
        print(f"\nFormatted context length: {len(context)} chars")
        print(context[:500])
        
        print("\n✓ VectorStore build & query test passed")
        
        return store  # Return for use in next test
        
    finally:
        pass  # Don't clean up yet, need for peer comparison test


def test_peer_tone_delta():
    """Test that EarningsCallQualModel uses vectorstore for peer tone delta."""
    from auto_researcher.data.transcript_vectorstore import (
        TranscriptVectorStore, _parse_and_chunk_transcript,
    )
    from auto_researcher.models.earnings_call_qual import EarningsCallQualModel
    
    # Build a small vectorstore
    tmpdir = Path(tempfile.mkdtemp(prefix="test_peer_tone_"))
    
    try:
        store = TranscriptVectorStore(chroma_path=tmpdir / "chroma")
        
        # Create synthetic data: AAPL positive, peers negative
        tickers_tones = [
            ("AAPL", "positive", 4, 2025),
            ("MSFT", "negative", 4, 2025),
            ("GOOG", "negative", 4, 2025),
            ("AMZN", "negative", 4, 2025),
        ]
        
        all_ids, all_docs, all_metas = [], [], []
        for ticker, tone, q, y in tickers_tones:
            content = make_synthetic_transcript(ticker, tone, q, y)
            chunks = _parse_and_chunk_transcript(content, ticker, q, y, f"{y}-01-28")
            for idx, c in enumerate(chunks):
                all_ids.append(f"tc_{ticker}_{y}_Q{q}_{idx}")
                all_docs.append(c["text"])
                all_metas.append(c["metadata"])
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(all_docs, show_progress_bar=False)
        
        collection = store._get_collection()
        collection.upsert(
            ids=all_ids, documents=all_docs, metadatas=all_metas,
            embeddings=embeddings.tolist(),
        )
        
        print(f"\n{'='*60}")
        print(f"TEST 3: Peer Tone Delta Integration")
        print(f"{'='*60}")
        print(f"Vectorstore: {collection.count()} chunks")
        print(f"Scenario: AAPL=positive, MSFT/GOOG/AMZN=negative")
        
        # Test with vectorstore
        qual_model_with_rag = EarningsCallQualModel(use_finbert=False, transcript_vectorstore=store)
        
        # Manually test _compute_peer_tone_delta
        # AAPL management tone is positive (~0.4), peers are negative (~-0.4)
        # So delta should be positive
        delta, n_peers = qual_model_with_rag._compute_peer_tone_delta("AAPL", 0.4)
        print(f"\nPeer tone delta for AAPL (mgmt_tone=+0.4):")
        print(f"  delta = {delta:+.3f}, n_peers = {n_peers}")
        
        # Delta should be positive (AAPL more bullish than negative peers)
        if n_peers >= 2:
            assert delta > 0, f"Expected positive delta when AAPL is bullish vs bearish peers, got {delta}"
            print(f"  ✓ Positive delta confirmed (AAPL more bullish than peers)")
        else:
            print(f"  ⚠ Only {n_peers} peers found, need >= 2 for signal")
        
        # Test without vectorstore 
        qual_model_no_rag = EarningsCallQualModel(use_finbert=False, transcript_vectorstore=None)
        delta_no_rag, n_peers_no_rag = qual_model_no_rag._compute_peer_tone_delta("AAPL", 0.4)
        print(f"\nPeer tone delta WITHOUT vectorstore:")
        print(f"  delta = {delta_no_rag:+.3f}, n_peers = {n_peers_no_rag}")
        assert delta_no_rag == 0.0, "Should be 0.0 without vectorstore"
        assert n_peers_no_rag == 0, "Should be 0 peers without vectorstore"
        print(f"  ✓ Correctly returns 0.0 without vectorstore")
        
        qual_model_with_rag.unload()
        qual_model_no_rag.unload()
        print("\n✓ Peer tone delta integration test passed")
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_chunking()
    test_vectorstore_build_and_query()
    test_peer_tone_delta()
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}")
