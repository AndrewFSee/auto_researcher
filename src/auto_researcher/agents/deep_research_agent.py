"""
Deep Research Agent — GPT-Researcher integration for stock deep dives.

Uses the gpt-researcher library to conduct comprehensive web-based research
on individual stocks. Produces markdown research reports with sources.

Usage:
    agent = DeepResearchAgent()
    result = await agent.research_stock("AAPL", context={"signal": "strong_buy", ...})
    # result.report  ->  markdown string
    # result.sources ->  list of URLs
    # result.costs   ->  float (estimated token cost)

Requirements:
    pip install gpt-researcher
    Environment variables: OPENAI_API_KEY, TAVILY_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepResearchConfig:
    """Configuration for deep research runs."""

    # Research type: "research_report" for standard, "deep" for deep recursive
    report_type: str = "research_report"

    # Deep research parameters (only used when report_type="deep")
    deep_breadth: int = 4  # Parallel research paths per level
    deep_depth: int = 2    # Recursive depth levels
    deep_concurrency: int = 4  # Max concurrent research ops

    # Report options
    tone: str = "formal and objective"
    max_subtopics: int = 5
    total_words: int = 2000

    # Controls
    verbose: bool = False
    timeout: int = 300  # 5 min default for deep research


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StockResearchResult:
    """Result from a single stock deep-research run."""

    ticker: str
    query: str
    report: str = ""
    sources: list[str] = field(default_factory=list)
    costs: float = 0.0
    latency_ms: int = 0
    error: str | None = None


@dataclass
class DeepResearchSummary:
    """Summary of a batch deep-research run."""

    results: list[StockResearchResult] = field(default_factory=list)
    total_costs: float = 0.0
    total_latency_ms: int = 0


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_stock_query(
    ticker: str,
    company_name: str = "",
    signal: str = "",
    composite_score: float = 0.0,
    sector: str = "",
    extra_context: str = "",
) -> str:
    """Build a focused research query for a stock."""
    parts = [
        f"Comprehensive investment research report on {ticker}",
    ]
    if company_name:
        parts[0] += f" ({company_name})"

    parts.append(
        "covering: recent news and catalysts, financial performance and outlook, "
        "competitive positioning, key risks, analyst sentiment, and insider activity."
    )

    if sector:
        parts.append(f"The company operates in the {sector} sector.")

    if signal and composite_score:
        parts.append(
            f"Our quantitative model currently rates this stock as '{signal}' "
            f"with a composite score of {composite_score:+.3f}. "
            "Evaluate whether recent qualitative developments support or "
            "contradict this quantitative signal."
        )

    if extra_context:
        parts.append(extra_context)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Deep Research Agent
# ---------------------------------------------------------------------------

class DeepResearchAgent:
    """
    Wraps gpt-researcher to do deep web research on stocks.

    Supports both standard and deep (recursive) research modes.
    """

    def __init__(self, config: DeepResearchConfig | None = None):
        self.config = config or DeepResearchConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Verify gpt-researcher is importable."""
        try:
            from gpt_researcher import GPTResearcher  # noqa: F401
        except ImportError:
            raise ImportError(
                "gpt-researcher is required for deep research. "
                "Install it with: pip install gpt-researcher"
            )

    async def research_stock(
        self,
        ticker: str,
        company_name: str = "",
        signal: str = "",
        composite_score: float = 0.0,
        sector: str = "",
        extra_context: str = "",
    ) -> StockResearchResult:
        """
        Run deep research on a single stock.

        Returns a StockResearchResult with the markdown report and sources.
        """
        from gpt_researcher import GPTResearcher

        query = build_stock_query(
            ticker=ticker,
            company_name=company_name,
            signal=signal,
            composite_score=composite_score,
            sector=sector,
            extra_context=extra_context,
        )

        result = StockResearchResult(ticker=ticker, query=query)
        t0 = time.time()

        try:
            researcher = GPTResearcher(
                query=query,
                report_type=self.config.report_type,
                tone=self.config.tone,
                max_subtopics=self.config.max_subtopics,
                verbose=self.config.verbose,
            )

            # Conduct research (web scraping + analysis)
            await researcher.conduct_research()

            # Generate the report
            report = await researcher.write_report()

            result.report = report or ""
            result.sources = researcher.get_source_urls() or []
            result.costs = researcher.get_costs() or 0.0

        except Exception as e:
            logger.error("Deep research failed for %s: %s", ticker, e)
            result.error = str(e)

        result.latency_ms = int((time.time() - t0) * 1000)
        return result

    async def research_batch(
        self,
        stocks: list[dict[str, Any]],
        max_concurrent: int = 2,
    ) -> DeepResearchSummary:
        """
        Research multiple stocks with concurrency control.

        Each item in stocks should be a dict with at least 'ticker',
        and optionally 'company_name', 'signal', 'composite_score', 'sector'.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        summary = DeepResearchSummary()

        async def _research_one(stock: dict) -> StockResearchResult:
            async with semaphore:
                return await self.research_stock(
                    ticker=stock.get("ticker", ""),
                    company_name=stock.get("company_name", ""),
                    signal=stock.get("signal", ""),
                    composite_score=stock.get("composite_score", 0.0),
                    sector=stock.get("sector", ""),
                )

        tasks = [_research_one(s) for s in stocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error("Research task failed: %s", r)
                continue
            summary.results.append(r)
            summary.total_costs += r.costs
            summary.total_latency_ms += r.latency_ms

        return summary

    def research_stock_sync(self, **kwargs) -> StockResearchResult:
        """Synchronous wrapper for research_stock."""
        return asyncio.run(self.research_stock(**kwargs))

    def research_batch_sync(self, stocks: list[dict], **kwargs) -> DeepResearchSummary:
        """Synchronous wrapper for research_batch."""
        return asyncio.run(self.research_batch(stocks, **kwargs))
