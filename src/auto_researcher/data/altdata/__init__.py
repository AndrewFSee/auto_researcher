"""
Alt-data adapter package.

Four cheap, legal public alt-data sleeves:

* :mod:`wikipedia_pageviews` — free REST API, lead indicator for retail attention.
* :mod:`sec_8k_realtime`     — EDGAR 8-K event stream with Item classification.
* :mod:`google_trends`       — search-interest z-scores via ``pytrends`` (optional dep).
* :mod:`reddit`              — WSB / r/investing post counts + FinBERT tone (optional dep).

All adapters share the :class:`~auto_researcher.data.altdata.base.AltDataAdapter`
protocol: ``fetch(tickers, start, end) -> pd.Series`` with a
``(date, ticker)`` MultiIndex and a normalized cross-sectional z-score.

Validation workflow
-------------------
Most alt-data signals don't survive an honest CPCV validation. The expected
path is: add the raw feed behind a config flag, run
``scripts/validate_signal_ic.py <adapter>`` to measure walk-forward IC with
deflated Sharpe, and *only then* enable by default in the composite. Having
the harness in place matters more than any single signal paying off.
"""

from __future__ import annotations

from .base import AltDataAdapter, AltDataCache, NormalizeMode, zscore_panel
from .wikipedia_pageviews import WikipediaPageviewsAdapter
from .sec_8k_realtime import SEC8KEventAdapter, EIGHT_K_ITEM_WEIGHTS

# Google Trends and Reddit require optional third-party packages; guard the
# imports so the rest of the altdata package remains usable without them.
try:
    from .google_trends import GoogleTrendsAdapter  # noqa: F401
    HAS_GOOGLE_TRENDS = True
except ImportError:
    HAS_GOOGLE_TRENDS = False

try:
    from .reddit import RedditMentionsAdapter  # noqa: F401
    HAS_REDDIT = True
except ImportError:
    HAS_REDDIT = False


__all__ = [
    "AltDataAdapter",
    "AltDataCache",
    "NormalizeMode",
    "zscore_panel",
    "WikipediaPageviewsAdapter",
    "SEC8KEventAdapter",
    "EIGHT_K_ITEM_WEIGHTS",
    "HAS_GOOGLE_TRENDS",
    "HAS_REDDIT",
]

if HAS_GOOGLE_TRENDS:
    __all__.append("GoogleTrendsAdapter")
if HAS_REDDIT:
    __all__.append("RedditMentionsAdapter")
