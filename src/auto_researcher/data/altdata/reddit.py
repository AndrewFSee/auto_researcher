"""
Reddit mentions adapter — WSB / r/investing post counts + FinBERT tone.

Why
---
Retail sentiment on r/wallstreetbets and r/investing has been shown to
lead short-term price action in the meme-era regime (GME, AMC, etc.).
The value is NOT so much "predict GME-style squeezes" as "detect the
1-sigma attention spike *before* the headline coverage kicks in" —
retail posts lead mainstream news by hours to days for smaller caps.

Signal
------
For each (ticker, date), we emit::

    mentions_weighted = sum(score_i * tone_i for post_i on date)

where ``score_i`` is the post's upvote score (logged) and ``tone_i`` is
the FinBERT tone of ``title + body``. Cross-sectional z-score is
applied downstream.

Dependencies
------------
Requires ``praw`` (Reddit client) + valid Reddit API credentials via
``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` / ``REDDIT_USER_AGENT``
env vars, or passed explicitly to the constructor. Also uses the
repo-local FinBERT via
:func:`auto_researcher.agents.finbert_sentiment.analyze_text`.

Historical coverage caveat
--------------------------
PRAW only exposes the last ~1000 posts per subreddit search, so this
adapter is **near-real-time only**. For historical backfills you want
PushShift dumps or a commercial archive; we deliberately don't bake
that in because PushShift has been unreliable since 2023.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .base import AltDataCache, FetchWindow, empty_altdata_series

logger = logging.getLogger(__name__)

try:
    import praw
    HAS_PRAW = True
except ImportError as _e:
    HAS_PRAW = False
    praw = None
    _PRAW_ERR = _e
else:
    _PRAW_ERR = None

# FinBERT is optional — if the helper is missing we fall back to neutral
# tone, so mention counts still work on their own.
try:
    from auto_researcher.agents.finbert_sentiment import analyze_text as _finbert_tone
    HAS_FINBERT = True
except Exception as _e:
    HAS_FINBERT = False
    _finbert_tone = None


DEFAULT_SUBREDDITS = ("wallstreetbets", "investing", "stocks")


class RedditMentionsAdapter:
    """Count weighted Reddit mentions per ticker per day."""

    name = "reddit_mentions"

    def __init__(
        self,
        subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
        ticker_aliases: Optional[Mapping[str, list[str]]] = None,
        cache_dir: Optional[Path] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "auto-researcher-altdata/1.0",
        limit_per_ticker: int = 200,
        use_finbert: bool = True,
        reddit_client: Optional[object] = None,
        sleep_between_tickers: float = 1.0,
    ):
        if not HAS_PRAW and reddit_client is None:
            raise ImportError(
                "praw is required for RedditMentionsAdapter. "
                f"Install with `pip install praw`. (import error: {_PRAW_ERR!r})"
            )
        self.subreddits = list(subreddits)
        self.ticker_aliases = {k.upper(): list(v) for k, v in (ticker_aliases or {}).items()}
        self.cache = AltDataCache(adapter=self.name, cache_dir=cache_dir)
        self.limit_per_ticker = limit_per_ticker
        self.use_finbert = use_finbert and HAS_FINBERT
        self.sleep_between_tickers = sleep_between_tickers

        if reddit_client is not None:
            self._reddit = reddit_client
        else:
            self._reddit = praw.Reddit(
                client_id=client_id or os.environ.get("REDDIT_CLIENT_ID"),
                client_secret=client_secret or os.environ.get("REDDIT_CLIENT_SECRET"),
                user_agent=os.environ.get("REDDIT_USER_AGENT", user_agent),
                read_only=True,
            )

    def fetch(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> pd.Series:
        window = FetchWindow.from_inputs(start, end)
        iso_start, iso_end = window.iso()

        rows: list[tuple[pd.Timestamp, str, float]] = []
        for ticker in tickers:
            aliases = [ticker.upper()] + self.ticker_aliases.get(ticker.upper(), [])
            cache_extra = "|".join(aliases)

            cached = AltDataCache.unwrap(
                self.cache.get(ticker, iso_start, iso_end, extra=cache_extra)
            )
            if cached is None:
                cached = self._fetch_mentions(aliases)
                if cached is not None:
                    self.cache.put(ticker, iso_start, iso_end, cached, extra=cache_extra)
                else:
                    continue
                time.sleep(self.sleep_between_tickers)

            # Group by day: sum(log1p(score) * tone).
            for entry in cached:
                try:
                    dt = pd.Timestamp(entry["created_utc"], unit="s").normalize()
                except (ValueError, TypeError):
                    continue
                if dt < window.start or dt > window.end:
                    continue
                score_log = float(np.log1p(max(int(entry.get("score", 0)), 0)))
                tone = float(entry.get("tone", 0.0))
                rows.append((dt, ticker, score_log * tone if tone != 0 else score_log))

        if not rows:
            return empty_altdata_series(self.name)

        df = pd.DataFrame(rows, columns=["date", "ticker", "weighted"])
        out = df.groupby(["date", "ticker"], as_index=True)["weighted"].sum()
        out.name = self.name
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_mentions(self, aliases: list[str]) -> Optional[list[dict]]:
        """Search each subreddit for any alias — returns list of post dicts."""
        try:
            out: list[dict] = []
            seen_ids: set[str] = set()
            for sub_name in self.subreddits:
                subreddit = self._reddit.subreddit(sub_name)
                for alias in aliases:
                    for post in subreddit.search(
                        alias, sort="new", limit=self.limit_per_ticker,
                    ):
                        pid = getattr(post, "id", None)
                        if pid is None or pid in seen_ids:
                            continue
                        seen_ids.add(pid)
                        title = getattr(post, "title", "") or ""
                        body = getattr(post, "selftext", "") or ""
                        tone = self._tone(f"{title}. {body}") if self.use_finbert else 0.0
                        out.append({
                            "id": pid,
                            "created_utc": getattr(post, "created_utc", 0),
                            "score": getattr(post, "score", 0),
                            "title": title,
                            "tone": tone,
                        })
            return out
        except Exception as e:
            logger.warning("reddit mentions fetch failed (%s): %s", aliases, e)
            return None

    def _tone(self, text: str) -> float:
        if not text or not self.use_finbert or _finbert_tone is None:
            return 0.0
        try:
            return float(_finbert_tone(text))
        except Exception:
            return 0.0
