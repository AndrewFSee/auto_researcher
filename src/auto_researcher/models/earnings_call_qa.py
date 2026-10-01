"""
Earnings-call Q&A analyzer.

The analyst-management Q&A section of an earnings call is a richer signal
than the prepared remarks — prepared remarks are scripted and optimistic,
while Q&A reveals hedging, deflection, and the delta between question
specificity and answer specificity. This module extracts five features
from a transcript's Q&A section:

1. ``hedge_density`` — fraction of management answers containing hedge
   phrases ("we believe", "could", "potentially", "headwinds"). High values
   correlate with softer forward guidance.
2. ``answer_length_ratio`` — median answer tokens / median question tokens.
   Terse answers to long questions often indicate evasion.
3. ``sentiment_gap`` — (management tone) − (analyst tone). Computed with
   FinBERT when available, with a deterministic fallback so tests don't
   require the 400MB model.
4. ``guidance_specificity`` — count of numeric-range patterns per answer
   ("$3.2B to $3.5B", "8-10% growth"). A proxy for explicit forward
   guidance strength.
5. ``qoq_tone_delta`` — optional, computed if a prior-quarter transcript
   is supplied. Pure-mean FinBERT tone of management answers, current
   minus previous.

Reference:
    Tetlock (2007); Loughran & McDonald (2011); Hassan, Hollander, van
    Lent & Tahoun (2019) on "Firm-level political risk" Q&A parsing.

Usage::

    from auto_researcher.models.earnings_call_qa import (
        analyze_qa_transcript,
    )

    features = analyze_qa_transcript(transcript_text)
    # → {'hedge_density': 0.33, 'answer_length_ratio': 0.62, ...}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

HEDGE_PATTERNS = [
    r"\bwe believe\b",
    r"\bwe expect\b",
    r"\bwe think\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bpotentially\b",
    r"\bpossibl[ey]\b",
    r"\bhopefully\b",
    r"\bheadwinds?\b",
    r"\bdifficult\b",
    r"\buncertain(ty)?\b",
    r"\bchalleng(es|ing)\b",
    r"\bsubject to\b",
    r"\btoo early\b",
    r"\bhard to say\b",
    r"\bmixed\b",
    r"\bsoft(en)?\b",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# "$3.2B to $3.5B", "8-10%", "between $1.1 and $1.3", "10 to 12 percent",
# "8% and 10%". Each side of the range may carry its own unit suffix.
_NUM_WITH_UNIT = r"\$?\d+(?:\.\d+)?\s*(?:%|percent|[MBK])?"
_GUIDANCE_RE = re.compile(
    rf"{_NUM_WITH_UNIT}\s*(?:to|-|–|—|and|through)\s*{_NUM_WITH_UNIT}",
    re.IGNORECASE,
)

_POS_WORDS = {
    "strong", "growth", "record", "beat", "exceed", "robust", "accelerat",
    "momentum", "outperform", "improve", "healthy", "confident", "optimistic",
    "upside", "expansion",
}
_NEG_WORDS = {
    "weak", "miss", "decline", "slow", "headwind", "challenging", "difficult",
    "pressure", "soft", "contraction", "disappointing", "uncertainty",
    "downturn", "concern", "risk",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QATurn:
    """One question or answer turn in the call."""

    role: str  # "analyst" or "management"
    speaker: str
    text: str


@dataclass
class QAFeatures:
    """Per-call feature vector from Q&A analysis."""

    hedge_density: float = 0.0
    answer_length_ratio: float = 0.0
    sentiment_gap: float = 0.0
    guidance_specificity: float = 0.0
    qoq_tone_delta: float = 0.0
    n_questions: int = 0
    n_answers: int = 0
    backend: str = "lexicon"  # "lexicon" or "finbert"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Q&A splitting
# ---------------------------------------------------------------------------

# Many transcript formats mark the Q&A boundary with a heading such as
# "Questions and Answers" or "Q&A Session". Everything above is prepared
# remarks. We try the most specific pattern first.
_QA_HEADERS = [
    r"Question[\s-]and[\s-]Answer",
    r"Questions?\s+and\s+Answers?",
    r"Q\s*&\s*A(?:\s+Session)?",
    r"Question and Answer Session",
]
_QA_HEADER_RE = re.compile("|".join(_QA_HEADERS), re.IGNORECASE)

# Speaker labels typically look like "John Smith — Morgan Stanley — Analyst"
# or "Jane Doe, CFO:" or just "Operator:". We match any line that begins
# with a capital letter, contains no sentence-ending punctuation, and
# terminates with a colon — then pass the full label to _classify_role.
_SPEAKER_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9.,'\-—– ]+?)\s*:\s*$",
    re.MULTILINE,
)

_ANALYST_HINTS = ("analyst", "equity research", "capital", "llc", "securities",
                  "bank", "co.", "inc.", "research")
_MGMT_HINTS = ("ceo", "cfo", "coo", "chief", "president", "director",
               "treasurer", "head of", "vp ", "vice president")


def _classify_role(speaker_label: str) -> str:
    """Heuristically classify a speaker label as analyst or management."""
    s = speaker_label.lower()
    if any(h in s for h in _ANALYST_HINTS):
        return "analyst"
    if any(h in s for h in _MGMT_HINTS):
        return "management"
    # Operator / moderator — treat as neither (filtered out).
    if "operator" in s or "moderator" in s:
        return "other"
    return "management"  # default: in-house speaker


def _split_into_turns(qa_text: str) -> list[QATurn]:
    """Split the Q&A block into (speaker, role, text) turns."""
    turns: list[QATurn] = []
    # Find all speaker-label lines; each turn's text runs up to the next label.
    matches = list(_SPEAKER_LINE_RE.finditer(qa_text))
    if not matches:
        return turns

    for i, m in enumerate(matches):
        label = m.group(1).strip()
        role = _classify_role(label)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(qa_text)
        text = qa_text[start:end].strip()
        if role != "other" and text:
            turns.append(QATurn(role=role, speaker=label, text=text))
    return turns


def extract_qa_section(transcript: str) -> str:
    """Return the Q&A portion of ``transcript``; fall back to whole text."""
    m = _QA_HEADER_RE.search(transcript)
    if m:
        return transcript[m.end():]
    return transcript


# ---------------------------------------------------------------------------
# Feature primitives
# ---------------------------------------------------------------------------

def _hedge_density(answers: Iterable[str]) -> float:
    texts = list(answers)
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if _HEDGE_RE.search(t))
    return hits / len(texts)


def _length_ratio(questions: list[str], answers: list[str]) -> float:
    if not questions or not answers:
        return 0.0
    q_lens = sorted(len(q.split()) for q in questions)
    a_lens = sorted(len(a.split()) for a in answers)
    q_med = q_lens[len(q_lens) // 2] or 1
    a_med = a_lens[len(a_lens) // 2]
    return a_med / q_med


def _guidance_specificity(answers: Iterable[str]) -> float:
    texts = list(answers)
    if not texts:
        return 0.0
    hits = sum(len(_GUIDANCE_RE.findall(t)) for t in texts)
    return hits / len(texts)


def _lexicon_tone(text: str) -> float:
    """Simple +/-1 lexicon tone scaled to [-1, 1]. Deterministic fallback."""
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if any(t.startswith(p) for p in _POS_WORDS))
    neg = sum(1 for t in tokens if any(t.startswith(n) for n in _NEG_WORDS))
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


@lru_cache(maxsize=1)
def _get_finbert():
    """Lazy-load FinBERT pipeline; None if transformers unavailable."""
    try:
        from transformers import pipeline
    except ImportError:
        return None
    try:
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
        )
    except Exception as e:  # network / disk / OOM
        logger.warning("FinBERT unavailable (%s) — using lexicon fallback", e)
        return None


def _finbert_tone(text: str) -> float:
    """FinBERT score in [-1, 1] where negative=-1, neutral=0, positive=+1."""
    pipe = _get_finbert()
    if pipe is None:
        return _lexicon_tone(text)
    try:
        # FinBERT max sequence is 512; we chunk long answers.
        chunks = [text[i:i + 1500] for i in range(0, len(text), 1500)][:6]
        scores: list[float] = []
        for chunk in chunks:
            out = pipe(chunk, truncation=True)[0]
            score_map = {d["label"].lower(): d["score"] for d in out}
            scores.append(
                score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
            )
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.debug("FinBERT scoring failed (%s); falling back to lexicon", e)
        return _lexicon_tone(text)


def _mean_tone(texts: Iterable[str], use_finbert: bool) -> float:
    values = [_finbert_tone(t) if use_finbert else _lexicon_tone(t) for t in texts]
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_qa_transcript(
    transcript: str,
    prior_transcript: Optional[str] = None,
    use_finbert: bool = False,
) -> QAFeatures:
    """Compute Q&A features from a single transcript.

    Args:
        transcript: Full earnings-call transcript text.
        prior_transcript: Optional prior-quarter transcript; when supplied,
            ``qoq_tone_delta`` is computed as current − previous mean mgmt
            tone. Use this to surface *changes* in tone QoQ — the level
            is less informative than the delta.
        use_finbert: If True, use FinBERT for sentiment; otherwise use a
            fast lexicon fallback. FinBERT is lazy-loaded; if the model
            cannot be fetched the lexicon is used anyway.

    Returns:
        QAFeatures record with all five signals populated. Missing Q&A
        section → all zeros (caller can drop the row).
    """
    qa_section = extract_qa_section(transcript)
    turns = _split_into_turns(qa_section)

    questions = [t.text for t in turns if t.role == "analyst"]
    answers = [t.text for t in turns if t.role == "management"]

    feats = QAFeatures(
        hedge_density=_hedge_density(answers),
        answer_length_ratio=_length_ratio(questions, answers),
        sentiment_gap=(
            _mean_tone(answers, use_finbert)
            - _mean_tone(questions, use_finbert)
        ),
        guidance_specificity=_guidance_specificity(answers),
        n_questions=len(questions),
        n_answers=len(answers),
        backend="finbert" if use_finbert and _get_finbert() is not None else "lexicon",
    )

    if prior_transcript is not None:
        prior_qa = extract_qa_section(prior_transcript)
        prior_turns = _split_into_turns(prior_qa)
        prior_answers = [t.text for t in prior_turns if t.role == "management"]
        feats.qoq_tone_delta = (
            _mean_tone(answers, use_finbert)
            - _mean_tone(prior_answers, use_finbert)
        )

    return feats


def analyze_many(
    transcripts_by_ticker_date: dict[tuple[str, str], str],
    use_finbert: bool = False,
) -> "pd.DataFrame":
    """Vectorized batch wrapper → DataFrame indexed by (ticker, date).

    The QoQ delta is computed per ticker using the previous chronological
    call for that ticker, so the caller does not have to stitch pairs.
    """
    import pandas as pd

    rows = []
    by_ticker: dict[str, list[tuple[str, str]]] = {}
    for (tkr, date), _ in transcripts_by_ticker_date.items():
        by_ticker.setdefault(tkr, []).append((tkr, date))
    for tkr in by_ticker:
        by_ticker[tkr].sort(key=lambda kv: kv[1])

    for tkr, pairs in by_ticker.items():
        prev_text = None
        for tkr_, date in pairs:
            text = transcripts_by_ticker_date[(tkr_, date)]
            feats = analyze_qa_transcript(
                text, prior_transcript=prev_text, use_finbert=use_finbert
            )
            rows.append({"ticker": tkr_, "date": date, **feats.to_dict()})
            prev_text = text

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["ticker", "date"]).sort_index()
    return df
