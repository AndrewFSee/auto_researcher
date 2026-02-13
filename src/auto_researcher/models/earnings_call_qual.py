"""
Qualitative Earnings Call Analysis Model
=========================================

Analyzes earnings call TRANSCRIPT TEXT for qualitative management signals
that predict future stock performance. This fills the gap between:
- PEAD model (purely quantitative: EPS/revenue surprise numbers)
- Filing Tone model (10-K annual filings, not earnings calls)
- Early Adopter model (technology keyword counting, not sentiment/tone)
- Sentiment agent (news headlines, not transcript text)

Academic Basis:
- Brockman & Cicon (2013): Management tone in calls predicts returns, IC ≈ 0.04-0.06
- Price et al. (2012): Q&A section tone more informative than prepared remarks
- Matsumoto et al. (2011): Analyst-management tone gap predicts earnings quality
- Li (2010): Uncertainty language predicts future earnings volatility
- Davis et al. (2015): Tone change QoQ is more informative than tone level

Signal Components:
1. Management Tone (FinBERT on management answers in Q&A)
2. Analyst-Management Tone Gap (divergence signal)
3. Hedging/Uncertainty Language (Loughran-McDonald)
4. Guidance Specificity (concrete numbers vs vague language)
5. QoQ Tone Change (sequential improvement/deterioration)

Output: -1 to +1 composite score, combined from the 5 sub-signals.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

QUAL_CONFIG = {
    # Weights for sub-signal combination (6 sub-signals)
    "management_tone_weight": 0.25,       # FinBERT sentiment of management answers
    "tone_gap_weight": 0.12,              # Management vs analyst divergence
    "hedging_weight": 0.13,               # Uncertainty/hedging language penalty
    "guidance_specificity_weight": 0.17,  # Concrete vs vague forward guidance
    "tone_change_weight": 0.18,           # QoQ tone improvement/deterioration
    "peer_tone_delta_weight": 0.15,       # Tone vs sector peers (RAG-powered)

    # Staleness: calls older than this get decayed
    "max_call_age_days": 100,             # ~1 quarter + buffer
    "decay_half_life_days": 45,           # Signal decays over 45 days

    # Thresholds for signal interpretation
    "strong_threshold": 0.5,              # |score| >= 0.5 -> strong signal
    "weak_threshold": 0.15,               # |score| >= 0.15 -> weak signal

    # Text analysis parameters
    "min_qa_sentences": 10,               # Minimum Q&A sentences for reliable analysis
    "max_sentences_per_section": 200,     # Cap to avoid OOM on very long calls

    # Peer comparison config
    "peer_compare_n_passages": 5,         # Passages per peer to retrieve
    "min_peers_for_signal": 2,            # Need at least 2 peers for meaningful comparison
}


# ==============================================================================
# HEDGING & UNCERTAINTY LEXICON (Loughran-McDonald + extensions)
# ==============================================================================
# Subset of LM uncertainty words most relevant to earnings calls,
# plus additional hedging phrases common in management language.

HEDGING_WORDS = {
    # Core LM uncertainty
    "approximately", "approximately", "assume", "assumed", "assuming",
    "assumption", "assumptions", "believe", "believed", "believes",
    "cautious", "cautiously", "conceivable", "conceivably",
    "conditional", "conditionally", "contingency", "contingent",
    "could", "depend", "dependent", "depending", "depends",
    "doubt", "doubtful", "estimate", "estimated", "estimates",
    "eventual", "eventually", "expect", "expected", "expecting",
    "exposure", "exposures", "fluctuate", "fluctuated",
    "fluctuating", "fluctuation", "fluctuations",
    "generally", "hope", "hoped", "hopefully", "hoping",
    "if", "imprecise", "imprecision", "indefinite", "indefinitely",
    "indicate", "indicated", "indicates", "indicating",
    "inherent", "inherently", "intend", "intended", "intending",
    "likelihood", "likely",
    "may", "maybe", "might",
    "nearly", "occasionally",
    "pending", "perhaps", "plan", "planned", "planning",
    "possible", "possibly", "potential", "potentially",
    "predict", "predicted", "predicting", "prediction", "predictions",
    "preliminary", "presumably", "probable", "probably",
    "project", "projected", "projecting", "projection", "projections",
    "prospect", "prospects",
    "risk", "risked", "risking", "risks", "risky",
    "roughly", "seem", "seemed", "seemingly", "seems",
    "should", "sometimes", "somewhat",
    "suggest", "suggested", "suggesting", "suggests",
    "susceptible",
    "tend", "tended", "tends",
    "tentative", "tentatively",
    "uncertain", "uncertainty", "unclear",
    "unpredictable", "unpredictability",
    "unsure", "variability", "variable", "varies", "volatile",
    "volatility", "would",
}

# Strong hedging phrases that indicate low confidence
STRONG_HEDGING_PHRASES = [
    "at this time", "at this point", "it's too early to",
    "difficult to predict", "hard to predict",
    "difficult to quantify", "hard to quantify",
    "remains to be seen", "wait and see",
    "we'll have to see", "time will tell",
    "subject to change", "no guarantees",
    "cautiously optimistic", "guardedly optimistic",
    "headwinds", "uncertainties remain",
    "challenging environment", "difficult environment",
    "not in a position to", "premature to",
]

# ==============================================================================
# GUIDANCE SPECIFICITY MARKERS
# ==============================================================================
# Concrete guidance language (numbers, ranges, specific metrics)

SPECIFIC_GUIDANCE_PATTERNS = [
    r'\$[\d,.]+\s*(billion|million|thousand|[bmk])',        # Dollar amounts
    r'\d+\.?\d*\s*%',                                        # Percentages
    r'(revenue|earnings|eps|margin|growth)\s+(of|at|to)\s+\$?[\d,.]+',
    r'(guidance|outlook|forecast|target)\s+(of|at|is|remains)\s+\$?[\d,.]+',
    r'(expect|guide|project|forecast)\s+.{0,30}\$?[\d,.]+', # Expect + number
    r'(range|between)\s+\$?[\d,.]+\s+(and|to)\s+\$?[\d,.]+', # Range guidance
    r'(raise|raised|raising|increase|increasing)\s+(our\s+)?(guidance|outlook|forecast)',
    r'(reaffirm|reiterate|maintain)\s+(our\s+)?(guidance|outlook|forecast)',
    r'basis\s+points?',                                      # Basis points
    r'(year[\s-]over[\s-]year|yoy|sequential)\s+\w+\s+of\s+\d+', # YoY/sequential + number
]

VAGUE_GUIDANCE_PATTERNS = [
    r'(expect|anticipate)\s+(to\s+)?(see|continue|remain)\s+(some|modest|moderate)',
    r'(roughly|approximately|around|about)\s+(in\s+line|similar|comparable)',
    r'(we\'ll|we will)\s+(provide|give)\s+(more\s+)?(detail|color|update)\s+(later|next|on)',
    r'(not\s+)?(ready|prepared|able)\s+to\s+(give|provide|share)\s+(specific|detailed)',
    r'(directionally|broadly|generally)\s+(consistent|similar|in\s+line)',
    r'too\s+early\s+to\s+(give|provide|quantify|comment)',
]


# ==============================================================================
# SPEAKER CLASSIFICATION PATTERNS
# ==============================================================================

# Patterns that identify analyst speakers (questions)
ANALYST_PATTERNS = [
    r"analyst",
    r"from\s+(goldman|morgan|jpmorgan|bank\s+of\s+america|barclays|citi|ubs|"
    r"deutsche|credit\s+suisse|rbc|wells\s+fargo|bernstein|cowen|"
    r"piper|wedbush|needham|oppenheimer|raymond\s+james|"
    r"jefferies|keybanc|btig|stifel|wolfe|loop|evercore|mizuho|"
    r"truist|canaccord|baird|susquehanna)",
]

# Patterns that identify management speakers (answers)
MANAGEMENT_PATTERNS = [
    r"\b(ceo|cfo|coo|cto|cmo|president|chairman|chief\s+\w+\s+officer)\b",
    r"\b(vice\s+president|vp|svp|evp)\b",
    r"\b(director|head\s+of|general\s+manager)\b",
]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TranscriptSection:
    """Parsed section of an earnings call transcript."""
    speaker: str
    text: str
    is_management: bool
    is_analyst: bool
    sentence_count: int = 0


@dataclass
class QualitativeMetrics:
    """Intermediate metrics from transcript analysis."""
    # FinBERT sentiment scores
    management_sentiment: float = 0.0      # -1 to +1
    analyst_sentiment: float = 0.0         # -1 to +1
    overall_sentiment: float = 0.0         # -1 to +1

    # Tone gap
    tone_gap: float = 0.0                  # management - analyst (positive = mgmt more optimistic)

    # Hedging
    hedging_ratio: float = 0.0             # hedging words / total management words
    strong_hedging_count: int = 0          # count of strong hedging phrases

    # Guidance specificity
    specific_guidance_count: int = 0       # concrete number mentions
    vague_guidance_count: int = 0          # vague language mentions
    guidance_specificity: float = 0.5      # 0 (vague) to 1 (specific)

    # Metadata
    management_sentence_count: int = 0
    analyst_sentence_count: int = 0
    total_word_count: int = 0
    call_date: Optional[str] = None
    quarter: int = 0
    year: int = 0


@dataclass
class EarningsCallQualSignal:
    """Output signal from qualitative earnings call analysis."""
    ticker: str
    analysis_date: datetime

    # Sub-signals (-1 to +1 each)
    management_tone: float = 0.0           # FinBERT sentiment of mgmt answers
    tone_gap_score: float = 0.0            # Mgmt more optimistic than analysts = bullish
    hedging_score: float = 0.0             # High hedging = bearish (negative)
    guidance_specificity_score: float = 0.0 # Specific guidance = bullish
    tone_change_score: float = 0.0         # QoQ improvement = bullish
    peer_tone_delta_score: float = 0.0     # More positive than sector peers = bullish

    # Composite
    composite_score: float = 0.0           # Weighted combination, -1 to +1
    signal: str = "neutral"                # strong_buy/buy/neutral/sell/strong_sell
    signal_strength: float = 0.0           # Absolute magnitude
    confidence: float = 0.0                # 0-1, based on data quality

    # Metadata
    call_date: Optional[str] = None
    days_since_call: int = 0
    signal_decay: float = 1.0              # Freshness decay factor
    is_actionable: bool = False

    # Detailed metrics for rationale
    metrics: Optional[QualitativeMetrics] = None
    prev_metrics: Optional[QualitativeMetrics] = None  # Previous quarter for QoQ

    # Summary
    summary: str = ""
    rationale: str = ""


# ==============================================================================
# MODEL
# ==============================================================================

class EarningsCallQualModel:
    """
    Qualitative Earnings Call Analysis Model.

    Analyzes the actual TEXT of earnings call transcripts to extract
    management tone, hedging, guidance quality, and sequential changes.
    This is complementary to:
    - PEAD (quantitative EPS/revenue surprise)
    - Filing Tone (annual 10-K filings)
    - Early Adopter (technology keyword detection)
    """

    def __init__(self, use_finbert: bool = True, transcript_vectorstore=None):
        """
        Initialize the model.

        Args:
            use_finbert: If True, use FinBERT for sentence-level sentiment.
                        If False, use Loughran-McDonald dictionary only (faster, no GPU).
            transcript_vectorstore: Optional TranscriptVectorStore for RAG-based
                                   peer tone comparison. If None, peer_tone_delta
                                   sub-signal is skipped (weight redistributed).
        """
        self.use_finbert = use_finbert
        self._finbert = None
        self._transcript_client = None
        self._transcript_vectorstore = transcript_vectorstore

        # Compile regex patterns
        self._analyst_re = [re.compile(p, re.IGNORECASE) for p in ANALYST_PATTERNS]
        self._management_re = [re.compile(p, re.IGNORECASE) for p in MANAGEMENT_PATTERNS]
        self._specific_guidance_re = [re.compile(p, re.IGNORECASE) for p in SPECIFIC_GUIDANCE_PATTERNS]
        self._vague_guidance_re = [re.compile(p, re.IGNORECASE) for p in VAGUE_GUIDANCE_PATTERNS]
        self._strong_hedging_re = [re.compile(re.escape(p), re.IGNORECASE) for p in STRONG_HEDGING_PHRASES]

        has_rag = transcript_vectorstore is not None
        logger.info("EarningsCallQualModel initialized (finbert=%s, rag=%s)", use_finbert, has_rag)

    @property
    def transcript_client(self):
        """Lazy-load transcript client."""
        if self._transcript_client is None:
            from auto_researcher.models.earnings_tech_signal import DefeatBetaTranscriptClient
            self._transcript_client = DefeatBetaTranscriptClient()
        return self._transcript_client

    @property
    def finbert(self):
        """Lazy-load FinBERT analyzer."""
        if self._finbert is None and self.use_finbert:
            try:
                from auto_researcher.agents.finbert_sentiment import FinBERTAnalyzer
                self._finbert = FinBERTAnalyzer(model_name="finbert")
                logger.info("FinBERT loaded for earnings call analysis")
            except Exception as e:
                logger.warning("FinBERT not available, falling back to dictionary: %s", e)
                self.use_finbert = False
        return self._finbert

    def unload(self):
        """Free FinBERT memory."""
        if self._finbert is not None:
            self._finbert.unload()
            self._finbert = None
        if self._transcript_client is not None:
            self._transcript_client.clear_cache()
            self._transcript_client = None
        self._transcript_vectorstore = None
        import gc
        gc.collect()
        logger.info("EarningsCallQualModel resources freed")

    # ------------------------------------------------------------------
    # TEXT PARSING
    # ------------------------------------------------------------------

    def _classify_speaker(self, speaker_line: str) -> Tuple[bool, bool]:
        """
        Classify a speaker line as management or analyst.

        Returns:
            (is_management, is_analyst)
        """
        speaker_lower = speaker_line.lower()

        # Check management patterns
        for pat in self._management_re:
            if pat.search(speaker_lower):
                return True, False

        # Check analyst patterns
        for pat in self._analyst_re:
            if pat.search(speaker_lower):
                return False, True

        # Heuristic: "Operator" is neither
        if "operator" in speaker_lower:
            return False, False

        # Default: if it looks like a question, likely analyst
        return False, False

    def _parse_transcript_sections(self, content: str) -> List[TranscriptSection]:
        """
        Parse a transcript into speaker-attributed sections.

        The transcript format is "Speaker: text\nSpeaker: text\n..."
        """
        sections = []
        current_speaker = ""
        current_text_parts = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if this is a new speaker line (format: "Name: text")
            colon_idx = line.find(":")
            if colon_idx > 0 and colon_idx < 80:
                potential_speaker = line[:colon_idx].strip()
                remaining_text = line[colon_idx + 1:].strip()

                # Heuristic: speaker names are typically 2-6 words
                word_count = len(potential_speaker.split())
                if 1 <= word_count <= 8 and not potential_speaker[0].isdigit():
                    # Save previous section
                    if current_speaker and current_text_parts:
                        text = " ".join(current_text_parts)
                        is_mgmt, is_analyst = self._classify_speaker(current_speaker)
                        sections.append(TranscriptSection(
                            speaker=current_speaker,
                            text=text,
                            is_management=is_mgmt,
                            is_analyst=is_analyst,
                            sentence_count=len(self._split_sentences(text)),
                        ))

                    current_speaker = potential_speaker
                    current_text_parts = [remaining_text] if remaining_text else []
                    continue

            # Continuation of current speaker
            current_text_parts.append(line)

        # Don't forget the last section
        if current_speaker and current_text_parts:
            text = " ".join(current_text_parts)
            is_mgmt, is_analyst = self._classify_speaker(current_speaker)
            sections.append(TranscriptSection(
                speaker=current_speaker,
                text=text,
                is_management=is_mgmt,
                is_analyst=is_analyst,
                sentence_count=len(self._split_sentences(text)),
            ))

        return sections

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (avoid nltk dependency)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _find_qa_boundary(self, sections: List[TranscriptSection]) -> int:
        """
        Find the index where Q&A section begins.

        Typically the Operator says something like "we will now begin the Q&A..."
        """
        for i, sec in enumerate(sections):
            text_lower = sec.text.lower()
            if any(marker in text_lower for marker in [
                "question-and-answer",
                "question and answer",
                "q&a session",
                "q&a portion",
                "open the line",
                "open it up for questions",
                "open the floor",
                "first question",
                "take questions",
                "begin the q&a",
                "operator instructions",
            ]):
                return i

        # Fallback: assume Q&A starts at ~40% through the call
        return max(1, len(sections) * 2 // 5)

    # ------------------------------------------------------------------
    # SENTIMENT ANALYSIS
    # ------------------------------------------------------------------

    def _analyze_sentiment_finbert(self, sentences: List[str]) -> float:
        """
        Compute average FinBERT sentiment across sentences.

        Returns:
            Float in [-1, +1]
        """
        if not sentences or self.finbert is None:
            return 0.0

        # Cap sentences to avoid OOM
        cap = QUAL_CONFIG["max_sentences_per_section"]
        if len(sentences) > cap:
            sentences = sentences[:cap]

        try:
            results = self.finbert.analyze_batch(sentences)
            if not results:
                return 0.0
            scores = [r.sentiment_score for r in results]
            return sum(scores) / len(scores)
        except Exception as e:
            logger.warning("FinBERT batch analysis failed: %s", e)
            return 0.0

    def _analyze_sentiment_dictionary(self, text: str) -> float:
        """
        Compute sentiment using Loughran-McDonald positive/negative word counts.

        Fallback when FinBERT is not available.

        Returns:
            Float in [-1, +1]
        """
        try:
            from auto_researcher.models.filing_tone import LM_NEGATIVE, LM_POSITIVE
        except ImportError:
            return 0.0

        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return 0.0

        pos_count = sum(1 for w in words if w in LM_POSITIVE)
        neg_count = sum(1 for w in words if w in LM_NEGATIVE)
        total = pos_count + neg_count
        if total == 0:
            return 0.0

        # Net tone: (pos - neg) / (pos + neg), naturally in [-1, +1]
        return (pos_count - neg_count) / total

    def _compute_sentiment(self, sentences: List[str], full_text: str) -> float:
        """Compute sentiment using FinBERT or dictionary fallback."""
        if self.use_finbert and self.finbert is not None:
            return self._analyze_sentiment_finbert(sentences)
        return self._analyze_sentiment_dictionary(full_text)

    # ------------------------------------------------------------------
    # HEDGING ANALYSIS
    # ------------------------------------------------------------------

    def _compute_hedging_metrics(self, management_text: str) -> Tuple[float, int]:
        """
        Compute hedging ratio and strong hedging phrase count.

        Returns:
            (hedging_ratio, strong_hedging_count)
        """
        words = re.findall(r'\b[a-z]+\b', management_text.lower())
        if not words:
            return 0.0, 0

        hedging_count = sum(1 for w in words if w in HEDGING_WORDS)
        hedging_ratio = hedging_count / len(words)

        # Count strong hedging phrases
        text_lower = management_text.lower()
        strong_count = sum(1 for pat in self._strong_hedging_re if pat.search(text_lower))

        return hedging_ratio, strong_count

    # ------------------------------------------------------------------
    # GUIDANCE SPECIFICITY
    # ------------------------------------------------------------------

    def _compute_guidance_specificity(self, management_text: str) -> Tuple[int, int, float]:
        """
        Measure how specific vs vague management's forward guidance is.

        Returns:
            (specific_count, vague_count, specificity_score)
            specificity_score: 0.0 (all vague) to 1.0 (all specific)
        """
        specific_count = sum(
            len(pat.findall(management_text))
            for pat in self._specific_guidance_re
        )
        vague_count = sum(
            len(pat.findall(management_text))
            for pat in self._vague_guidance_re
        )

        total = specific_count + vague_count
        if total == 0:
            return 0, 0, 0.5  # Neutral if no guidance language detected

        specificity = specific_count / total
        return specific_count, vague_count, specificity

    # ------------------------------------------------------------------
    # PEER TONE COMPARISON (RAG-powered)
    # ------------------------------------------------------------------

    def _compute_peer_tone_delta(
        self,
        ticker: str,
        management_tone: float,
        sector: str = "",
    ) -> Tuple[float, int]:
        """
        Compare management tone to sector peers using transcript vectorstore.

        Retrieves management Q&A passages for the sector and computes
        sentiment across peers, then returns delta (positive = more bullish
        than peers).

        Args:
            ticker: The primary ticker being analyzed.
            management_tone: The primary ticker's management tone score.
            sector: Sector name for finding peers (uses vectorstore metadata).

        Returns:
            (peer_tone_delta, n_peers_found)
            peer_tone_delta: in [-1, +1], positive = more bullish than peers
            n_peers_found: how many peers were found for comparison
        """
        if self._transcript_vectorstore is None:
            return 0.0, 0

        try:
            # Use thematic query to find similar-context management passages
            # from OTHER companies discussing the same topics
            query = "management outlook guidance earnings next quarter"
            results = self._transcript_vectorstore.query_by_theme(
                theme=query,
                management_only=True,
                qa_only=True,
                n_results=30,
            )

            if not results["documents"]:
                return 0.0, 0

            # Group passages by ticker, excluding the primary ticker
            peer_texts: Dict[str, List[str]] = {}
            for doc, meta in zip(results["documents"], results["metadatas"]):
                peer_ticker = meta.get("ticker", "")
                if peer_ticker == ticker.upper():
                    continue
                if peer_ticker not in peer_texts:
                    peer_texts[peer_ticker] = []
                peer_texts[peer_ticker].append(doc)

            n_peers = len(peer_texts)
            if n_peers < QUAL_CONFIG["min_peers_for_signal"]:
                return 0.0, n_peers

            # Compute average sentiment across peer management passages
            peer_sentiments = []
            for peer_ticker, texts in peer_texts.items():
                combined_text = " ".join(texts)
                sentences = self._split_sentences(combined_text)
                if sentences:
                    peer_sent = self._compute_sentiment(sentences[:20], combined_text[:5000])
                    peer_sentiments.append(peer_sent)

            if not peer_sentiments:
                return 0.0, 0

            avg_peer_tone = sum(peer_sentiments) / len(peer_sentiments)
            # Delta: how much more positive is this company vs peers?
            # Scale: 0.1 delta -> ~0.3 score (similar to QoQ change scaling)
            raw_delta = management_tone - avg_peer_tone
            peer_tone_delta = max(-1.0, min(1.0, raw_delta * 3.0))

            logger.debug(
                "%s peer tone: mgmt=%.3f, avg_peers=%.3f (n=%d), delta=%.3f",
                ticker, management_tone, avg_peer_tone, n_peers, peer_tone_delta,
            )
            return peer_tone_delta, n_peers

        except Exception as e:
            logger.warning("Peer tone comparison failed for %s: %s", ticker, e)
            return 0.0, 0

    # ------------------------------------------------------------------
    # CORE ANALYSIS
    # ------------------------------------------------------------------

    def _analyze_transcript(self, transcript: Dict) -> Optional[QualitativeMetrics]:
        """
        Analyze a single transcript and return qualitative metrics.
        """
        content = transcript.get("content", "")
        if not content or len(content) < 500:
            logger.debug("Transcript too short to analyze")
            return None

        # Parse into speaker-attributed sections
        sections = self._parse_transcript_sections(content)
        if len(sections) < 3:
            logger.debug("Too few sections in transcript")
            return None

        # Find Q&A boundary
        qa_start = self._find_qa_boundary(sections)
        qa_sections = sections[qa_start:]

        # Separate management vs analyst text in Q&A
        mgmt_sections = [s for s in qa_sections if s.is_management]
        analyst_sections = [s for s in qa_sections if s.is_analyst]

        # Combine text
        mgmt_text = " ".join(s.text for s in mgmt_sections)
        analyst_text = " ".join(s.text for s in analyst_sections)
        all_mgmt_text = " ".join(s.text for s in sections if s.is_management)

        # Split into sentences
        mgmt_sentences = self._split_sentences(mgmt_text)
        analyst_sentences = self._split_sentences(analyst_text)

        mgmt_sentence_count = len(mgmt_sentences)
        analyst_sentence_count = len(analyst_sentences)

        if mgmt_sentence_count < QUAL_CONFIG["min_qa_sentences"]:
            # If Q&A too short, use entire transcript management text
            mgmt_text = all_mgmt_text
            mgmt_sentences = self._split_sentences(mgmt_text)
            mgmt_sentence_count = len(mgmt_sentences)

        if mgmt_sentence_count < 5:
            logger.debug("Not enough management text to analyze")
            return None

        # 1. Sentiment analysis
        mgmt_sentiment = self._compute_sentiment(mgmt_sentences, mgmt_text)
        analyst_sentiment = self._compute_sentiment(analyst_sentences, analyst_text) if analyst_sentences else 0.0
        overall_sentiment = self._compute_sentiment(
            mgmt_sentences[:50] + analyst_sentences[:50],
            mgmt_text + " " + analyst_text,
        )

        # 2. Tone gap (management - analyst)
        tone_gap = mgmt_sentiment - analyst_sentiment

        # 3. Hedging analysis (on all management text, not just Q&A)
        hedging_ratio, strong_hedging_count = self._compute_hedging_metrics(all_mgmt_text)

        # 4. Guidance specificity (on all management text)
        specific_count, vague_count, guidance_specificity = self._compute_guidance_specificity(all_mgmt_text)

        total_words = len(re.findall(r'\b\w+\b', content))

        return QualitativeMetrics(
            management_sentiment=mgmt_sentiment,
            analyst_sentiment=analyst_sentiment,
            overall_sentiment=overall_sentiment,
            tone_gap=tone_gap,
            hedging_ratio=hedging_ratio,
            strong_hedging_count=strong_hedging_count,
            specific_guidance_count=specific_count,
            vague_guidance_count=vague_count,
            guidance_specificity=guidance_specificity,
            management_sentence_count=mgmt_sentence_count,
            analyst_sentence_count=analyst_sentence_count,
            total_word_count=total_words,
            call_date=transcript.get("date"),
            quarter=transcript.get("quarter", 0),
            year=transcript.get("year", 0),
        )

    def _compute_signal(
        self,
        ticker: str,
        current: QualitativeMetrics,
        previous: Optional[QualitativeMetrics] = None,
    ) -> EarningsCallQualSignal:
        """
        Compute the qualitative signal from transcript metrics.
        """
        cfg = QUAL_CONFIG

        # --- Sub-signal 1: Management Tone ---
        # FinBERT sentiment is in [-1, +1], use directly
        management_tone = current.management_sentiment

        # --- Sub-signal 2: Tone Gap ---
        # Positive gap = management more optimistic than analysts = bullish
        # Cap at ±0.5 to avoid extreme values
        tone_gap_raw = current.tone_gap
        tone_gap_score = max(-1.0, min(1.0, tone_gap_raw * 2.0))

        # --- Sub-signal 3: Hedging Score ---
        # Higher hedging = more bearish (invert)
        # Typical hedging ratio: 3-8% of words
        # Baseline ~5%, below = confident, above = hedging
        hedging_baseline = 0.05
        hedging_deviation = current.hedging_ratio - hedging_baseline
        # Scale: 2% above baseline -> -0.5 score
        hedging_score = -hedging_deviation * 25  # 0.04 deviation -> ±1.0
        # Penalty for strong hedging phrases
        if current.strong_hedging_count >= 3:
            hedging_score -= 0.2
        hedging_score = max(-1.0, min(1.0, hedging_score))

        # --- Sub-signal 4: Guidance Specificity ---
        # Map 0-1 specificity to -1 to +1 (0.5 = neutral)
        guidance_specificity_score = (current.guidance_specificity - 0.5) * 2.0
        guidance_specificity_score = max(-1.0, min(1.0, guidance_specificity_score))

        # --- Sub-signal 5: QoQ Tone Change ---
        tone_change_score = 0.0
        if previous is not None:
            tone_delta = current.management_sentiment - previous.management_sentiment
            # Scale: 0.1 improvement -> ~0.3 score
            tone_change_score = max(-1.0, min(1.0, tone_delta * 3.0))

            # Also factor in hedging change
            hedging_delta = current.hedging_ratio - previous.hedging_ratio
            # Decreasing hedging is bullish
            hedging_change_signal = -hedging_delta * 10
            tone_change_score = 0.7 * tone_change_score + 0.3 * max(-1.0, min(1.0, hedging_change_signal))

        # --- Sub-signal 6: Peer Tone Delta (RAG-powered) ---
        peer_tone_delta_score = 0.0
        n_peers_found = 0
        if self._transcript_vectorstore is not None:
            peer_tone_delta_score, n_peers_found = self._compute_peer_tone_delta(
                ticker, management_tone,
            )

        # --- Composite Score ---
        # If vectorstore not available, redistribute peer_tone_delta weight
        # to the other sub-signals proportionally
        peer_weight = cfg["peer_tone_delta_weight"]
        if self._transcript_vectorstore is None or n_peers_found < cfg["min_peers_for_signal"]:
            # No peer data — redistribute weight to other sub-signals
            peer_weight_actual = 0.0
            redistrib = cfg["peer_tone_delta_weight"] / (1.0 - cfg["peer_tone_delta_weight"])
            w_tone = cfg["management_tone_weight"] * (1.0 + redistrib)
            w_gap = cfg["tone_gap_weight"] * (1.0 + redistrib)
            w_hedge = cfg["hedging_weight"] * (1.0 + redistrib)
            w_guide = cfg["guidance_specificity_weight"] * (1.0 + redistrib)
            w_change = cfg["tone_change_weight"] * (1.0 + redistrib)
        else:
            peer_weight_actual = peer_weight
            w_tone = cfg["management_tone_weight"]
            w_gap = cfg["tone_gap_weight"]
            w_hedge = cfg["hedging_weight"]
            w_guide = cfg["guidance_specificity_weight"]
            w_change = cfg["tone_change_weight"]

        composite = (
            w_tone * management_tone
            + w_gap * tone_gap_score
            + w_hedge * hedging_score
            + w_guide * guidance_specificity_score
            + w_change * tone_change_score
            + peer_weight_actual * peer_tone_delta_score
        )
        composite = max(-1.0, min(1.0, composite))

        # --- Signal decay based on call age ---
        days_since = 0
        signal_decay = 1.0
        if current.call_date:
            try:
                call_dt = datetime.strptime(str(current.call_date)[:10], "%Y-%m-%d")
                days_since = (datetime.now() - call_dt).days
                half_life = cfg["decay_half_life_days"]
                signal_decay = 0.5 ** (max(0, days_since - 14) / half_life)  # No decay for first 2 weeks
            except (ValueError, TypeError):
                pass

        decayed_composite = composite * signal_decay

        # --- Signal interpretation ---
        abs_score = abs(decayed_composite)
        if abs_score >= cfg["strong_threshold"]:
            signal = "strong_buy" if decayed_composite > 0 else "strong_sell"
            strength = abs_score
        elif abs_score >= cfg["weak_threshold"]:
            signal = "buy" if decayed_composite > 0 else "sell"
            strength = abs_score
        else:
            signal = "neutral"
            strength = abs_score

        is_actionable = (
            abs_score >= cfg["weak_threshold"]
            and days_since <= cfg["max_call_age_days"]
            and current.management_sentence_count >= QUAL_CONFIG["min_qa_sentences"]
        )

        # Confidence: based on data quality
        confidence = min(1.0, current.management_sentence_count / 50)  # More text = more confident
        if previous is not None:
            confidence = min(1.0, confidence + 0.2)  # QoQ comparison boosts confidence
        confidence *= signal_decay  # Stale signals are less confident

        # --- Summary ---
        summary_parts = []
        if management_tone > 0.1:
            summary_parts.append(f"positive mgmt tone ({management_tone:+.2f})")
        elif management_tone < -0.1:
            summary_parts.append(f"negative mgmt tone ({management_tone:+.2f})")

        if tone_gap_score > 0.2:
            summary_parts.append("mgmt more optimistic than analysts")
        elif tone_gap_score < -0.2:
            summary_parts.append("analysts more positive than mgmt")

        if hedging_score < -0.3:
            summary_parts.append(f"high hedging ({current.hedging_ratio:.1%})")
        elif hedging_score > 0.3:
            summary_parts.append(f"confident language ({current.hedging_ratio:.1%} hedging)")

        if guidance_specificity_score > 0.3:
            summary_parts.append(f"specific guidance ({current.specific_guidance_count} concrete refs)")
        elif guidance_specificity_score < -0.3:
            summary_parts.append(f"vague guidance ({current.vague_guidance_count} vague refs)")

        if tone_change_score > 0.2:
            summary_parts.append("improving tone QoQ")
        elif tone_change_score < -0.2:
            summary_parts.append("deteriorating tone QoQ")

        if peer_tone_delta_score > 0.2:
            summary_parts.append(f"more bullish than peers ({n_peers_found} peers)")
        elif peer_tone_delta_score < -0.2:
            summary_parts.append(f"less bullish than peers ({n_peers_found} peers)")

        summary = "; ".join(summary_parts) if summary_parts else "neutral call tone"

        rationale = (
            f"Q{current.quarter} {current.year}: "
            f"mgmt_tone={management_tone:+.2f}, gap={tone_gap_score:+.2f}, "
            f"hedging={hedging_score:+.2f}, guidance={guidance_specificity_score:+.2f}, "
            f"change={tone_change_score:+.2f}, peer_delta={peer_tone_delta_score:+.2f} "
            f"-> composite={decayed_composite:+.3f}"
        )

        return EarningsCallQualSignal(
            ticker=ticker,
            analysis_date=datetime.now(),
            management_tone=management_tone,
            tone_gap_score=tone_gap_score,
            hedging_score=hedging_score,
            guidance_specificity_score=guidance_specificity_score,
            tone_change_score=tone_change_score,
            peer_tone_delta_score=peer_tone_delta_score,
            composite_score=decayed_composite,
            signal=signal,
            signal_strength=strength,
            confidence=confidence,
            call_date=current.call_date,
            days_since_call=days_since,
            signal_decay=signal_decay,
            is_actionable=is_actionable,
            metrics=current,
            prev_metrics=previous,
            summary=summary,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_signal(self, ticker: str) -> Optional[EarningsCallQualSignal]:
        """
        Analyze the most recent earnings call(s) for a ticker.

        Fetches the latest 2 transcripts (current + previous quarter for QoQ),
        analyzes qualitative signals, and returns a composite score.

        Args:
            ticker: Stock ticker to analyze.

        Returns:
            EarningsCallQualSignal or None if no transcripts available.
        """
        ticker = ticker.upper()

        try:
            # Fetch latest 2 transcripts
            transcripts = self.transcript_client.get_transcripts(ticker, limit=4)

            if not transcripts:
                logger.info("No transcripts available for %s", ticker)
                return None

            # Analyze current (most recent) transcript
            current_metrics = self._analyze_transcript(transcripts[0])
            if current_metrics is None:
                logger.info("Failed to analyze current transcript for %s", ticker)
                return None

            # Analyze previous quarter for QoQ comparison
            prev_metrics = None
            if len(transcripts) >= 2:
                prev_metrics = self._analyze_transcript(transcripts[1])
                if prev_metrics is not None:
                    logger.debug("QoQ comparison available for %s: Q%d %d vs Q%d %d",
                                ticker, current_metrics.quarter, current_metrics.year,
                                prev_metrics.quarter, prev_metrics.year)

            # Compute signal
            signal = self._compute_signal(ticker, current_metrics, prev_metrics)
            logger.info(
                "%s earnings call qual: composite=%+.3f signal=%s (decay=%.2f, conf=%.2f)",
                ticker, signal.composite_score, signal.signal,
                signal.signal_decay, signal.confidence,
            )
            return signal

        except Exception as e:
            logger.error("EarningsCallQualModel failed for %s: %s", ticker, e)
            return None
