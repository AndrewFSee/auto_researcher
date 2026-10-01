"""
Transcript Vector Store.

ChromaDB-backed vector store for earnings call transcripts, enabling RAG
retrieval for qualitative earnings analysis. Transcripts are chunked by
speaker turn, embedded with sentence-transformers, and stored with rich
metadata (ticker, quarter, year, speaker, is_management, is_qa_section)
for efficient filtered retrieval.

Use Cases:
    1. Cross-company thematic search ("Which companies discuss tariff headwinds?")
    2. Peer tone comparison (compare management tone across sector peers)
    3. Historical context retrieval ("When did AAPL last discuss margin compression?")
    4. Thematic trend detection across the entire transcript corpus

Usage:
    from auto_researcher.data.transcript_vectorstore import TranscriptVectorStore

    store = TranscriptVectorStore()
    store.build_from_parquet()  # One-time indexing (takes ~20-30 min for 223K transcripts)

    # Query for a specific ticker
    results = store.query("AAPL", query_text="guidance for next quarter")

    # Cross-ticker thematic search
    results = store.query_by_theme("AI capital expenditure", n_results=20)

    # Peer comparison
    results = store.query_peer_comparison("AAPL", ["MSFT", "GOOG", "AMZN"],
                                          query_text="margin expansion")

    # Historical context for a ticker
    results = store.query_ticker_history("TSLA", query_text="production ramp",
                                          n_results=10)
"""

import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Paths
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "transcript_chroma"

# Embedding model - same as news vectorstore for consistency
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Batch sizes
EMBED_BATCH_SIZE = 512  # Sentence-transformers handles large batches well
CHROMA_ADD_BATCH = 5000  # ChromaDB upsert batch limit

# Chunking config
MAX_CHUNK_CHARS = 1500  # Max characters per chunk (~300 words)
MIN_CHUNK_CHARS = 100   # Skip very short speaker turns
MERGE_SHORT_TURNS = True  # Merge consecutive same-speaker turns

# Speaker classification patterns (subset from earnings_call_qual.py)
_ANALYST_PATTERNS = [
    re.compile(r"analyst", re.IGNORECASE),
    re.compile(
        r"from\s+(goldman|morgan|jpmorgan|bank\s+of\s+america|barclays|citi|ubs|"
        r"deutsche|credit\s+suisse|rbc|wells\s+fargo|bernstein|cowen|"
        r"piper|wedbush|needham|oppenheimer|raymond\s+james|"
        r"jefferies|keybanc|btig|stifel|wolfe|loop|evercore|mizuho|"
        r"truist|canaccord|baird|susquehanna)",
        re.IGNORECASE,
    ),
]

_MANAGEMENT_PATTERNS = [
    re.compile(r"\b(ceo|cfo|coo|cto|cmo|president|chairman|chief\s+\w+\s+officer)\b", re.IGNORECASE),
    re.compile(r"\b(vice\s+president|vp|svp|evp)\b", re.IGNORECASE),
    re.compile(r"\b(director|head\s+of|general\s+manager)\b", re.IGNORECASE),
]

_QA_MARKERS = [
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
]


def _classify_speaker(speaker: str, management_names: Optional[set] = None) -> str:
    """Classify speaker as 'management', 'analyst', 'operator', or 'unknown'.

    Args:
        speaker: Speaker label (may be a name or a titled description).
        management_names: Pre-identified management speaker names (lowercase).
                          When provided, overrides pattern-based classification.
    """
    s = speaker.lower().strip()
    if "operator" in s:
        return "operator"
    # Check if name was pre-identified as management
    if management_names and s in management_names:
        return "management"
    for pat in _MANAGEMENT_PATTERNS:
        if pat.search(s):
            return "management"
    for pat in _ANALYST_PATTERNS:
        if pat.search(s):
            return "analyst"
    return "unknown"


def _is_qa_section(text: str) -> bool:
    """Check if text contains Q&A section markers."""
    lower = text.lower()
    return any(m in lower for m in _QA_MARKERS)


def _identify_management_speakers(
    turns: List[Tuple[str, str]],
) -> set:
    """Pre-identify management speakers from prepared remarks.

    Heuristic: speakers who give substantial remarks BEFORE the Q&A
    transition (excluding Operator) are management.  Analysts only
    appear during Q&A.

    Returns:
        Set of lowercase speaker names classified as management.
    """
    management: set = set()
    qa_started = False
    analyst_candidates: set = set()

    for speaker, text in turns:
        s_lower = speaker.lower().strip()
        if "operator" in s_lower:
            # Check if operator introduces an analyst ("from Goldman")
            if qa_started:
                for pat in _ANALYST_PATTERNS:
                    m = pat.search(text)
                    if m:
                        # Next non-operator speaker is likely the analyst
                        pass
            if _is_qa_section(text):
                qa_started = True
            continue

        if _is_qa_section(text):
            qa_started = True

        if not qa_started:
            # Speakers with substantial prepared remarks are management
            # (skip tiny IR coordinator transitions)
            if len(text) > 200:
                management.add(s_lower)
            elif len(text) > 50:
                # Short turns in prepared remarks could be IR coordinator
                # Still management (they work for the company)
                management.add(s_lower)
        else:
            # In Q&A section:
            # - Already known management speakers remain management
            # - New speakers giving short turns are likely analysts
            if s_lower in management:
                continue
            # Check analyst patterns in operator's preceding intro
            for pat in _ANALYST_PATTERNS:
                if pat.search(speaker):
                    analyst_candidates.add(s_lower)
                    break
            else:
                # Unknown in Q&A — if they ask short questions, likely analyst
                if len(text) < 500:
                    analyst_candidates.add(s_lower)
                else:
                    # Long answer in Q&A — might be management not seen before
                    management.add(s_lower)

    return management


def _parse_and_chunk_transcript(
    content: str,
    ticker: str,
    quarter: int,
    year: int,
    call_date: str,
) -> List[Dict]:
    """
    Parse a transcript into speaker-attributed chunks suitable for embedding.

    Chunking strategy:
    - Primary unit: speaker turn (one speaker's contiguous text)
    - Long turns are split at sentence boundaries to stay under MAX_CHUNK_CHARS
    - Very short turns are merged with adjacent same-speaker turns
    - Each chunk carries full metadata for filtered retrieval

    Returns:
        List of dicts with keys: text, metadata
    """
    chunks = []
    in_qa = False

    # Parse speaker turns
    current_speaker = ""
    current_text_parts = []
    turns: List[Tuple[str, str]] = []  # (speaker, text)

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        colon_idx = line.find(":")
        if colon_idx > 0 and colon_idx < 80:
            potential_speaker = line[:colon_idx].strip()
            remaining = line[colon_idx + 1:].strip()
            word_count = len(potential_speaker.split())
            if 1 <= word_count <= 8 and not potential_speaker[0].isdigit():
                # Save previous
                if current_speaker and current_text_parts:
                    turns.append((current_speaker, " ".join(current_text_parts)))
                current_speaker = potential_speaker
                current_text_parts = [remaining] if remaining else []
                continue

        current_text_parts.append(line)

    # Last turn
    if current_speaker and current_text_parts:
        turns.append((current_speaker, " ".join(current_text_parts)))

    if not turns:
        return chunks

    # Pre-identify management speakers from speaking patterns
    mgmt_names = _identify_management_speakers(turns)

    # Merge consecutive same-speaker short turns
    if MERGE_SHORT_TURNS:
        merged: List[Tuple[str, str]] = []
        for speaker, text in turns:
            if (
                merged
                and merged[-1][0] == speaker
                and len(merged[-1][1]) + len(text) < MAX_CHUNK_CHARS
            ):
                merged[-1] = (speaker, merged[-1][1] + " " + text)
            else:
                merged.append((speaker, text))
        turns = merged

    # Convert turns to chunks
    for speaker, text in turns:
        # Track Q&A boundary
        if _is_qa_section(text):
            in_qa = True

        if len(text) < MIN_CHUNK_CHARS:
            continue

        role = _classify_speaker(speaker, management_names=mgmt_names)

        # Split long turns at sentence boundaries
        if len(text) > MAX_CHUNK_CHARS:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk_parts = []
            current_len = 0

            for sent in sentences:
                if current_len + len(sent) > MAX_CHUNK_CHARS and current_chunk_parts:
                    chunk_text = " ".join(current_chunk_parts)
                    if len(chunk_text) >= MIN_CHUNK_CHARS:
                        chunks.append({
                            "text": f"{speaker}: {chunk_text}",
                            "metadata": {
                                "ticker": ticker.upper(),
                                "quarter": quarter,
                                "year": year,
                                "call_date": call_date,
                                "speaker": speaker[:100],
                                "role": role,
                                "is_management": role == "management",
                                "is_analyst": role == "analyst",
                                "is_qa": in_qa,
                                "char_count": len(chunk_text),
                            },
                        })
                    current_chunk_parts = [sent]
                    current_len = len(sent)
                else:
                    current_chunk_parts.append(sent)
                    current_len += len(sent)

            # Remaining
            if current_chunk_parts:
                chunk_text = " ".join(current_chunk_parts)
                if len(chunk_text) >= MIN_CHUNK_CHARS:
                    chunks.append({
                        "text": f"{speaker}: {chunk_text}",
                        "metadata": {
                            "ticker": ticker.upper(),
                            "quarter": quarter,
                            "year": year,
                            "call_date": call_date,
                            "speaker": speaker[:100],
                            "role": role,
                            "is_management": role == "management",
                            "is_analyst": role == "analyst",
                            "is_qa": in_qa,
                            "char_count": len(chunk_text),
                        },
                    })
        else:
            # Whole turn is one chunk
            chunks.append({
                "text": f"{speaker}: {text}",
                "metadata": {
                    "ticker": ticker.upper(),
                    "quarter": quarter,
                    "year": year,
                    "call_date": call_date,
                    "speaker": speaker[:100],
                    "role": role,
                    "is_management": role == "management",
                    "is_analyst": role == "analyst",
                    "is_qa": in_qa,
                    "char_count": len(text),
                },
            })

    return chunks


class TranscriptVectorStore:
    """
    ChromaDB vector store over earnings call transcript chunks.

    Supports:
    - Building the index from DefeatBeta parquet (one-time or incremental)
    - Querying by ticker with semantic search + metadata filters
    - Cross-company thematic search
    - Peer comparison queries
    - Historical context retrieval per ticker
    """

    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.chroma_path = Path(chroma_path) if chroma_path else DEFAULT_CHROMA_PATH
        self.embedding_model_name = embedding_model
        self._chroma_client = None
        self._collection = None

    def _get_chroma_client(self):
        """Lazy-init ChromaDB persistent client."""
        if self._chroma_client is None:
            import chromadb
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
            )
        return self._chroma_client

    def _get_collection(self):
        """Get or create the transcript chunks collection."""
        if self._collection is None:
            client = self._get_chroma_client()
            self._collection = client.get_or_create_collection(
                name="transcript_chunks",
                metadata={
                    "description": "Earnings call transcript chunks for RAG",
                    "embedding_model": self.embedding_model_name,
                    "hnsw:space": "cosine",
                },
            )
        return self._collection

    def get_index_count(self) -> int:
        """Number of chunks in the index."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def build_from_parquet(
        self,
        max_transcripts: Optional[int] = None,
        rebuild: bool = False,
        tickers: Optional[List[str]] = None,
        min_year: Optional[int] = None,
    ) -> int:
        """
        Build the vector store from the DefeatBeta transcript parquet.

        Memory-efficient: streams through the parquet in row-group batches,
        parsing → chunking → embedding → upserting each batch before moving
        to the next.  Peak memory ~200-400 MB.

        Args:
            max_transcripts: Limit total transcripts to index (for testing).
            rebuild: If True, drop and rebuild entire index.
            tickers: If provided, only index these tickers.
            min_year: Only index transcripts from this year onward
                      (e.g. 2023).  Drastically speeds up the build.

        Returns:
            Number of chunks indexed.
        """
        import gc

        if rebuild:
            try:
                client = self._get_chroma_client()
                client.delete_collection("transcript_chunks")
                self._collection = None
                logger.info("Dropped existing transcript collection for rebuild")
            except Exception:
                pass

        collection = self._get_collection()
        existing_count = collection.count()
        logger.info(f"Existing transcript chunks in index: {existing_count:,}")

        # ----------------------------------------------------------
        # Locate parquet file
        # ----------------------------------------------------------
        from auto_researcher.models.earnings_tech_signal import (
            DefeatBetaTranscriptClient,
            TRANSCRIPT_CACHE_PATH,
        )

        client = DefeatBetaTranscriptClient()
        if not client._ensure_downloaded():
            logger.error("Failed to download/find transcript parquet. Cannot build index.")
            return 0
        del client

        import pyarrow.parquet as pq
        import pandas as pd

        logger.info(f"Loading transcripts from {TRANSCRIPT_CACHE_PATH}")
        t0 = time.time()

        pf = pq.ParquetFile(TRANSCRIPT_CACHE_PATH)
        total_rows = pf.metadata.num_rows
        n_row_groups = pf.metadata.num_row_groups
        logger.info(f"Parquet has {total_rows:,} transcripts across "
                     f"{n_row_groups} row groups")

        # ----------------------------------------------------------
        # Pre-scan: for each row group, read only year+symbol columns
        # to decide which row groups to process (skip groups that have
        # no rows matching our year/ticker filters).
        # ----------------------------------------------------------
        tickers_upper = set(t.upper() for t in tickers) if tickers else None
        rg_candidates: List[int] = []

        if min_year or tickers_upper:
            logger.info("Pre-scanning row groups for year/ticker filtering...")
            for rg_idx in range(n_row_groups):
                scan_cols = ["fiscal_year"]
                if tickers_upper:
                    scan_cols.append("symbol")
                tbl = pf.read_row_group(rg_idx, columns=scan_cols)
                years_arr = tbl.column("fiscal_year").to_pylist()
                has_year = (not min_year) or any(
                    y is not None and int(y) >= min_year for y in years_arr
                )
                has_ticker = True
                if tickers_upper:
                    syms = tbl.column("symbol").to_pylist()
                    has_ticker = any(
                        s and str(s).upper() in tickers_upper for s in syms
                    )
                del tbl
                if has_year and has_ticker:
                    rg_candidates.append(rg_idx)
            logger.info(f"  {len(rg_candidates)}/{n_row_groups} row groups "
                        f"match filters (min_year={min_year}, "
                        f"tickers={len(tickers_upper) if tickers_upper else 'all'})")
        else:
            rg_candidates = list(range(n_row_groups))

        if not rg_candidates:
            logger.warning("No row groups match the given filters.")
            return 0

        # ----------------------------------------------------------
        # Load embedding model once (~80 MB)
        # ----------------------------------------------------------
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {self.embedding_model_name}...")
        embed_model = SentenceTransformer(self.embedding_model_name)

        # ----------------------------------------------------------
        # Track already-indexed transcript keys for dedup
        # ----------------------------------------------------------
        indexed_keys: set = set()
        if existing_count > 0 and not rebuild:
            logger.info("Scanning existing index for duplicate detection...")
            try:
                sample_size = min(existing_count, 50000)
                sample = collection.get(limit=sample_size, include=[])
                for doc_id in sample["ids"]:
                    parts = doc_id.split("_")
                    if len(parts) >= 4:
                        indexed_keys.add(f"{parts[1]}_{parts[2]}_{parts[3]}")
                logger.info(f"Found {len(indexed_keys):,} already-indexed transcripts to skip")
                del sample
            except Exception as e:
                logger.warning(f"Could not read existing IDs: {e}")

        # ----------------------------------------------------------
        # Stream through qualifying row groups
        # ----------------------------------------------------------
        total_indexed = 0
        total_skipped = 0
        total_transcripts_processed = 0
        total_qualifying = 0          # transcripts matching filters
        t_embed_cum = 0.0
        t_parse_cum = 0.0

        for rg_pos, rg_idx in enumerate(rg_candidates):
            # Read one row group at a time
            rg_table = pf.read_row_group(
                rg_idx,
                columns=["symbol", "report_date", "fiscal_quarter",
                         "fiscal_year", "transcripts"],
            )
            batch_df = rg_table.to_pandas()
            del rg_table

            # ---- Pre-filter DataFrame before iteration ----
            if min_year:
                batch_df = batch_df[batch_df["fiscal_year"].fillna(0).astype(int) >= min_year]
            if tickers_upper:
                batch_df = batch_df[
                    batch_df["symbol"].fillna("").str.upper().isin(tickers_upper)
                ]

            total_transcripts_processed += len(batch_df)

            if batch_df.empty:
                gc.collect()
                continue

            # ---- Extract columns as lists (much faster than iterrows) ----
            t_parse_start = time.time()
            symbols = batch_df["symbol"].tolist()
            dates = batch_df["report_date"].tolist()
            quarters = batch_df["fiscal_quarter"].tolist()
            years_list = batch_df["fiscal_year"].tolist()
            transcripts_list = batch_df["transcripts"].tolist()
            del batch_df
            gc.collect()

            batch_chunks: List[Dict] = []
            batch_ids: List[str] = []

            for idx in range(len(symbols)):
                total_qualifying += 1
                if max_transcripts and total_qualifying > max_transcripts:
                    break

                ticker = str(symbols[idx]).upper() if symbols[idx] else ""
                if not ticker:
                    continue

                call_date = str(dates[idx]) if dates[idx] else ""
                quarter = int(quarters[idx]) if pd.notna(quarters[idx]) else 0
                year = int(years_list[idx]) if pd.notna(years_list[idx]) else 0

                # Skip if already indexed
                transcript_key = f"{ticker}_{year}_Q{quarter}"
                if transcript_key in indexed_keys:
                    continue

                # Convert transcript data to string
                import numpy as np
                raw_transcript = transcripts_list[idx]
                # Handle numpy.ndarray (from parquet) or list
                if isinstance(raw_transcript, np.ndarray):
                    raw_transcript = raw_transcript.tolist()
                if isinstance(raw_transcript, list):
                    content_parts = []
                    for item in raw_transcript:
                        if isinstance(item, dict):
                            speaker = item.get("speaker", "")
                            text = item.get("content", "")
                            content_parts.append(f"{speaker}: {text}")
                        else:
                            content_parts.append(str(item))
                    content = "\n".join(content_parts)
                else:
                    content = str(raw_transcript)

                if not content or len(content) < 500:
                    total_skipped += 1
                    continue

                # Parse into chunks
                chunks = _parse_and_chunk_transcript(
                    content, ticker, quarter, year, call_date,
                )

                for c_idx, chunk in enumerate(chunks):
                    chunk_id = f"tc_{ticker}_{year}_Q{quarter}_{c_idx}"
                    meta = chunk["metadata"]
                    meta["is_management"] = bool(meta["is_management"])
                    meta["is_analyst"] = bool(meta["is_analyst"])
                    meta["is_qa"] = bool(meta["is_qa"])
                    batch_ids.append(chunk_id)
                    batch_chunks.append(chunk)

                # Flush to ChromaDB when batch is large enough
                if len(batch_chunks) >= EMBED_BATCH_SIZE:
                    t_parse_cum += time.time() - t_parse_start
                    t_e = time.time()
                    total_indexed += self._embed_and_upsert(
                        embed_model, collection, batch_chunks, batch_ids,
                    )
                    t_embed_cum += time.time() - t_e
                    batch_chunks.clear()
                    batch_ids.clear()
                    gc.collect()
                    t_parse_start = time.time()

            t_parse_cum += time.time() - t_parse_start

            # Flush remaining chunks from this row group
            if batch_chunks:
                t_e = time.time()
                total_indexed += self._embed_and_upsert(
                    embed_model, collection, batch_chunks, batch_ids,
                )
                t_embed_cum += time.time() - t_e
                batch_chunks.clear()
                batch_ids.clear()
                gc.collect()

            # ---- Progress with ETA ----
            elapsed = time.time() - t0
            pct = (rg_pos + 1) / len(rg_candidates) * 100
            if rg_pos > 0:
                eta_sec = elapsed / (rg_pos + 1) * (len(rg_candidates) - rg_pos - 1)
                eta_str = f"{eta_sec/60:.0f}m" if eta_sec < 3600 else f"{eta_sec/3600:.1f}h"
            else:
                eta_str = "calculating..."
            logger.info(
                f"  [{rg_pos+1}/{len(rg_candidates)}] {pct:.0f}% | "
                f"{total_indexed:,} chunks | "
                f"parse {t_parse_cum:.0f}s, embed {t_embed_cum:.0f}s | "
                f"ETA {eta_str}"
            )

            if max_transcripts and total_qualifying >= max_transcripts:
                break

        del embed_model
        gc.collect()

        total_elapsed = time.time() - t0
        logger.info(
            f"Transcript indexing complete: {total_indexed:,} chunks from "
            f"{total_qualifying:,} qualifying transcripts in {total_elapsed:.1f}s "
            f"({total_skipped:,} skipped, {total_transcripts_processed:,} scanned). "
            f"Total in store: {collection.count():,}"
        )

        return total_indexed

    def ensure_tickers_indexed(
        self,
        tickers: List[str],
        min_year: int = 2023,
    ) -> int:
        """
        Ensure the given tickers have transcripts in the index.

        Only triggers a targeted build for tickers that are NOT already
        present.  Called automatically by the pipeline so the vectorstore
        is always populated for the companies being analyzed.

        Returns:
            Number of NEW chunks indexed (0 if all tickers were present).
        """
        if not tickers:
            return 0

        collection = self._get_collection()
        existing_count = collection.count()
        if existing_count == 0:
            # Empty store — must build for these tickers
            logger.info(f"Empty transcript index.  Building for {len(tickers)} tickers "
                        f"(min_year={min_year})...")
            return self.build_from_parquet(
                tickers=tickers, min_year=min_year,
            )

        # Check which tickers are already indexed
        tickers_upper = [t.upper() for t in tickers]
        indexed_tickers: set = set()
        try:
            sample_size = min(existing_count, 50000)
            sample = collection.get(limit=sample_size, include=[])
            for doc_id in sample["ids"]:
                parts = doc_id.split("_")
                if len(parts) >= 2:
                    indexed_tickers.add(parts[1])
            del sample
        except Exception:
            pass

        missing = [t for t in tickers_upper if t not in indexed_tickers]
        if not missing:
            logger.info(f"All {len(tickers)} tickers already indexed in transcript store.")
            return 0

        logger.info(f"{len(missing)}/{len(tickers)} tickers not yet indexed: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                    f"Building targeted index...")
        return self.build_from_parquet(
            tickers=missing, min_year=min_year,
        )

    def _embed_and_upsert(
        self,
        embed_model,
        collection,
        chunks: List[Dict],
        ids: List[str],
    ) -> int:
        """Embed a batch of chunks and upsert to ChromaDB. Returns count indexed."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        embeddings_list = embeddings.tolist()
        del embeddings

        indexed = 0
        for sub_start in range(0, len(ids), CHROMA_ADD_BATCH):
            sub_end = min(sub_start + CHROMA_ADD_BATCH, len(ids))
            try:
                collection.upsert(
                    ids=ids[sub_start:sub_end],
                    documents=texts[sub_start:sub_end],
                    metadatas=metas[sub_start:sub_end],
                    embeddings=embeddings_list[sub_start:sub_end],
                )
                indexed += sub_end - sub_start
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")

        return indexed

    # ------------------------------------------------------------------
    # QUERY METHODS
    # ------------------------------------------------------------------

    def query(
        self,
        ticker: str,
        query_text: Optional[str] = None,
        n_results: int = 10,
        management_only: bool = False,
        qa_only: bool = False,
        lookback_quarters: Optional[int] = None,
    ) -> Dict:
        """
        Query transcript chunks for a specific ticker.

        Args:
            ticker: Stock ticker symbol.
            query_text: Semantic query (e.g., "margin guidance for next quarter").
                        If None, defaults to general management outlook query.
            n_results: Maximum results to return.
            management_only: Only return management speaker chunks.
            qa_only: Only return Q&A section chunks.
            lookback_quarters: Limit to last N quarters (by year/quarter metadata).

        Returns:
            Dict with keys: documents, metadatas, distances, ids
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("Transcript vector store is empty. Run build_from_parquet() first.")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        if query_text is None:
            query_text = f"{ticker} management outlook guidance earnings"

        # Build where filter
        conditions = [{"ticker": ticker.upper()}]

        if management_only:
            conditions.append({"is_management": True})
        if qa_only:
            conditions.append({"is_qa": True})

        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )

            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            dists = results["distances"][0] if results["distances"] else []
            ids = results["ids"][0] if results["ids"] else []

            # Filter by lookback quarters
            if lookback_quarters and metas:
                now = datetime.now()
                # Approximate: each quarter is ~90 days
                cutoff_year = now.year
                cutoff_quarter = ((now.month - 1) // 3) + 1 - lookback_quarters
                while cutoff_quarter <= 0:
                    cutoff_quarter += 4
                    cutoff_year -= 1

                filtered = [
                    (d, m, dist, id_)
                    for d, m, dist, id_ in zip(docs, metas, dists, ids)
                    if (m.get("year", 0), m.get("quarter", 0)) >= (cutoff_year, cutoff_quarter)
                ]
                if filtered:
                    docs, metas, dists, ids = zip(*filtered)
                    docs, metas, dists, ids = list(docs), list(metas), list(dists), list(ids)
                else:
                    docs, metas, dists, ids = [], [], [], []

            return {
                "documents": docs,
                "metadatas": metas,
                "distances": dists,
                "ids": ids,
            }

        except Exception as e:
            logger.warning(f"Transcript query failed for {ticker}: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def query_by_theme(
        self,
        theme: str,
        tickers: Optional[List[str]] = None,
        n_results: int = 20,
        management_only: bool = True,
        qa_only: bool = False,
    ) -> Dict:
        """
        Search for a theme across all transcripts.

        Great for finding which companies are discussing specific topics
        (e.g., "tariff impact", "AI spending", "supply chain disruption").

        Args:
            theme: Semantic theme to search for.
            tickers: Optional list of tickers to filter to.
            n_results: Max results.
            management_only: Only management speaker chunks.
            qa_only: Only Q&A section.

        Returns:
            Dict with documents, metadatas, distances, ids.
        """
        collection = self._get_collection()

        conditions = []
        if tickers:
            conditions.append({"ticker": {"$in": [t.upper() for t in tickers]}})
        if management_only:
            conditions.append({"is_management": True})
        if qa_only:
            conditions.append({"is_qa": True})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            results = collection.query(
                query_texts=[theme],
                n_results=n_results,
                where=where_filter,
            )

            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            dists = results["distances"][0] if results["distances"] else []
            ids = results["ids"][0] if results["ids"] else []

            return {
                "documents": docs,
                "metadatas": metas,
                "distances": dists,
                "ids": ids,
            }
        except Exception as e:
            logger.warning(f"Theme query failed for '{theme}': {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def query_peer_comparison(
        self,
        ticker: str,
        peers: List[str],
        query_text: Optional[str] = None,
        n_per_company: int = 5,
        management_only: bool = True,
    ) -> Dict[str, Dict]:
        """
        Compare transcript passages across a company and its peers.

        Retrieves the most relevant passages for the same query across
        multiple companies, enabling tone/content comparison.

        Args:
            ticker: Primary ticker.
            peers: List of peer tickers to compare against.
            query_text: Semantic query. Defaults to management outlook.
            n_per_company: Results per company.
            management_only: Only management speakers.

        Returns:
            Dict mapping ticker -> {documents, metadatas, distances, ids}
        """
        if query_text is None:
            query_text = "management outlook guidance next quarter"

        all_tickers = [ticker.upper()] + [p.upper() for p in peers]
        results = {}

        for t in all_tickers:
            results[t] = self.query(
                ticker=t,
                query_text=query_text,
                n_results=n_per_company,
                management_only=management_only,
            )

        return results

    def query_ticker_history(
        self,
        ticker: str,
        query_text: Optional[str] = None,
        n_results: int = 10,
        management_only: bool = True,
    ) -> Dict:
        """
        Retrieve historical transcript passages for a ticker.

        Unlike query() which might emphasize recency, this explicitly
        searches across all available quarters for a ticker.

        Args:
            ticker: Stock ticker.
            query_text: Semantic query. Defaults to broad management discussion.
            n_results: Max results (across all quarters).
            management_only: Only management speakers.

        Returns:
            Dict with documents, metadatas, distances, ids — sorted by date.
        """
        results = self.query(
            ticker=ticker,
            query_text=query_text,
            n_results=n_results,
            management_only=management_only,
        )

        # Sort by date (most recent first)
        if results["metadatas"]:
            combined = list(zip(
                results["documents"],
                results["metadatas"],
                results["distances"],
                results["ids"],
            ))
            combined.sort(
                key=lambda x: (x[1].get("year", 0), x[1].get("quarter", 0)),
                reverse=True,
            )
            results["documents"] = [c[0] for c in combined]
            results["metadatas"] = [c[1] for c in combined]
            results["distances"] = [c[2] for c in combined]
            results["ids"] = [c[3] for c in combined]

        return results

    # ------------------------------------------------------------------
    # RAG FORMATTING
    # ------------------------------------------------------------------

    def format_context_for_analysis(
        self,
        ticker: str,
        query_text: Optional[str] = None,
        n_results: int = 8,
        management_only: bool = True,
        include_peers: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieve transcript chunks and format as context for analysis.

        This is the main method the EarningsCallQualModel calls for RAG.

        Args:
            ticker: Stock ticker.
            query_text: Semantic query for retrieval.
            n_results: Number of chunks to retrieve.
            management_only: Only management speakers.
            include_peers: If provided, also retrieve peer passages.

        Returns:
            Formatted string ready for analysis context.
        """
        lines = []

        # Primary ticker results
        results = self.query(
            ticker=ticker,
            query_text=query_text,
            n_results=n_results,
            management_only=management_only,
        )

        if results["documents"]:
            lines.append(f"=== TRANSCRIPT CONTEXT FOR {ticker} ===")
            lines.append(f"(Showing {len(results['documents'])} most relevant passages)\n")

            for i, (doc, meta) in enumerate(
                zip(results["documents"], results["metadatas"]), 1
            ):
                q = meta.get("quarter", "?")
                y = meta.get("year", "?")
                role = meta.get("role", "unknown")
                qa_label = "[Q&A]" if meta.get("is_qa") else "[Prepared]"
                date_str = str(meta.get("call_date", ""))[:10]

                lines.append(
                    f"{i}. [{date_str}] Q{q} {y} {qa_label} [{role.upper()}]"
                )
                # Truncate very long chunks
                text_preview = doc[:500].replace("\n", " ").strip()
                if len(doc) > 500:
                    text_preview += "..."
                lines.append(f"   {text_preview}")
                lines.append("")

        # Peer comparison
        if include_peers:
            peer_results = self.query_peer_comparison(
                ticker=ticker,
                peers=include_peers,
                query_text=query_text,
                n_per_company=3,
                management_only=management_only,
            )

            for peer_ticker, peer_data in peer_results.items():
                if peer_ticker == ticker.upper():
                    continue
                if not peer_data["documents"]:
                    continue

                lines.append(f"\n--- PEER: {peer_ticker} ---")
                for i, (doc, meta) in enumerate(
                    zip(peer_data["documents"], peer_data["metadatas"]), 1
                ):
                    q = meta.get("quarter", "?")
                    y = meta.get("year", "?")
                    qa_label = "[Q&A]" if meta.get("is_qa") else "[Prepared]"
                    text_preview = doc[:300].replace("\n", " ").strip()
                    if len(doc) > 300:
                        text_preview += "..."
                    lines.append(f"  {i}. [Q{q} {y}] {qa_label} {text_preview}")
                lines.append("")

        return "\n".join(lines)

    def get_peer_sentiment_context(
        self,
        ticker: str,
        peers: List[str],
        topic: str = "outlook and guidance",
    ) -> Dict[str, List[str]]:
        """
        Get peer management passages on a topic for tone comparison.

        Used by the EarningsCallQualModel to compute peer_tone_delta.

        Args:
            ticker: Primary ticker.
            peers: Peer tickers.
            topic: Topic to search for.

        Returns:
            Dict mapping ticker -> list of management text passages.
        """
        all_tickers = [ticker.upper()] + [p.upper() for p in peers]
        result: Dict[str, List[str]] = {}

        for t in all_tickers:
            query_result = self.query(
                ticker=t,
                query_text=topic,
                n_results=5,
                management_only=True,
                qa_only=True,
            )
            result[t] = query_result.get("documents", [])

        return result


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(description="Transcript Vector Store Management")
    parser.add_argument(
        "action",
        choices=["build", "query", "theme", "peers", "history", "stats"],
        help="Action to perform",
    )
    parser.add_argument("--ticker", help="Ticker to query")
    parser.add_argument("--peers", help="Comma-separated peer tickers")
    parser.add_argument("--theme", help="Theme for cross-ticker query")
    parser.add_argument("--query", help="Custom query text")
    parser.add_argument("--n", type=int, default=10, help="Number of results")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--max", type=int, help="Max transcripts to process")
    parser.add_argument("--tickers", help="Comma-separated tickers to index (for partial build)")
    parser.add_argument("--mgmt-only", action="store_true", default=True,
                        help="Management speakers only (default: True)")
    parser.add_argument("--min-year", type=int, default=None,
                        help="Only index transcripts from this year onward (e.g. 2023)")

    args = parser.parse_args()

    if args.action == "build":
        store = TranscriptVectorStore()
        tickers_filter = args.tickers.split(",") if args.tickers else None
        count = store.build_from_parquet(
            rebuild=args.rebuild,
            max_transcripts=args.max,
            tickers=tickers_filter,
            min_year=args.min_year,
        )
        print(f"\nIndexed {count:,} chunks. Total in store: {store.get_index_count():,}")

    elif args.action == "query":
        if not args.ticker:
            print("Provide --ticker")
        else:
            store = TranscriptVectorStore()
            context = store.format_context_for_analysis(
                ticker=args.ticker,
                query_text=args.query,
                n_results=args.n,
                management_only=args.mgmt_only,
            )
            print(context)

    elif args.action == "theme":
        if not args.theme:
            print("Provide --theme")
        else:
            store = TranscriptVectorStore()
            tickers_filter = args.tickers.split(",") if args.tickers else None
            results = store.query_by_theme(
                args.theme,
                tickers=tickers_filter,
                n_results=args.n,
                management_only=args.mgmt_only,
            )
            for doc, meta in zip(results["documents"], results["metadatas"]):
                t = meta.get("ticker", "?")
                q = meta.get("quarter", "?")
                y = meta.get("year", "?")
                role = meta.get("role", "?")
                print(f"[{t}] Q{q} {y} [{role}] {doc[:120]}...")

    elif args.action == "peers":
        if not args.ticker or not args.peers:
            print("Provide --ticker and --peers (comma-separated)")
        else:
            store = TranscriptVectorStore()
            peer_list = [p.strip() for p in args.peers.split(",")]
            results = store.query_peer_comparison(
                args.ticker,
                peer_list,
                query_text=args.query,
                n_per_company=args.n,
                management_only=args.mgmt_only,
            )
            for t, data in results.items():
                print(f"\n=== {t} ===")
                for doc, meta in zip(data["documents"], data["metadatas"]):
                    q = meta.get("quarter", "?")
                    y = meta.get("year", "?")
                    print(f"  [Q{q} {y}] {doc[:120]}...")

    elif args.action == "history":
        if not args.ticker:
            print("Provide --ticker")
        else:
            store = TranscriptVectorStore()
            results = store.query_ticker_history(
                args.ticker,
                query_text=args.query,
                n_results=args.n,
                management_only=args.mgmt_only,
            )
            for doc, meta in zip(results["documents"], results["metadatas"]):
                q = meta.get("quarter", "?")
                y = meta.get("year", "?")
                date_str = str(meta.get("call_date", ""))[:10]
                print(f"[{date_str}] Q{q} {y}: {doc[:150]}...")

    elif args.action == "stats":
        store = TranscriptVectorStore()
        total = store.get_index_count()
        print(f"Transcript vector store: {total:,} chunks")
        print(f"Chroma path: {store.chroma_path}")

        if total > 0:
            # Sample some metadata to show distribution
            collection = store._get_collection()
            sample = collection.get(limit=100, include=["metadatas"])
            if sample["metadatas"]:
                tickers = set(m.get("ticker", "?") for m in sample["metadatas"])
                years = set(m.get("year", 0) for m in sample["metadatas"])
                mgmt_count = sum(1 for m in sample["metadatas"] if m.get("is_management"))
                qa_count = sum(1 for m in sample["metadatas"] if m.get("is_qa"))
                print(f"Sample tickers: {len(tickers)} unique in first 100 chunks")
                print(f"Year range: {min(years)} - {max(years)}")
                print(f"Management chunks: {mgmt_count}/100")
                print(f"Q&A chunks: {qa_count}/100")
