"""
News Vector Store.

ChromaDB-backed vector store for news articles, enabling RAG retrieval
for sentiment analysis. Articles are embedded with sentence-transformers
and stored with metadata (ticker, date, topic, sentiment) for efficient
filtered retrieval.

Usage:
    from auto_researcher.data.news_vectorstore import NewsVectorStore
    
    store = NewsVectorStore()
    store.build_from_db()  # One-time indexing
    
    # Retrieve relevant articles for a ticker
    results = store.query("AAPL", n_results=10)
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"[{meta['topic']}] {doc[:80]}...")
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Disable ChromaDB telemetry before any chromadb import (posthog compat issue)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)

# Paths
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "news.db"
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "news_chroma"
DEFAULT_TOPIC_IC_PATH = Path(__file__).parent.parent.parent.parent / "data" / "topic_ic.json"
DEFAULT_TOPIC_IC_DETAILED_PATH = Path(__file__).parent.parent.parent.parent / "data" / "topic_ic_detailed.json"

# Embedding model - small, fast, good for financial text
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Batch size for embedding
EMBED_BATCH_SIZE = 256


class NewsVectorStore:
    """
    ChromaDB vector store over news articles.
    
    Supports:
    - Building the index from news.db (one-time or incremental)
    - Querying by ticker with optional date/topic filters
    - Returning articles with metadata for RAG injection
    - Loading topic IC weights for context-aware retrieval
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        topic_ic_path: Optional[Path] = None,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.chroma_path = Path(chroma_path) if chroma_path else DEFAULT_CHROMA_PATH
        self.embedding_model_name = embedding_model
        self.topic_ic_path = Path(topic_ic_path) if topic_ic_path else DEFAULT_TOPIC_IC_PATH
        
        self._chroma_client = None
        self._collection = None
        self._embedding_fn = None
        self._topic_model = None
        self._topic_ics: dict[str, float] = {}
        self._topic_ics_detailed: dict[str, dict] = {}
        
        # Load topic ICs
        self._load_topic_ics()
    
    def _load_topic_ics(self):
        """Load topic IC weights from JSON."""
        if self.topic_ic_path.exists():
            try:
                with open(self.topic_ic_path) as f:
                    self._topic_ics = json.load(f)
                logger.info(f"Loaded topic ICs for {len(self._topic_ics)} topics")
            except Exception as e:
                logger.warning(f"Failed to load topic ICs: {e}")
        
        detailed_path = self.topic_ic_path.parent / "topic_ic_detailed.json"
        if detailed_path.exists():
            try:
                with open(detailed_path) as f:
                    self._topic_ics_detailed = json.load(f)
            except Exception:
                pass
    
    @property
    def topic_ics(self) -> dict[str, float]:
        """Per-topic IC values."""
        return self._topic_ics
    
    @property
    def topic_ics_detailed(self) -> dict[str, dict]:
        """Per-topic IC values with detailed stats."""
        return self._topic_ics_detailed
    
    def _get_chroma_client(self):
        """Lazy-init ChromaDB client."""
        if self._chroma_client is None:
            import chromadb
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
            )
        return self._chroma_client
    
    def _get_collection(self):
        """Get or create the news collection."""
        if self._collection is None:
            client = self._get_chroma_client()
            self._collection = client.get_or_create_collection(
                name="news_articles",
                metadata={
                    "description": "News articles for RAG-based sentiment analysis",
                    "embedding_model": self.embedding_model_name,
                    "hnsw:space": "cosine",
                },
            )
        return self._collection
    
    def _get_topic_model(self):
        """Lazy-init topic classification model."""
        if self._topic_model is None:
            try:
                from ..models.topic_sentiment import TopicSentimentModel
                self._topic_model = TopicSentimentModel()
            except ImportError:
                logger.warning("TopicSentimentModel not available")
        return self._topic_model
    
    def _classify_topic(self, text: str) -> str:
        """Classify article text into a topic."""
        model = self._get_topic_model()
        if model is None:
            return "general"
        try:
            result = model.analyze_article(text)
            return result.topic.primary_topic
        except Exception:
            return "general"
    
    def _get_embedding_fn(self):
        """Lazy-init the embedding function."""
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
            )
        return self._embedding_fn
    
    def get_index_count(self) -> int:
        """Get the number of documents in the index."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
    
    def build_from_db(
        self,
        batch_size: int = EMBED_BATCH_SIZE,
        max_articles: Optional[int] = None,
        rebuild: bool = False,
    ) -> int:
        """
        Build the vector store from news.db.
        
        Args:
            batch_size: Number of articles to embed per batch.
            max_articles: Limit total articles (for testing).
            rebuild: If True, drop and rebuild the entire index.
            
        Returns:
            Number of articles indexed.
        """
        from sentence_transformers import SentenceTransformer
        
        if rebuild:
            try:
                client = self._get_chroma_client()
                client.delete_collection("news_articles")
                self._collection = None
                logger.info("Dropped existing collection for rebuild")
            except Exception:
                pass
        
        collection = self._get_collection()
        existing_count = collection.count()
        
        if not self.db_path.exists():
            logger.error(f"News database not found: {self.db_path}")
            return 0
        
        # Load articles from SQLite
        logger.info(f"Loading articles from {self.db_path}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT id, ticker, title, snippet, full_text, 
                       published_date, source, sentiment_score, sentiment_label
                FROM articles
                WHERE title IS NOT NULL AND title != ''
                ORDER BY published_date DESC
            """
            if max_articles:
                query += f" LIMIT {max_articles}"
            
            rows = conn.execute(query).fetchall()
        
        logger.info(f"Loaded {len(rows):,} articles from DB (existing index: {existing_count:,})")
        
        # Get existing IDs to skip duplicates
        existing_ids = set()
        if existing_count > 0 and not rebuild:
            try:
                all_results = collection.get(limit=existing_count, include=[])
                existing_ids = set(all_results["ids"])
                logger.info(f"Found {len(existing_ids):,} existing IDs to skip")
            except Exception as e:
                logger.warning(f"Could not fetch existing IDs: {e}")
        
        # Pre-classify all topics in bulk (much faster than per-article)
        logger.info("Classifying topics in bulk...")
        topic_model = self._get_topic_model()
        
        # Build all doc texts and metadata first, then embed in large batches
        all_ids = []
        all_docs = []
        all_metas = []
        
        t0 = time.time()
        
        for i, row in enumerate(rows):
            doc_id = f"art_{row['id']}"
            if doc_id in existing_ids:
                continue
            
            title = row["title"] or ""
            snippet = row["snippet"] or ""
            full_text = row["full_text"] or ""
            
            if full_text and len(full_text) > 50:
                doc_text = f"{title}. {full_text[:1000]}"
            elif snippet:
                doc_text = f"{title}. {snippet}"
            else:
                doc_text = title
            
            if not doc_text.strip():
                continue
            
            # Quick topic classification (keyword-based, very fast)
            topic = self._classify_topic(doc_text) if topic_model else "general"
            
            meta = {
                "ticker": row["ticker"] or "",
                "source": row["source"] or "",
                "topic": topic,
                "published_date": row["published_date"] or "",
                "sentiment_score": float(row["sentiment_score"]) if row["sentiment_score"] is not None else 0.0,
                "sentiment_label": row["sentiment_label"] or "neutral",
                "title": title[:200],
                "article_db_id": int(row["id"]),
            }
            if topic in self._topic_ics:
                meta["topic_ic"] = self._topic_ics[topic]
            
            all_ids.append(doc_id)
            all_docs.append(doc_text)
            all_metas.append(meta)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Prepared {i+1:,}/{len(rows):,} articles...")
        
        if not all_docs:
            logger.info("No new articles to index")
            return 0
        
        logger.info(f"Prepared {len(all_docs):,} articles for embedding in {time.time()-t0:.1f}s")
        
        # Embed in large batches using sentence-transformers directly (much faster)
        logger.info(f"Embedding with {self.embedding_model_name}...")
        model = SentenceTransformer(self.embedding_model_name)
        
        indexed = 0
        embed_batch = 512  # Larger batches for sentence-transformers
        
        for start in range(0, len(all_docs), embed_batch):
            end = min(start + embed_batch, len(all_docs))
            batch_docs = all_docs[start:end]
            batch_ids = all_ids[start:end]
            batch_metas = all_metas[start:end]
            
            # Embed batch
            embeddings = model.encode(batch_docs, show_progress_bar=False, batch_size=64)
            embeddings_list = embeddings.tolist()
            
            # Add to ChromaDB (with pre-computed embeddings, no re-embedding)
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings_list,
                )
                indexed += len(batch_ids)
                elapsed = time.time() - t0
                rate = indexed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Indexed {indexed:,}/{len(all_docs):,} "
                    f"({indexed/len(all_docs)*100:.0f}%, {rate:.0f} docs/sec)"
                )
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
        
        elapsed = time.time() - t0
        logger.info(
            f"Indexing complete: {indexed:,} articles indexed in {elapsed:.1f}s "
            f"({indexed/elapsed:.0f} docs/sec)"
        )
        
        return indexed
    
    def add_articles(
        self,
        articles: list[dict],
        ticker: str = "",
    ) -> int:
        """
        Add articles to the vector store incrementally (no full rebuild).
        
        Used to index freshly fetched articles (yfinance, DefeatBeta, etc.)
        before querying so RAG can find them immediately.
        
        Args:
            articles: List of dicts with keys:
                - title (str, required)
                - snippet (str, optional)
                - source (str, optional)
                - published (str or datetime, optional)
                - url (str, optional)
                - ticker (str, optional — overridden by ticker param)
                - sentiment_score (float, optional)
            ticker: Ticker symbol to tag all articles with.
            
        Returns:
            Number of articles successfully added.
        """
        if not articles:
            return 0
        
        collection = self._get_collection()
        topic_model = self._get_topic_model()
        
        # Prepare documents
        all_ids = []
        all_docs = []
        all_metas = []
        
        # Get existing IDs to avoid duplicates
        existing_ids = set()
        try:
            count = collection.count()
            if count > 0:
                # Check by querying for this ticker's recent docs
                pass  # We'll use title-hash IDs to dedupe
        except Exception:
            pass
        
        for art in articles:
            title = art.get("title", "").strip()
            if not title:
                continue
            
            snippet = art.get("snippet", "") or ""
            doc_text = f"{title}. {snippet}" if snippet else title
            
            # Create deterministic ID from title to avoid duplicates
            title_hash = hash(title.lower()[:80])
            doc_id = f"live_{abs(title_hash)}"
            all_ids.append(doc_id)
            
            # Classify topic
            topic = self._classify_topic(doc_text) if topic_model else "general"
            
            # Parse published date
            pub_date = art.get("published", "")
            if hasattr(pub_date, "isoformat"):
                pub_date = pub_date.isoformat()
            elif not isinstance(pub_date, str):
                pub_date = ""
            
            art_ticker = ticker.upper() or art.get("ticker", "").upper()
            
            meta = {
                "ticker": art_ticker,
                "source": art.get("source", "live"),
                "topic": topic,
                "published_date": str(pub_date),
                "sentiment_score": float(art.get("sentiment_score", 0.0)),
                "sentiment_label": art.get("sentiment_label", "neutral"),
                "title": title[:200],
                "article_db_id": 0,  # Not from DB
            }
            if topic in self._topic_ics:
                meta["topic_ic"] = self._topic_ics[topic]
            
            all_docs.append(doc_text)
            all_metas.append(meta)
        
        if not all_docs:
            return 0
        
        # Embed with sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
            embeddings = model.encode(all_docs, show_progress_bar=False, batch_size=32)
            embeddings_list = embeddings.tolist()
            
            # Upsert to handle potential duplicates gracefully
            collection.upsert(
                ids=all_ids,
                documents=all_docs,
                metadatas=all_metas,
                embeddings=embeddings_list,
            )
            logger.info(f"Added {len(all_docs)} live articles for {ticker} to vector store")
            return len(all_docs)
        except Exception as e:
            logger.warning(f"Failed to add live articles to vector store: {e}")
            return 0

    def query(
        self,
        ticker: str,
        query_text: Optional[str] = None,
        n_results: int = 10,
        lookback_days: int = 30,
        topic_filter: Optional[str] = None,
        min_sentiment_magnitude: float = 0.0,
    ) -> dict:
        """
        Query the vector store for articles relevant to a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            query_text: Optional semantic query (e.g., "earnings guidance").
                        If None, uses the ticker as the query.
            n_results: Maximum number of results to return.
            lookback_days: Only return articles from the last N days.
            topic_filter: Optional topic to filter by.
            min_sentiment_magnitude: Minimum |sentiment_score| to include.
            
        Returns:
            Dict with keys: documents, metadatas, distances, ids
        """
        collection = self._get_collection()
        
        if collection.count() == 0:
            logger.warning("Vector store is empty. Run build_from_db() first.")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        # Build where filter
        where_filter = {"ticker": ticker.upper()}
        
        if topic_filter:
            where_filter = {
                "$and": [
                    {"ticker": ticker.upper()},
                    {"topic": topic_filter},
                ]
            }
        
        # Build query text
        if query_text is None:
            query_text = f"{ticker} stock news sentiment outlook"
        
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )
            
            # Flatten from batch format
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            dists = results["distances"][0] if results["distances"] else []
            ids = results["ids"][0] if results["ids"] else []
            
            # Filter by date if metadata has it
            if lookback_days and metas:
                cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
                filtered = [
                    (d, m, dist, id_)
                    for d, m, dist, id_ in zip(docs, metas, dists, ids)
                    if m.get("published_date", "") >= cutoff or not m.get("published_date")
                ]
                if filtered:
                    docs, metas, dists, ids = zip(*filtered)
                    docs, metas, dists, ids = list(docs), list(metas), list(dists), list(ids)
                else:
                    docs, metas, dists, ids = [], [], [], []
            
            # Filter by sentiment magnitude
            if min_sentiment_magnitude > 0 and metas:
                filtered = [
                    (d, m, dist, id_)
                    for d, m, dist, id_ in zip(docs, metas, dists, ids)
                    if abs(m.get("sentiment_score", 0)) >= min_sentiment_magnitude
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
            logger.warning(f"Vector store query failed for {ticker}: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    def query_by_theme(
        self,
        theme: str,
        tickers: Optional[list[str]] = None,
        n_results: int = 20,
        lookback_days: int = 14,
    ) -> dict:
        """
        Query by theme across multiple tickers.
        
        Useful for finding cross-ticker thematic signals like
        "FDA approvals", "tariff impact", "AI spending", etc.
        
        Args:
            theme: Semantic theme to search for.
            tickers: Optional list of tickers to filter.
            n_results: Max results.
            lookback_days: Date filter.
            
        Returns:
            Dict with documents, metadatas, distances, ids.
        """
        collection = self._get_collection()
        
        where_filter = None
        if tickers:
            where_filter = {"ticker": {"$in": [t.upper() for t in tickers]}}
        
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
    
    def format_context_for_prompt(
        self,
        ticker: str,
        n_results: int = 10,
        lookback_days: int = 30,
        include_topic_ic: bool = True,
        query_text: Optional[str] = None,
    ) -> str:
        """
        Retrieve articles and format them as context for an LLM prompt.
        
        This is the main method the sentiment agent calls for RAG.
        
        Args:
            ticker: Stock ticker.
            n_results: Number of articles to retrieve.
            lookback_days: How far back to look.
            include_topic_ic: Whether to include topic IC weights.
            query_text: Optional custom query for retrieval.
            
        Returns:
            Formatted string ready to inject into an LLM prompt.
        """
        results = self.query(
            ticker=ticker,
            query_text=query_text,
            n_results=n_results,
            lookback_days=lookback_days,
        )
        
        docs = results["documents"]
        metas = results["metadatas"]
        
        if not docs:
            return ""
        
        lines = []
        lines.append(f"=== RETRIEVED NEWS ARTICLES FOR {ticker} ===")
        lines.append(f"(Showing {len(docs)} most relevant articles)\n")
        
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            topic = meta.get("topic", "general")
            sentiment = meta.get("sentiment_score", 0.0)
            date_str = meta.get("published_date", "unknown")[:10]
            source = meta.get("source", "Unknown")
            title = meta.get("title", "")
            
            # Topic IC badge
            ic_badge = ""
            if include_topic_ic and topic in self._topic_ics:
                ic_val = self._topic_ics[topic]
                ic_detail = self._topic_ics_detailed.get(topic, {})
                sig = ic_detail.get("significant", False)
                if sig:
                    ic_badge = f" [IC={ic_val:+.4f}*]"
                else:
                    ic_badge = f" [IC={ic_val:+.4f}]"
            
            # Sentiment badge
            if sentiment > 0.2:
                sent_badge = "📈"
            elif sentiment < -0.2:
                sent_badge = "📉"
            else:
                sent_badge = "➖"
            
            lines.append(f"{i}. [{date_str}] [{topic.upper()}{ic_badge}] {sent_badge} {title}")
            
            # Show article text (truncated)
            text_preview = doc[:300].replace("\n", " ").strip()
            if len(doc) > 300:
                text_preview += "..."
            lines.append(f"   {text_preview}")
            lines.append(f"   Source: {source} | Sentiment: {sentiment:+.2f}")
            lines.append("")
        
        # Add topic IC context
        if include_topic_ic and self._topic_ics:
            lines.append("=== TOPIC PREDICTIVE POWER (IC = correlation with 10-day returns) ===")
            lines.append("Topics marked * are statistically significant (p < 0.05):")
            
            # Sort by absolute IC
            sorted_topics = sorted(
                self._topic_ics.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for topic, ic in sorted_topics:
                detail = self._topic_ics_detailed.get(topic, {})
                sig = "*" if detail.get("significant", False) else " "
                direction = "→ positive sentiment predicts gains" if ic > 0 else "→ positive sentiment predicts losses"
                lines.append(f"  {topic:15s}: IC={ic:+.4f}{sig}  {direction}")
            
            lines.append("")
            lines.append("IMPORTANT: Weight your analysis more heavily on topics with higher |IC|.")
            lines.append("For M&A topics, POSITIVE sentiment is actually a CONTRARIAN indicator (negative IC).")
            lines.append("")
        
        return "\n".join(lines)


# ==============================================================================
# TOPICS TABLE MANAGEMENT
# ==============================================================================

class TopicsTable:
    """
    Manages the article_topics table in news.db.
    
    Stores per-article topic classifications so they don't need
    to be recomputed every time.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._topic_model = None
        self._ensure_table()
    
    def _ensure_table(self):
        """Create the topics table if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS article_topics (
                    article_id INTEGER PRIMARY KEY,
                    topic TEXT NOT NULL,
                    topic_sentiment REAL,
                    topic_confidence REAL,
                    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic 
                ON article_topics(topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_article_topic 
                ON article_topics(article_id, topic)
            """)
            conn.commit()
        logger.info("article_topics table ready")
    
    def _get_topic_model(self):
        """Lazy-init the topic model."""
        if self._topic_model is None:
            from ..models.topic_sentiment import TopicSentimentModel
            self._topic_model = TopicSentimentModel()
        return self._topic_model
    
    def classify_and_store(
        self,
        batch_size: int = 1000,
        max_articles: Optional[int] = None,
    ) -> int:
        """
        Classify all unclassified articles and store topics.
        
        Args:
            batch_size: Process in batches of this size.
            max_articles: Limit total articles to process.
            
        Returns:
            Number of articles classified.
        """
        model = self._get_topic_model()
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Find articles without topic classification
            query = """
                SELECT a.id, a.title, a.snippet, a.sentiment_score
                FROM articles a
                LEFT JOIN article_topics t ON a.id = t.article_id
                WHERE t.article_id IS NULL
                  AND a.title IS NOT NULL AND a.title != ''
                ORDER BY a.published_date DESC
            """
            if max_articles:
                query += f" LIMIT {max_articles}"
            
            rows = conn.execute(query).fetchall()
            logger.info(f"Found {len(rows):,} unclassified articles")
            
            classified = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                
                values = []
                for row in batch:
                    text = row["title"] or ""
                    if row["snippet"]:
                        text += ". " + row["snippet"]
                    
                    try:
                        result = model.analyze_article(text)
                        topic = result.topic.primary_topic
                        topic_sent = result.topic_adjusted_sentiment
                        topic_conf = result.topic.confidence
                    except Exception:
                        topic = "general"
                        topic_sent = float(row["sentiment_score"] or 0.0)
                        topic_conf = 0.0
                    
                    values.append((
                        row["id"],
                        topic,
                        topic_sent,
                        topic_conf,
                    ))
                
                conn.executemany("""
                    INSERT OR IGNORE INTO article_topics 
                    (article_id, topic, topic_sentiment, topic_confidence)
                    VALUES (?, ?, ?, ?)
                """, values)
                conn.commit()
                
                classified += len(values)
                logger.info(f"  Classified {classified:,}/{len(rows):,}")
            
            return classified
    
    def get_topic_distribution(self) -> dict:
        """Get the distribution of topics in the table."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("""
                SELECT topic, COUNT(*) as cnt
                FROM article_topics
                GROUP BY topic
                ORDER BY cnt DESC
            """).fetchall()
            return {row[0]: row[1] for row in rows}
    
    def get_articles_by_topic(
        self,
        ticker: str,
        topic: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get articles for a ticker filtered by topic."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT a.*, t.topic, t.topic_sentiment, t.topic_confidence
                FROM articles a
                JOIN article_topics t ON a.id = t.article_id
                WHERE a.ticker = ? AND t.topic = ?
                ORDER BY a.published_date DESC
                LIMIT ?
            """, (ticker.upper(), topic, limit)).fetchall()
            return [dict(r) for r in rows]


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="News Vector Store Management")
    parser.add_argument(
        "action",
        choices=["build", "query", "classify-topics", "stats"],
        help="Action to perform",
    )
    parser.add_argument("--ticker", help="Ticker to query")
    parser.add_argument("--theme", help="Theme for cross-ticker query")
    parser.add_argument("--n", type=int, default=10, help="Number of results")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--max", type=int, help="Max articles to process")
    
    args = parser.parse_args()
    
    if args.action == "build":
        store = NewsVectorStore()
        count = store.build_from_db(rebuild=args.rebuild, max_articles=args.max)
        print(f"\nIndexed {count:,} articles. Total in store: {store.get_index_count():,}")
        
    elif args.action == "query":
        store = NewsVectorStore()
        if args.ticker:
            context = store.format_context_for_prompt(
                ticker=args.ticker,
                n_results=args.n,
            )
            print(context)
        elif args.theme:
            results = store.query_by_theme(args.theme, n_results=args.n)
            for doc, meta in zip(results["documents"], results["metadatas"]):
                print(f"[{meta['ticker']}] [{meta['topic']}] {meta.get('title', '')[:80]}")
        else:
            print("Provide --ticker or --theme")
    
    elif args.action == "classify-topics":
        tt = TopicsTable()
        count = tt.classify_and_store(max_articles=args.max)
        print(f"\nClassified {count:,} articles")
        print("\nTopic distribution:")
        for topic, cnt in tt.get_topic_distribution().items():
            print(f"  {topic:15s}: {cnt:,}")
    
    elif args.action == "stats":
        store = NewsVectorStore()
        print(f"Vector store articles: {store.get_index_count():,}")
        print(f"\nTopic ICs:")
        for topic, ic in sorted(store.topic_ics.items(), key=lambda x: abs(x[1]), reverse=True):
            detail = store.topic_ics_detailed.get(topic, {})
            sig = "*" if detail.get("significant") else " "
            print(f"  {topic:15s}: IC={ic:+.6f}{sig}")
        
        tt = TopicsTable()
        dist = tt.get_topic_distribution()
        if dist:
            print(f"\nTopics table distribution:")
            for topic, cnt in dist.items():
                print(f"  {topic:15s}: {cnt:,}")
        else:
            print("\nTopics table: empty (run 'classify-topics' first)")
