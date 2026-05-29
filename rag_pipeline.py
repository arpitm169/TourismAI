"""
rag_pipeline.py
===============
Builds and queries a ChromaDB-backed RAG pipeline over tourism data.
Each destination row is vectorised with a sentence-transformer embedding;
additional tourism knowledge documents are injected at init time.

Author : Tourism-AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─── Graceful imports ────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[WARN] chromadb not installed – RAG will use in-memory fallback.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

VECTORSTORE_PATH = "vectorstore"
COLLECTION_NAME  = "tourism_knowledge"
EMBED_MODEL      = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim


# ─── Tourism Knowledge Base ──────────────────────────────────────────────────

TOURISM_KNOWLEDGE_DOCS: List[Dict[str, str]] = [
    {
        "id"      : "kb_001",
        "source"  : "Tourism Economics 101",
        "content" : """
        Revenue per visitor is the single most actionable KPI in destination
        management. Destinations with revenue-per-visitor above $150 are
        considered premium; those below $30 risk over-tourism with low ROI.
        Key levers: accommodation quality, ancillary experiences, local spending.
        """,
    },
    {
        "id"      : "kb_002",
        "source"  : "Sustainable Tourism Guidelines (UNWTO)",
        "content" : """
        Sustainable tourism balances economic growth with ecological preservation.
        Carrying capacity analysis, community-based tourism models, and green
        certification programmes can raise both revenue and visitor satisfaction
        scores. Destinations with high ratings (≥4.0) and eco-certification see
        18% higher repeat-visit rates on average.
        """,
    },
    {
        "id"      : "kb_003",
        "source"  : "Adventure & Eco-Tourism Market Report 2024",
        "content" : """
        Adventure tourism is growing at 46% CAGR globally. Mountain and
        wildlife-based destinations with accommodation infrastructure command
        premium pricing. Average spend: $2,400 per trip. Key markets: millennials
        and Gen-Z travellers aged 22-35 seeking authentic, low-impact experiences.
        """,
    },
    {
        "id"      : "kb_004",
        "source"  : "Cultural Heritage Tourism Analysis",
        "content" : """
        Cultural and heritage destinations attract high-quality tourists with
        longer average stays (5.2 nights vs 2.8 for city breaks). UNESCO World
        Heritage Site status increases visitor numbers by 25-40% within three
        years of designation. Investment in interpretive infrastructure yields
        3.2x ROI over five years.
        """,
    },
    {
        "id"      : "kb_005",
        "source"  : "Accommodation Impact Study",
        "content" : """
        Destinations with local accommodation options generate 2.5x more
        local economic multiplier than day-trip-only sites. Boutique and
        eco-lodges outperform large hotels on satisfaction scores by 0.8 points
        and generate 35% more revenue per room. Capacity constraints are
        the #1 limiting factor for revenue growth in low-accommodation sites.
        """,
    },
    {
        "id"      : "kb_006",
        "source"  : "AI & Digital Transformation in Tourism",
        "content" : """
        Personalisation through AI recommendation engines increases booking
        conversion by 23%. Predictive demand forecasting reduces operational
        costs by 18%. Smart pricing algorithms have shown 14% revenue uplift
        for accommodation operators. Data-driven destination management is
        recognised as a top-5 priority by WTTC member nations for 2025-2030.
        """,
    },
    {
        "id"      : "kb_007",
        "source"  : "Revenue Optimisation Strategies",
        "content" : """
        High-revenue-potential destinations typically share: rating ≥ 4.0,
        accommodation availability, unique category positioning, low direct
        competition in the region, and digital visibility. Investments in
        digital marketing, improved signage, guided tours, and local crafts
        markets can lift revenue 20-40% within 18 months.
        """,
    },
    {
        "id"      : "kb_008",
        "source"  : "Beach & Coastal Tourism Report",
        "content" : """
        Coastal destinations account for 40% of global tourism revenue. Key
        threats: erosion, over-development, and seasonal demand variance.
        Year-round revenue can be improved through off-peak festivals,
        wellness tourism, and marine activity diversification. Blue Flag
        certification raises visitor numbers by 15-20%.
        """,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# TourismRAGPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TourismRAGPipeline:
    """
    Manages document ingestion into ChromaDB and semantic search retrieval.
    Falls back to in-memory cosine similarity search when ChromaDB is absent.
    """

    def __init__(self, persist_dir: str = VECTORSTORE_PATH) -> None:
        self.persist_dir = persist_dir
        self._docs: List[Dict]   = []       # fallback store
        self._embeds: Optional[np.ndarray] = None

        if CHROMA_AVAILABLE:
            self._init_chroma()
        elif ST_AVAILABLE:
            self._model = SentenceTransformer(EMBED_MODEL)
            print("[RAG] Running in in-memory SentenceTransformer mode.")
        else:
            print("[RAG] No vector engine available – returning keyword matches.")

    # ── Chroma init ──────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingest dataset rows ──────────────────────────────────────────────────

    def ingest_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """
        Convert each dataset row into a text document and upsert to vectorstore.
        Returns the number of documents ingested.
        """
        documents, ids, metadatas = [], [], []

        for _, row in df.iterrows():
            text = self._row_to_text(row)
            uid  = self._make_id(row)
            meta = {
                "location"   : str(row.get("Location", "")),
                "country"    : str(row.get("Country", "")),
                "category"   : str(row.get("Category", "")),
                "revenue"    : float(row.get("Revenue", 0)),
                "visitors"   : float(row.get("Visitors", 0)),
                "rating"     : float(row.get("Rating", 0)),
                "high_rev"   : int(row.get("High_Revenue_Potential", 0)),
                "source"     : "dataset",
            }
            documents.append(text)
            ids.append(uid)
            metadatas.append(meta)
            self._docs.append({"id": uid, "content": text, "meta": meta})

        # Ingest knowledge-base documents
        for doc in TOURISM_KNOWLEDGE_DOCS:
            documents.append(textwrap.dedent(doc["content"]).strip())
            ids.append(doc["id"])
            metadatas.append({"source": doc["source"], "high_rev": 0,
                               "location": "", "country": "", "category": "",
                               "revenue": 0.0, "visitors": 0.0, "rating": 0.0})
            self._docs.append({
                "id": doc["id"],
                "content": textwrap.dedent(doc["content"]).strip(),
                "meta": {"source": doc["source"]},
            })

        if CHROMA_AVAILABLE:
            # Upsert in batches
            for i in range(0, len(documents), batch_size):
                self._collection.upsert(
                    documents  = documents[i:i+batch_size],
                    ids        = ids[i:i+batch_size],
                    metadatas  = metadatas[i:i+batch_size],
                )
        elif ST_AVAILABLE:
            self._embeds = self._model.encode(
                [d["content"] for d in self._docs], show_progress_bar=False
            )

        return len(documents)

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search. Returns list of {content, source, score} dicts.
        """
        if CHROMA_AVAILABLE:
            where = filter_dict if filter_dict else None
            results = self._collection.query(
                query_texts  = [question],
                n_results    = n_results,
                where        = where,
            )
            docs = []
            for i, doc in enumerate(results["documents"][0]):
                docs.append({
                    "content"  : doc,
                    "metadata" : results["metadatas"][0][i],
                    "distance" : results["distances"][0][i],
                })
            return docs

        elif ST_AVAILABLE and self._embeds is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec   = self._model.encode([question])
            scores  = cosine_similarity(q_vec, self._embeds)[0]
            top_idx = np.argsort(scores)[::-1][:n_results]
            return [
                {
                    "content"  : self._docs[i]["content"],
                    "metadata" : self._docs[i]["meta"],
                    "distance" : float(1 - scores[i]),
                }
                for i in top_idx
            ]

        else:
            # keyword fallback
            q_lower = question.lower()
            return [
                {"content": d["content"], "metadata": d.get("meta", {}), "distance": 0.5}
                for d in self._docs
                if any(tok in d["content"].lower() for tok in q_lower.split())
            ][:n_results]

    def get_context_for_prompt(self, question: str, n_results: int = 4) -> str:
        """Return retrieved chunks formatted for injection into an LLM prompt."""
        results = self.query(question, n_results=n_results)
        if not results:
            return "No relevant context found."
        parts = []
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", r["metadata"].get("location", "Unknown"))
            parts.append(f"[Context {i} – {src}]\n{r['content'].strip()}")
        return "\n\n".join(parts)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        acc = "has" if row.get("Accommodation_Available", 0) == 1 else "does not have"
        hrp = "HIGH" if row.get("High_Revenue_Potential", 0) == 1 else "standard"
        return (
            f"Location: {row.get('Location', 'Unknown')} in {row.get('Country', 'Unknown')}. "
            f"Category: {row.get('Category', 'Unknown')}. "
            f"Annual visitors: {int(row.get('Visitors', 0)):,}. "
            f"Average rating: {row.get('Rating', 0):.1f}/5. "
            f"Annual revenue: ${int(row.get('Revenue', 0)):,}. "
            f"Revenue per visitor: ${row.get('Revenue_per_Visitor', 0):.2f}. "
            f"The destination {acc} accommodation. "
            f"Revenue potential classification: {hrp}."
        )

    @staticmethod
    def _make_id(row: pd.Series) -> str:
        key = f"{row.get('Location', '')}{row.get('Country', '')}"
        return "dst_" + hashlib.md5(key.encode()).hexdigest()[:12]

    def collection_size(self) -> int:
        if CHROMA_AVAILABLE:
            return self._collection.count()
        return len(self._docs)
