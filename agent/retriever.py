"""
Hybrid retriever: LlamaIndex BM25 + pgvector with Reciprocal Rank Fusion.

Architecture
------------
- Vector leg  : pgvector cosine similarity (existing db_utils.vector_search)
- BM25 leg    : LlamaIndex BM25Retriever built from all stored chunks
- Fusion      : Reciprocal Rank Fusion (RRF) merges both ranked lists

The module exposes a single ``hybrid_retriever`` singleton that the agent
tools use. Call ``await hybrid_retriever.build_index()`` once at startup
and ``await hybrid_retriever.rebuild_index()`` after every ingestion run.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# LlamaIndex imports (optional at module level so the module is importable
# even when llama-index is not installed; errors are raised at call time).
try:
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.warning(
        "llama-index-retrievers-bm25 not installed — BM25 leg disabled; "
        "run: pip install llama-index-retrievers-bm25"
    )


class HybridRetriever:
    """
    Combines BM25 keyword retrieval with dense vector retrieval and fuses
    the ranked lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    bm25_top_k:
        How many results to request from the BM25 retriever.
    vector_top_k:
        How many results to request from pgvector.
    rrf_k:
        RRF constant (typically 60). Larger values smooth rank differences.
    """

    def __init__(
        self,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k
        self._bm25: Optional[object] = None  # BM25Retriever
        self._node_map: Dict[str, Dict[str, Any]] = {}

    # ── Index management ──────────────────────────────────────────────────────

    async def build_index(self, chunks: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build (or rebuild) the in-memory BM25 index.

        Parameters
        ----------
        chunks:
            Pre-loaded list of chunk dicts (keys: chunk_id, content, …).
            When *None* the method loads all chunks from the database.
        """
        if not _LLAMA_AVAILABLE:
            logger.warning("BM25 index skipped — llama-index not installed")
            return

        if chunks is None:
            chunks = await self._load_all_chunks()

        if not chunks:
            logger.info("No chunks in database — BM25 index is empty")
            return

        nodes = [self._to_node(c) for c in chunks]
        self._node_map = {str(c["chunk_id"]): c for c in chunks}

        loop = asyncio.get_event_loop()
        self._bm25 = await loop.run_in_executor(
            None,
            lambda: BM25Retriever.from_defaults(
                nodes=nodes, similarity_top_k=self.bm25_top_k
            ),
        )
        logger.info("BM25 index built with %d nodes", len(nodes))

    async def rebuild_index(self) -> None:
        """Reload all chunks from the database and rebuild the BM25 index."""
        logger.info("Rebuilding BM25 index after ingestion …")
        await self.build_index(chunks=None)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        embedding: List[float],
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run both search legs in parallel then merge with RRF.

        Returns a list of chunk dicts enriched with ``combined_score`` and
        ``rrf_score`` fields, sorted descending by RRF score.
        """
        from .db_utils import vector_search  # late import to avoid circular

        vector_task = vector_search(
            embedding=embedding, limit=self.vector_top_k, user_id=user_id
        )
        bm25_task = self._bm25_search(query)

        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task, return_exceptions=True
        )

        if isinstance(vector_results, Exception):
            logger.error("Vector search error: %s", vector_results)
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.error("BM25 search error: %s", bm25_results)
            bm25_results = []

        return self._rrf_fusion(vector_results, bm25_results, limit)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        if not self._bm25:
            return []

        loop = asyncio.get_event_loop()
        retrieved_nodes = await loop.run_in_executor(None, self._bm25.retrieve, query)

        results = []
        for node in retrieved_nodes:
            chunk_id = node.node_id
            base = self._node_map.get(chunk_id, {})
            results.append(
                {
                    **base,
                    "chunk_id": chunk_id,
                    "content": node.text,
                    "bm25_score": float(node.score or 0.0),
                    "similarity": float(node.score or 0.0),
                    "document_title": node.metadata.get("document_title", ""),
                    "document_source": node.metadata.get("document_source", ""),
                    "document_id": node.metadata.get("document_id", ""),
                    "metadata": node.metadata,
                }
            )
        return results

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion over the two ranked lists."""
        rrf_scores: Dict[str, float] = {}
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, r in enumerate(vector_results):
            cid = str(r.get("chunk_id", ""))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + self.rrf_k)
            merged[cid] = {**r, "vector_rank": rank + 1}

        for rank, r in enumerate(bm25_results):
            cid = str(r.get("chunk_id", ""))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + self.rrf_k)
            if cid not in merged:
                merged[cid] = {**r}
            merged[cid]["bm25_rank"] = rank + 1

        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:limit]

        output = []
        for cid in top_ids:
            row = merged[cid]
            row["rrf_score"] = rrf_scores[cid]
            row["combined_score"] = rrf_scores[cid]
            row.setdefault("similarity", row.get("rrf_score", 0.0))
            output.append(row)

        return output

    @staticmethod
    async def _load_all_chunks() -> List[Dict[str, Any]]:
        """Load every chunk from Supabase for BM25 index construction."""
        from .db_utils import _client

        if not _client:
            return []
        try:
            result = await (
                _client.table("chunks")
                .select("id, document_id, content, metadata, documents(title, source)")
                .execute()
            )
            rows = []
            for r in result.data or []:
                doc = r.pop("documents", None) or {}
                rows.append(
                    {
                        "chunk_id": r["id"],
                        "document_id": r["document_id"],
                        "content": r["content"],
                        "metadata": r.get("metadata", {}),
                        "document_title": doc.get("title", ""),
                        "document_source": doc.get("source", ""),
                    }
                )
            return rows
        except Exception as exc:
            logger.error("Failed to load chunks for BM25 index: %s", exc)
            return []

    @staticmethod
    def _to_node(chunk: Dict[str, Any]) -> "TextNode":
        return TextNode(
            id_=str(chunk["chunk_id"]),
            text=chunk.get("content", ""),
            metadata={
                "document_id": str(chunk.get("document_id", "")),
                "document_title": chunk.get("document_title", ""),
                "document_source": chunk.get("document_source", ""),
            },
        )


# ── Module-level singleton ────────────────────────────────────────────────────

hybrid_retriever = HybridRetriever()
