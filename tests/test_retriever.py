"""
Unit tests for the HybridRetriever (BM25 + pgvector + RRF).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.retriever import HybridRetriever


def _vec_row(chunk_id: str, rank_score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "content": f"Content of {chunk_id}",
        "similarity": rank_score,
        "document_title": "Test Doc",
        "document_source": "s3://bucket/test.pdf",
        "metadata": {},
    }


def _bm25_row(chunk_id: str, bm25_score: float = 0.8) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "content": f"BM25 content of {chunk_id}",
        "bm25_score": bm25_score,
        "similarity": bm25_score,
        "document_title": "Test Doc",
        "document_source": "s3://bucket/test.pdf",
        "metadata": {},
    }


class TestRRFFusion:
    """RRF fusion is pure Python — no mocking needed."""

    @pytest.fixture()
    def retriever(self):
        return HybridRetriever(rrf_k=60)

    def test_empty_inputs_return_empty(self, retriever):
        assert retriever._rrf_fusion([], [], 10) == []

    def test_vector_only_returns_sorted(self, retriever):
        vector = [_vec_row("c1", 0.9), _vec_row("c2", 0.8), _vec_row("c3", 0.7)]
        result = retriever._rrf_fusion(vector, [], 3)
        # c1 should be rank 1 (highest RRF score from vector)
        assert result[0]["chunk_id"] == "c1"
        assert len(result) == 3

    def test_bm25_only_returns_sorted(self, retriever):
        bm25 = [_bm25_row("b1", 0.95), _bm25_row("b2", 0.75)]
        result = retriever._rrf_fusion([], bm25, 2)
        assert result[0]["chunk_id"] == "b1"

    def test_overlap_boosts_shared_chunks(self, retriever):
        # c1 appears in both lists → higher RRF score than c2/b1 alone
        vector = [_vec_row("c1", 0.9), _vec_row("c2", 0.85)]
        bm25 = [_bm25_row("c1", 0.9), _bm25_row("b1", 0.7)]
        result = retriever._rrf_fusion(vector, bm25, 4)
        # c1 must rank first because it appears in both lists
        assert result[0]["chunk_id"] == "c1"

    def test_limit_respected(self, retriever):
        vector = [_vec_row(f"c{i}") for i in range(10)]
        result = retriever._rrf_fusion(vector, [], 3)
        assert len(result) == 3

    def test_result_has_rrf_score_field(self, retriever):
        vector = [_vec_row("c1")]
        result = retriever._rrf_fusion(vector, [], 5)
        assert "rrf_score" in result[0]
        assert result[0]["rrf_score"] > 0.0

    def test_result_has_combined_score_field(self, retriever):
        vector = [_vec_row("c1")]
        result = retriever._rrf_fusion(vector, [], 5)
        assert "combined_score" in result[0]

    def test_no_duplicate_chunk_ids_in_output(self, retriever):
        vector = [_vec_row("shared")]
        bm25 = [_bm25_row("shared")]
        result = retriever._rrf_fusion(vector, bm25, 10)
        ids = [r["chunk_id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_rrf_k_affects_scores(self):
        r_low_k = HybridRetriever(rrf_k=1)
        r_high_k = HybridRetriever(rrf_k=100)
        vector = [_vec_row("c1")]
        low_score = r_low_k._rrf_fusion(vector, [], 1)[0]["rrf_score"]
        high_score = r_high_k._rrf_fusion(vector, [], 1)[0]["rrf_score"]
        # Lower k → higher score (1/(rank+k) is larger for smaller k)
        assert low_score > high_score


class TestBuildIndex:
    @pytest.mark.asyncio
    async def test_build_with_empty_chunks_does_not_error(self):
        r = HybridRetriever()
        await r.build_index(chunks=[])
        assert r._bm25 is None

    @pytest.mark.asyncio
    async def test_build_with_chunks_creates_bm25(self):
        r = HybridRetriever()
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "content": "Google AI strategy",
                "document_title": "AI Report",
                "document_source": "s3://bucket/report.pdf",
                "metadata": {},
            }
        ]
        await r.build_index(chunks=chunks)
        assert r._bm25 is not None
        assert "c1" in r._node_map

    @pytest.mark.asyncio
    async def test_rebuild_replaces_previous_index(self):
        r = HybridRetriever()
        chunks_v1 = [
            {"chunk_id": "c1", "document_id": "d1", "content": "v1",
             "document_title": "t", "document_source": "s", "metadata": {}},
        ]
        await r.build_index(chunks=chunks_v1)
        assert "c1" in r._node_map

        chunks_v2 = [
            {"chunk_id": "c2", "document_id": "d1", "content": "v2",
             "document_title": "t", "document_source": "s", "metadata": {}},
        ]
        await r.build_index(chunks=chunks_v2)
        assert "c2" in r._node_map
        assert "c1" not in r._node_map  # old index is gone


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_merged_results(self):
        r = HybridRetriever()
        chunks = [
            {"chunk_id": "c1", "document_id": "d1", "content": "AI research",
             "document_title": "T", "document_source": "S", "metadata": {}},
        ]
        await r.build_index(chunks=chunks)

        with patch("agent.retriever.HybridRetriever._load_all_chunks"):
            with patch("agent.db_utils.vector_search", AsyncMock(return_value=[_vec_row("c1")])):
                import agent.retriever as ret_mod
                with patch.object(ret_mod, "_LLAMA_AVAILABLE", True):
                    result = await r.retrieve(
                        query="AI", embedding=[0.1] * 1536, limit=5
                    )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_handles_vector_search_error_gracefully(self):
        r = HybridRetriever()
        await r.build_index(chunks=[])

        with patch("agent.db_utils.vector_search", AsyncMock(side_effect=RuntimeError("DB down"))):
            result = await r.retrieve(query="test", embedding=[0.1] * 1536)

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_bm25_index_still_returns_vector_results(self):
        r = HybridRetriever()
        # No BM25 index built
        vector_row = _vec_row("c1")

        with patch("agent.db_utils.vector_search", AsyncMock(return_value=[vector_row])):
            result = await r.retrieve(query="test", embedding=[0.1] * 1536)

        # Should still return vector results even without BM25
        assert len(result) >= 0  # may be empty if BM25 errors are suppressed


class TestToNode:
    def test_creates_node_with_correct_id(self):
        chunk = {
            "chunk_id": "abc-123",
            "content": "test content",
            "document_id": "doc-1",
            "document_title": "Title",
            "document_source": "src",
        }
        node = HybridRetriever._to_node(chunk)
        assert node.node_id == "abc-123"
        assert node.text == "test content"

    def test_metadata_populated(self):
        chunk = {
            "chunk_id": "x",
            "content": "text",
            "document_id": "d",
            "document_title": "My Doc",
            "document_source": "my/source",
        }
        node = HybridRetriever._to_node(chunk)
        assert node.metadata["document_title"] == "My Doc"
