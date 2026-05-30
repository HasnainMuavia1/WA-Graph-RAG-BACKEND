"""
End-to-end test: ingest a document → vector search → hybrid search → evaluate.
Run: .venv/bin/python e2e_test.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


async def main():
    from agent.db_utils import initialize_database, close_database, get_document_by_source
    from ingestion.ingest_service import IngestService

    print("\n=== 1. DB CONNECTION ===")
    await initialize_database()
    print("Supabase connected.")

    # ── Ingest ────────────────────────────────────────────────────────────────
    print("\n=== 2. INGEST DOCUMENT ===")
    content = (
        "# OpenAI and the GPT-4 Era\n\n"
        "OpenAI released GPT-4 in March 2023, marking a significant leap in language model capabilities. "
        "GPT-4 is a multimodal model capable of processing both text and images. "
        "Microsoft invested $10 billion in OpenAI in January 2023, deepening their strategic partnership. "
        "Azure OpenAI Service allows enterprise customers to access GPT-4 through Microsoft's cloud platform. "
        "OpenAI CEO Sam Altman stated that GPT-4 is the most capable model they have ever released. "
        "The model achieves human-level performance on many professional and academic benchmarks. "
        "GPT-4 passed the bar exam in the 90th percentile; GPT-3.5 scored around the 10th percentile. "
        "The context window of GPT-4 supports up to 128,000 tokens in its turbo variant. "
        "GPT-4 is used in ChatGPT, GitHub Copilot, Microsoft Bing Chat, and many enterprise products."
    )

    svc = IngestService()
    result = await svc.ingest_document(
        content=content,
        source="test://openai-gpt4-overview-v2",
        title="OpenAI GPT-4 Overview",
        metadata={"category": "AI", "year": 2023},
        access_level="public",
    )
    print(f"Status   : {result.status}")
    print(f"Doc ID   : {result.document_id}")
    print(f"Chunks   : {result.chunks_created}")
    if result.error:
        print(f"Error    : {result.error}")
        return

    # ── Verify in DB via Supabase ─────────────────────────────────────────────
    print("\n=== 3. VERIFY IN DB ===")
    doc = await get_document_by_source("test://openai-gpt4-overview-v2")
    print(f"Title    : {doc['title']}")
    print(f"Access   : {doc['access_level']}")
    print(f"Hash     : {doc['content_hash'][:20]}...")

    # ── Vector search ─────────────────────────────────────────────────────────
    print("\n=== 4. VECTOR SEARCH ===")
    from agent.tools import vector_search_tool, VectorSearchInput
    vec_results = await vector_search_tool(VectorSearchInput(query="Microsoft OpenAI investment", limit=3))
    if vec_results:
        for i, r in enumerate(vec_results, 1):
            print(f"[{i}] score={r.score:.3f}  doc={r.document_title}")
            print(f"     {r.content[:120]}...")
    else:
        print("No vector results returned.")

    # ── Hybrid search ─────────────────────────────────────────────────────────
    print("\n=== 5. HYBRID SEARCH (BM25 + vector + RRF) ===")
    from agent.tools import hybrid_search_tool, HybridSearchInput
    hyb_results = await hybrid_search_tool(HybridSearchInput(query="GPT-4 bar exam performance", limit=3))
    if hyb_results:
        for i, r in enumerate(hyb_results, 1):
            print(f"[{i}] score={r.score:.3f}  doc={r.document_title}")
            print(f"     {r.content[:120]}...")
    else:
        print("No hybrid results returned.")

    # ── Deduplication test ────────────────────────────────────────────────────
    print("\n=== 6. DEDUPLICATION TEST (re-ingest same content) ===")
    result2 = await svc.ingest_document(
        content=content,
        source="test://openai-gpt4-overview-v2",
        title="OpenAI GPT-4 Overview",
    )
    print(f"Re-ingest status: {result2.status} (expected: skipped)")

    print("\n=== 7. UPDATE TEST (changed content) ===")
    updated_content = content + "\nGPT-4o was released in May 2024 as a faster and cheaper multimodal successor."
    result3 = await svc.ingest_document(
        content=updated_content,
        source="test://openai-gpt4-overview-v2",
        title="OpenAI GPT-4 Overview (Updated)",
    )
    print(f"Update status: {result3.status} (expected: updated)")
    print(f"New chunks  : {result3.chunks_created}")

    await close_database()
    print("\n=== ALL TESTS COMPLETE ===")


asyncio.run(main())
