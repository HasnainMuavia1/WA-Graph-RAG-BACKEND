"""
Pydantic AI RAG agent with vector and knowledge-graph search tools.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .tools import (
    DocumentListInput,
    GraphSearchInput,
    HybridSearchInput,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Runtime dependencies injected into every agent tool call."""

    session_id: str
    user_id: Optional[str] = None
    retrieved_chunks: Optional[list] = None
    graph_facts: Optional[list] = None
    selected_retrieval_tool: Optional[str] = None


_model = OpenAIModel(
    os.getenv("LLM_CHOICE", "gpt-4o-mini"),
    provider=OpenAIProvider(
        api_key=(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""),
    ),
)

rag_agent: Agent = Agent(
    _model,
    deps_type=AgentDependencies,
)


@rag_agent.system_prompt
async def _dynamic_system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Pull the admin-editable system prompt from settings (cached, fail-safe)."""
    from .settings_store import get_system_prompt

    return await get_system_prompt()


@rag_agent.tool
async def search_documents(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
) -> str:
    """
    Search the document knowledge base using hybrid vector + keyword search.
    Returns the most relevant document chunks ranked by relevance score.
    """
    results = await hybrid_search_tool(
        HybridSearchInput(query=query, limit=limit, user_id=ctx.deps.user_id)
    )
    if ctx.deps.retrieved_chunks is None:
        ctx.deps.retrieved_chunks = []
    ctx.deps.retrieved_chunks.extend(results)
    ctx.deps.selected_retrieval_tool = "hybrid_search"

    if not results:
        return "No relevant documents found."

    lines = []
    for i, chunk in enumerate(results, 1):
        lines.append(
            f"[{i}] {chunk.document_title} (score={chunk.score:.3f})\n"
            f"    {chunk.content[:500]}"
        )
    return "\n\n".join(lines)


@rag_agent.tool
async def search_knowledge_graph_facts(
    ctx: RunContext[AgentDependencies],
    query: str,
) -> str:
    """
    Search the knowledge graph for structured facts and entity relationships.
    Returns facts relevant to the query, with temporal validity where available.
    """
    results = await graph_search_tool(GraphSearchInput(query=query))
    if ctx.deps.graph_facts is None:
        ctx.deps.graph_facts = []
    ctx.deps.graph_facts.extend(results)
    if not ctx.deps.selected_retrieval_tool:
        ctx.deps.selected_retrieval_tool = "graph_search"

    if not results:
        return "No knowledge-graph facts found."

    lines = []
    for i, r in enumerate(results, 1):
        entry = f"[{i}] {r.fact}"
        if r.valid_at:
            entry += f" (valid from {r.valid_at})"
        lines.append(entry)
    return "\n\n".join(lines)


@rag_agent.tool
async def list_available_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 5,
) -> str:
    """List the titles and sources of recently available documents."""
    docs = await list_documents_tool(
        DocumentListInput(limit=limit, user_id=ctx.deps.user_id)
    )
    if not docs:
        return "No documents available."
    return "\n".join(
        f"- {d.title}  (source: {d.source}, chunks: {d.chunk_count})" for d in docs
    )
