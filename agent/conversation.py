"""
Reusable agent-turn execution, decoupled from the HTTP layer.

Both the FastAPI `/chat` endpoint and the WhatsApp Celery worker need to:
  1. resolve/create a session,
  2. prepend the sliding-window conversation history,
  3. run the RAG agent,
  4. persist the turn back into LangChain memory.

Keeping it here (instead of in `api.py`) lets the worker process run a turn
without importing the whole FastAPI application.
"""

from __future__ import annotations

import logging
from typing import Optional

from .agent import rag_agent, AgentDependencies
from .guardrails import check_input, apply_output_guardrails
from .session_memory import memory_manager

logger = logging.getLogger(__name__)


async def run_agent_turn(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> tuple[str, AgentDependencies]:
    """Run one guarded agent turn with memory and return (assistant's reply, deps).

    Applies input guardrails (injection/abuse/length) before the model and
    output guardrails (leak scrub, PII redaction, Roman-Urdu enforcement) after.
    """
    session_id = memory_manager.get_or_create(session_id)

    # ── Input guardrails (fail closed) ────────────────────────────────────────
    verdict = check_input(message)
    if not verdict.allowed:
        blocked = verdict.user_message or "Maazrat, main is sawal ka jawab nahi de sakta."
        memory_manager.add_turn(session_id, message, blocked)
        return blocked, AgentDependencies(session_id=session_id, user_id=user_id)
    safe_message = verdict.sanitized_input or message

    try:
        deps = AgentDependencies(session_id=session_id, user_id=user_id)
        deps.retrieved_chunks = []
        deps.graph_facts = []
        deps.selected_retrieval_tool = None

        history = memory_manager.get_context_string(session_id)
        full_prompt = (
            f"Previous conversation:\n{history}\n\nCurrent question: {safe_message}"
            if history
            else safe_message
        )
        result = await rag_agent.run(full_prompt, deps=deps)
        # pydantic-ai >=1.0 exposes `.output`; older versions used `.data`.
        response: str = getattr(result, "output", None) or getattr(result, "data", "")

        # ── Output guardrails ─────────────────────────────────────────────────
        response = await apply_output_guardrails(response)

        memory_manager.add_turn(session_id, message, response)
        return response, deps
    except Exception as exc:
        logger.error("Agent turn failed (session=%s): %s", session_id, exc)
        error_response = (
            "Maazrat — aap ka message process karte hue masla hua. Baraye meharbani dobara koshish karein."
        )
        memory_manager.add_turn(session_id, message, error_response)
        return error_response, AgentDependencies(session_id=session_id, user_id=user_id)

