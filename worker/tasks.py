"""
Celery tasks — the actual units of background work.

Two families:

* **Ingestion** (`ingest_*`) — wrap `IngestService` so document indexing runs
  off the request path, with automatic retry/backoff on transient failures.
* **Messaging** (`process_whatsapp_message_task`) — the full inbound WhatsApp
  pipeline: (voice → Deepgram transcript) → RAG agent → reply back to the user.

Every task delegates its async body through `run_async` (see bootstrap.py),
which guarantees the data layer is initialized on the worker's event loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .bootstrap import run_async
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Retry policy shared by ingestion tasks — transient S3/OpenAI/graph hiccups.
_INGEST_RETRY = dict(
    autoretry_for=(Exception,),
    retry_backoff=True,  # 1s, 2s, 4s, …
    retry_backoff_max=300,  # cap at 5 minutes
    retry_jitter=True,
    max_retries=5,
)


# ── Ingestion tasks ───────────────────────────────────────────────────────────


@celery_app.task(name="worker.tasks.ingest_all_task", bind=True, **_INGEST_RETRY)
def ingest_all_task(self) -> Dict[str, int]:
    """Full sweep of every configured S3 bucket (private + public)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_all_task starting (id=%s)", self.request.id)
    stats = run_async(ingest_service.ingest_all_s3_buckets())
    result = {
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "total": stats.total,
    }
    logger.info("Task ingest_all_task complete: %s", result)
    return result


@celery_app.task(name="worker.tasks.ingest_bucket_task", bind=True, **_INGEST_RETRY)
def ingest_bucket_task(
    self, bucket_type: str = "private", prefix: str = ""
) -> Dict[str, int]:
    """Ingest a single bucket / prefix."""
    from ingestion.ingest_service import ingest_service

    logger.info(
        "Task ingest_bucket_task '%s/%s' (id=%s)", bucket_type, prefix, self.request.id
    )
    stats = run_async(
        ingest_service.ingest_from_s3(bucket_type=bucket_type, prefix=prefix)
    )
    return {
        "bucket_type": bucket_type,
        "prefix": prefix,
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
    }


@celery_app.task(name="worker.tasks.ingest_single_s3_task", bind=True, **_INGEST_RETRY)
def ingest_single_s3_task(self, s3_key: str, bucket_name: str) -> Dict[str, Any]:
    """Ingest one S3 object (used by the S3/SNS webhook)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_single_s3_task '%s' (id=%s)", s3_key, self.request.id)
    result = run_async(ingest_service.ingest_single_s3_object(s3_key, bucket_name))
    return {
        "source": result.source,
        "status": result.status,
        "document_id": result.document_id,
        "chunks_created": result.chunks_created,
        "error": result.error,
    }


@celery_app.task(name="worker.tasks.ingest_document_task", bind=True, **_INGEST_RETRY)
def ingest_document_task(
    self,
    content: str,
    source: str,
    title: str,
    metadata: Optional[Dict[str, Any]] = None,
    access_level: str = "public",
) -> Dict[str, Any]:
    """Ingest an already-parsed document (used by the upload endpoint)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_document_task '%s' (id=%s)", source, self.request.id)
    result = run_async(
        ingest_service.ingest_document(
            content=content,
            source=source,
            title=title,
            metadata=metadata or {},
            access_level=access_level,
        )
    )
    return {
        "source": result.source,
        "status": result.status,
        "document_id": result.document_id,
        "chunks_created": result.chunks_created,
        "error": result.error,
    }


# ── Messaging task (WhatsApp inbound pipeline) ────────────────────────────────


@celery_app.task(
    name="worker.tasks.process_whatsapp_message_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=3,
)
def process_whatsapp_message_task(
    self, message: Dict[str, Any], contact_wa_id: str
) -> Dict[str, Any]:
    """
    Process one inbound WhatsApp message end-to-end.

    `message` is a single entry from the webhook's
    `entry[].changes[].value.messages[]` array. Supports `text` and `audio`
    (voice note) message types. The reply is sent back via the WhatsApp client.
    """
    logger.info(
        "Task process_whatsapp_message_task from %s (id=%s)",
        contact_wa_id,
        self.request.id,
    )
    return run_async(_process_whatsapp_message(message, contact_wa_id))


async def _process_whatsapp_message(
    message: Dict[str, Any], contact_wa_id: str
) -> Dict[str, Any]:
    from integrations.whatsapp import whatsapp_client
    from integrations.deepgram_client import transcriber, TranscriptionError
    from agent.conversation import run_agent_turn
    from agent import conversation_store

    msg_id = message.get("id", "")
    msg_type = message.get("type", "")
    contact_name = message.get("_contact_name")
    # Conversation continuity: one session per WhatsApp user.
    session_id = f"whatsapp:{contact_wa_id}"

    # Best-effort read receipt (non-blocking semantics).
    if msg_id:
        await whatsapp_client.mark_read(msg_id)

    user_text: str = ""
    transcribed = False

    if msg_type == "text":
        user_text = (message.get("text") or {}).get("body", "").strip()

    elif msg_type == "audio":
        audio_meta = message.get("audio") or {}
        media_id = audio_meta.get("id")
        if not media_id:
            await whatsapp_client.send_text(
                contact_wa_id, "I couldn't read that voice note."
            )
            return {"status": "error", "reason": "no media id"}
        try:
            audio_bytes, mime = await whatsapp_client.fetch_media_bytes(media_id)
            user_text = await transcriber.transcribe(audio_bytes, mime_type=mime)
            transcribed = True
        except TranscriptionError as exc:
            logger.warning("Transcription failed: %s", exc)
            await whatsapp_client.send_text(
                contact_wa_id,
                "I couldn't transcribe your voice note. Please try again or send text.",
            )
            return {"status": "error", "reason": str(exc)}

    else:
        await whatsapp_client.send_text(
            contact_wa_id,
            "I can answer text messages and voice notes. Please send one of those.",
        )
        return {"status": "ignored", "type": msg_type}

    if not user_text:
        await whatsapp_client.send_text(
            contact_wa_id, "Your message was empty — please try again."
        )
        return {"status": "ignored", "reason": "empty"}

    # Persist the inbound user message (durable transcript for the admin screen).
    try:
        await conversation_store.record_inbound(
            contact_wa_id,
            user_text,
            message_type=msg_type,
            transcribed=transcribed,
            wa_message_id=msg_id,
            contact_name=contact_name,
        )
    except Exception as exc:  # never block the reply on logging
        logger.warning("Failed to persist inbound message: %s", exc)

    # ── Out-of-Scope (Tier 1) Classifier ─────────────────────────────────────
    from agent.guardrails import classify_query_scope

    scope_verdict = await classify_query_scope(user_text)

    if scope_verdict == "out_of_scope":
        from agent.settings_store import get_config

        reply = (await get_config()).out_of_scope_message
        if transcribed:
            reply = f'🎙️ I heard: "{user_text}"\n\n{reply}'
        await whatsapp_client.send_text(contact_wa_id, reply)
        try:
            await conversation_store.record_outbound(
                contact_wa_id, reply, sender="agent"
            )
        except Exception as exc:
            logger.warning("Failed to persist outbound out-of-scope message: %s", exc)
        return {"status": "out_of_scope_replied", "chars": len(reply)}

    # ── RAG Agent Turn Execution ─────────────────────────────────────────────
    reply, deps = await run_agent_turn(
        message=user_text, session_id=session_id, user_id=session_id
    )

    # If we transcribed a voice note, echo what we heard for transparency.
    if transcribed:
        reply = f'🎙️ I heard: "{user_text}"\n\n{reply}'

    # ── Extract search metrics and build provenance & debug metadata ──────────
    import json
    from datetime import datetime
    from agent.session_memory import memory_manager

    chunks = getattr(deps, "retrieved_chunks", []) or []
    graph_facts = getattr(deps, "graph_facts", []) or []
    selected_tool = getattr(deps, "selected_retrieval_tool", "none") or "none"

    prov_sources = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        page_sec = meta.get("page") or meta.get("section") or meta.get("page_number")
        page_sec_str = f"Page/Section {page_sec}" if page_sec else None
        prov_sources.append(
            {
                "source_document_name": chunk.document_title,
                "chunk_id": chunk.chunk_id,
                "page_section": page_sec_str,
                "retrieval_method_used": selected_tool,
                "confidence_score": float(chunk.score),
            }
        )

    debug_metadata = {
        "provenance": {
            "sources": prov_sources,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_query": user_text,
            "final_answer": reply,
        },
        "debug": {
            "selected_retrieval_tool": selected_tool,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "score": float(c.score),
                    "document_title": c.document_title,
                    "document_source": c.document_source,
                }
                for c in chunks
            ],
            "neo4j_results": [
                {"fact": f.fact, "valid_at": f.valid_at} for f in graph_facts
            ],
            "redis_session_context": memory_manager.get_context_string(session_id),
            "final_generated_prompt_summary": f"Previous conversation context + current question: {user_text}",
            "final_answer": reply,
            "guardrail_result": {
                "input_allowed": True,
                "input_reason": None,
                "output_applied": True,
            },
        },
    }

    # ── Confidence Gate (Tier 2) — stable cosine-similarity threshold ─────────
    from agent.tools import is_low_confidence as _gate, max_cosine_similarity

    top_cosine = max_cosine_similarity(chunks)
    is_low_confidence = _gate(chunks)

    if is_low_confidence:
        # Suppress response to student. Celery alert and admin notification only.
        logger.warning(
            "Low confidence for query '%s' (top_cosine=%.3f). Suppressing reply.",
            user_text,
            top_cosine,
        )

        # Trigger background Celery alert
        notify_admin_weak_context_task.delay(contact_wa_id, user_text)

        # Post a red system alert card inside the thread visible to admin only
        system_alert = (
            "⚠️ [SYSTEM ALERT] Low confidence detected: context is weak/missing. "
            "Celery notification has been triggered. Student was NOT replied to. Please reply manually."
        )
        system_alert_with_prov = (
            system_alert + "\n\n<!--PROVENANCE:" + json.dumps(debug_metadata) + "-->"
        )
        try:
            await conversation_store.record_outbound(
                contact_wa_id, system_alert_with_prov, sender="agent"
            )
        except Exception as exc:
            logger.warning("Failed to persist outbound system alert: %s", exc)
        return {"status": "low_confidence_suppressed", "top_cosine": top_cosine}

    # ── Normal RAG Response ──────────────────────────────────────────────────
    await whatsapp_client.send_text(contact_wa_id, reply)

    # Persist the agent's outbound reply with hidden provenance metadata
    reply_with_prov = reply + "\n\n<!--PROVENANCE:" + json.dumps(debug_metadata) + "-->"
    try:
        await conversation_store.record_outbound(
            contact_wa_id, reply_with_prov, sender="agent"
        )
    except Exception as exc:
        logger.warning("Failed to persist outbound reply message: %s", exc)

    return {"status": "replied", "transcribed": transcribed, "chars": len(reply)}


@celery_app.task(name="worker.tasks.notify_admin_weak_context_task")
def notify_admin_weak_context_task(wa_id: str, user_query: str) -> Dict[str, Any]:
    """
    Background task to notify/alert admins about a low-confidence university question.
    """
    logger.info(
        "ADMIN ALERT [Celery Task]: Student %s asked low-confidence university question: '%s'",
        wa_id,
        user_query,
    )
    return {"status": "notified", "wa_id": wa_id, "query": user_query}
