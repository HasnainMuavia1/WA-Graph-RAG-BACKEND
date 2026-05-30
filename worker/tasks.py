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
    retry_backoff=True,        # 1s, 2s, 4s, …
    retry_backoff_max=300,     # cap at 5 minutes
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
def ingest_bucket_task(self, bucket_type: str = "private", prefix: str = "") -> Dict[str, int]:
    """Ingest a single bucket / prefix."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_bucket_task '%s/%s' (id=%s)", bucket_type, prefix, self.request.id)
    stats = run_async(ingest_service.ingest_from_s3(bucket_type=bucket_type, prefix=prefix))
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
def process_whatsapp_message_task(self, message: Dict[str, Any], contact_wa_id: str) -> Dict[str, Any]:
    """
    Process one inbound WhatsApp message end-to-end.

    `message` is a single entry from the webhook's
    `entry[].changes[].value.messages[]` array. Supports `text` and `audio`
    (voice note) message types. The reply is sent back via the WhatsApp client.
    """
    logger.info("Task process_whatsapp_message_task from %s (id=%s)", contact_wa_id, self.request.id)
    return run_async(_process_whatsapp_message(message, contact_wa_id))


async def _process_whatsapp_message(message: Dict[str, Any], contact_wa_id: str) -> Dict[str, Any]:
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
            await whatsapp_client.send_text(contact_wa_id, "I couldn't read that voice note.")
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
        await whatsapp_client.send_text(contact_wa_id, "Your message was empty — please try again.")
        return {"status": "ignored", "reason": "empty"}

    # Persist the inbound user message (durable transcript for the admin screen).
    try:
        await conversation_store.record_inbound(
            contact_wa_id, user_text,
            message_type=msg_type, transcribed=transcribed,
            wa_message_id=msg_id, contact_name=contact_name,
        )
    except Exception as exc:  # never block the reply on logging
        logger.warning("Failed to persist inbound message: %s", exc)

    # Run the RAG agent with per-user memory.
    reply = await run_agent_turn(message=user_text, session_id=session_id, user_id=session_id)

    # If we transcribed a voice note, echo what we heard for transparency.
    if transcribed:
        reply = f'🎙️ I heard: "{user_text}"\n\n{reply}'

    await whatsapp_client.send_text(contact_wa_id, reply)

    # Persist the agent's outbound reply.
    try:
        await conversation_store.record_outbound(contact_wa_id, reply, sender="agent")
    except Exception as exc:
        logger.warning("Failed to persist outbound message: %s", exc)

    return {"status": "replied", "transcribed": transcribed, "chars": len(reply)}
