"""
WhatsApp Cloud API webhook router.

Two endpoints under /api/v1/whatsapp/webhook:

* **GET**  — Meta's one-time verification handshake. Meta calls this with
  `hub.mode`, `hub.verify_token`, `hub.challenge`; we echo the challenge back
  iff the token matches `WHATSAPP_VERIFY_TOKEN`.

* **POST** — inbound events. We (1) verify the `X-Hub-Signature-256` HMAC,
  (2) ACK immediately with 200 (Meta retries aggressively on non-200), and
  (3) hand each user message to a Celery task so transcription + agent run
  happen off the request path.

Status events (delivered/read receipts) are acknowledged and ignored.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from integrations.whatsapp import whatsapp_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/whatsapp", tags=["whatsapp"])


@router.get("/webhook")
async def verify_webhook(request: Request):
    """Meta verification handshake — echo hub.challenge when the token matches."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    result = whatsapp_client.verify_subscription(mode, token, challenge)
    if result is not None:
        logger.info("WhatsApp webhook verified")
        return PlainTextResponse(content=result, status_code=200)

    logger.warning("WhatsApp webhook verification failed (mode=%s)", mode)
    return PlainTextResponse(content="Verification failed", status_code=403)


@router.post("/webhook")
async def receive_webhook(request: Request):
    """Receive inbound WhatsApp events and enqueue them for processing."""
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not whatsapp_client.verify_signature(raw_body, signature):
        logger.warning("WhatsApp webhook signature verification failed")
        return Response(status_code=403)

    try:
        payload = await request.json()
    except Exception:
        # Always 200 so Meta doesn't hammer us with retries on a bad body.
        return {"status": "ignored"}

    queued = 0
    # Lazy import keeps the API importable even if Celery isn't installed in
    # some lightweight contexts (e.g. unit tests of the verify handshake).
    from worker.tasks import process_whatsapp_message_task

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            if not messages:
                # Could be a status callback (sent/delivered/read) — ignore.
                continue
            # `contacts[0].wa_id` is the sender's WhatsApp id (E.164, no '+').
            contacts = value.get("contacts", [])
            contact_name = None
            if contacts:
                contact_name = (contacts[0].get("profile") or {}).get("name")
            for msg in messages:
                wa_id = msg.get("from") or (contacts[0].get("wa_id") if contacts else None)
                if not wa_id:
                    continue
                # Carry the display name through so the store can label the thread.
                msg = {**msg, "_contact_name": contact_name}
                process_whatsapp_message_task.delay(message=msg, contact_wa_id=wa_id)
                queued += 1

    logger.info("WhatsApp webhook accepted — %d message(s) queued", queued)
    return {"status": "accepted", "queued": queued}
