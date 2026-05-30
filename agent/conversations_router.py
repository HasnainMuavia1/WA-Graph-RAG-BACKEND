"""
Agent Conversation endpoints — /api/v1/conversations

Powers the admin "Agent Conversation" screen: list WhatsApp conversations, read
the full user↔agent transcript, and send a manual reply (which is delivered to
the user via the WhatsApp Cloud API and saved to the transcript).

All routes require an authenticated, active dashboard user.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import conversation_store
from .auth_deps import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

_auth = get_current_active_user

# WhatsApp ids are digits only (country code + number, no '+').
_WA_ID_RE = re.compile(r"^\d{6,15}$")


def _normalize_wa_id(raw: str) -> str:
    """Strip '+'/spaces/dashes and validate an E.164-style WhatsApp id."""
    digits = re.sub(r"\D", "", raw or "")
    if not _WA_ID_RE.match(digits):
        raise HTTPException(status_code=400, detail="Invalid WhatsApp number (use digits, country code first).")
    return digits


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096, description="Reply text")


class StartConversationRequest(BaseModel):
    wa_id: str = Field(..., description="Recipient WhatsApp number (digits, country code first)")
    contact_name: Optional[str] = Field(None, max_length=120)


@router.get("")
async def list_conversations_endpoint(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    _user=Depends(_auth),
):
    """List WhatsApp conversations, most recently active first."""
    convos = await conversation_store.list_conversations(limit=limit, offset=offset, search=search)
    return {"conversations": convos, "limit": limit, "offset": offset, "count": len(convos)}


@router.post("")
async def start_conversation_endpoint(body: StartConversationRequest, _user=Depends(_auth)):
    """Start (or open) a conversation with any number — for admin-initiated chats."""
    wa_id = _normalize_wa_id(body.wa_id)
    conv = await conversation_store.start_conversation(wa_id, body.contact_name)
    return {"status": "ok", "conversation": conv}


@router.get("/{wa_id}/messages")
async def get_messages_endpoint(
    wa_id: str,
    limit: int = 200,
    after: Optional[str] = None,
    mark_read: bool = True,
    _user=Depends(_auth),
):
    """Return the transcript for a conversation (oldest → newest).

    `after` (ISO timestamp) returns only newer messages — used by polling.
    Opening a thread (no `after`) marks it read by default.
    """
    conv = await conversation_store.get_conversation(wa_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await conversation_store.get_messages(wa_id, limit=limit, after=after)
    if mark_read and not after:
        await conversation_store.mark_read(wa_id)
    return {"conversation": conv, "messages": messages, "count": len(messages)}


@router.post("/{wa_id}/messages")
async def send_message_endpoint(
    wa_id: str,
    body: SendMessageRequest,
    user=Depends(_auth),
):
    """Send a manual admin message: deliver via WhatsApp, then persist it.

    The conversation is created on the fly if it doesn't exist yet, so admins can
    start a brand-new chat by sending the first message.
    """
    from integrations.whatsapp import whatsapp_client, WhatsAppConfigError

    wa_id = _normalize_wa_id(wa_id)
    await conversation_store.start_conversation(wa_id)

    try:
        resp = await whatsapp_client.send_text(wa_id, body.content)
    except WhatsAppConfigError as exc:
        raise HTTPException(status_code=503, detail=f"WhatsApp not configured: {exc}")
    except Exception as exc:
        logger.error("Admin send to %s failed: %s", wa_id, exc)
        raise HTTPException(status_code=502, detail=f"WhatsApp send failed: {exc}")

    wa_message_id = None
    try:
        wa_message_id = (resp.get("messages") or [{}])[0].get("id")
    except Exception:
        pass

    msg = await conversation_store.record_outbound(
        wa_id, body.content, sender="admin", wa_message_id=wa_message_id
    )
    return {"status": "sent", "message": msg}


@router.post("/{wa_id}/read")
async def mark_read_endpoint(wa_id: str, _user=Depends(_auth)):
    """Reset a conversation's unread counter."""
    await conversation_store.mark_read(wa_id)
    return {"status": "ok"}
