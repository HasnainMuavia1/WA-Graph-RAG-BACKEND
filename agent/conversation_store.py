"""
Persistent WhatsApp conversation log (Supabase).

This is the durable transcript that powers the admin "Agent Conversation" screen
— distinct from the ephemeral Redis agent-context memory in `session_memory.py`.

Tables (see migration `create_wa_conversation_tables`):
  * wa_conversations — one row per WhatsApp contact (`wa_id`), with rollup fields
    (last message preview/time/direction, unread count).
  * wa_messages — every inbound/outbound message.

Every function is async and uses the shared Supabase client (`db_utils._client`),
which both the API process and the Celery worker initialize.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


def _client():
    """Return the live Supabase client (raises if the app isn't initialized)."""
    from . import db_utils

    if db_utils._client is None:
        raise RuntimeError("Supabase client not initialized")
    return db_utils._client


async def _ensure_conversation(wa_id: str, contact_name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch the conversation for `wa_id`, creating it on first contact."""
    client = _client()
    res = await client.table("wa_conversations").select("*").eq("wa_id", wa_id).limit(1).execute()
    rows = res.data or []
    if rows:
        # Backfill a contact name if we learned one and it was empty.
        if contact_name and not rows[0].get("contact_name"):
            upd = await client.table("wa_conversations").update(
                {"contact_name": contact_name}
            ).eq("id", rows[0]["id"]).execute()
            if upd.data:
                return upd.data[0]
        return rows[0]

    ins = await client.table("wa_conversations").insert(
        {"wa_id": wa_id, "contact_name": contact_name, "channel": "whatsapp"}
    ).execute()
    return ins.data[0]


async def _touch_conversation(
    conversation_id: str,
    *,
    preview: str,
    direction: str,
    created_at: str,
    inc_unread: int = 0,
    reset_unread: bool = False,
) -> None:
    """Update a conversation's rollup fields after a new message."""
    client = _client()
    patch: Dict[str, Any] = {
        "last_message_preview": preview[:_PREVIEW_LEN],
        "last_direction": direction,
        "last_message_at": created_at,
        "updated_at": created_at,
    }
    if reset_unread:
        patch["unread_count"] = 0
    elif inc_unread:
        # Read-modify-write (no live traffic contention expected per conversation).
        cur = await client.table("wa_conversations").select("unread_count").eq(
            "id", conversation_id
        ).limit(1).execute()
        current = (cur.data or [{}])[0].get("unread_count", 0) or 0
        patch["unread_count"] = current + inc_unread
    await client.table("wa_conversations").update(patch).eq("id", conversation_id).execute()


async def record_inbound(
    wa_id: str,
    content: str,
    *,
    message_type: str = "text",
    transcribed: bool = False,
    wa_message_id: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist an inbound (user → agent) message and bump unread."""
    conv = await _ensure_conversation(wa_id, contact_name)
    client = _client()
    ins = await client.table("wa_messages").insert({
        "conversation_id": conv["id"],
        "wa_id": wa_id,
        "direction": "inbound",
        "sender": "user",
        "message_type": message_type,
        "content": content,
        "transcribed": transcribed,
        "wa_message_id": wa_message_id,
    }).execute()
    msg = ins.data[0]
    await _touch_conversation(
        conv["id"], preview=content, direction="inbound",
        created_at=msg["created_at"], inc_unread=1,
    )
    return msg


async def record_outbound(
    wa_id: str,
    content: str,
    *,
    sender: str = "agent",  # "agent" (auto reply) | "admin" (manual)
    wa_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist an outbound (→ user) message. Admin replies reset unread."""
    conv = await _ensure_conversation(wa_id)
    client = _client()
    ins = await client.table("wa_messages").insert({
        "conversation_id": conv["id"],
        "wa_id": wa_id,
        "direction": "outbound",
        "sender": sender,
        "message_type": "text",
        "content": content,
        "wa_message_id": wa_message_id,
    }).execute()
    msg = ins.data[0]
    await _touch_conversation(
        conv["id"], preview=content, direction="outbound",
        created_at=msg["created_at"], reset_unread=(sender == "admin"),
    )
    return msg


async def start_conversation(
    wa_id: str, contact_name: Optional[str] = None
) -> Dict[str, Any]:
    """Create (or return) a conversation for `wa_id` — used to start a new chat."""
    return await _ensure_conversation(wa_id, contact_name)


async def list_conversations(
    limit: int = 50, offset: int = 0, search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List conversations, most recently active first."""
    client = _client()
    q = client.table("wa_conversations").select("*")
    if search:
        # Match wa_id or contact name.
        q = q.or_(f"wa_id.ilike.%{search}%,contact_name.ilike.%{search}%")
    res = await q.order("last_message_at", desc=True).range(offset, offset + limit - 1).execute()
    return res.data or []


async def get_conversation(wa_id: str) -> Optional[Dict[str, Any]]:
    client = _client()
    res = await client.table("wa_conversations").select("*").eq("wa_id", wa_id).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None


async def get_messages(
    wa_id: str, limit: int = 100, after: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return messages for a conversation in chronological order.

    `after` (an ISO timestamp) returns only messages newer than it — used by the
    frontend's polling to fetch just the delta.
    """
    client = _client()
    q = client.table("wa_messages").select("*").eq("wa_id", wa_id)
    if after:
        q = q.gt("created_at", after)
    res = await q.order("created_at", desc=False).limit(limit).execute()
    return res.data or []


async def mark_read(wa_id: str) -> None:
    """Reset a conversation's unread counter (admin opened the thread)."""
    client = _client()
    await client.table("wa_conversations").update({"unread_count": 0}).eq("wa_id", wa_id).execute()
