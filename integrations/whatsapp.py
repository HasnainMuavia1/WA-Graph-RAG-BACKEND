"""
WhatsApp Cloud API client (Meta Graph API).

Handles the three things the bot needs:

* **Send** outbound text replies (`messages` endpoint).
* **Fetch + download** inbound media (voice notes, images, documents) — a
  two-step flow: media-id → temporary CDN url → bytes.
* **Verify** that inbound webhooks genuinely came from Meta (HMAC-SHA256 of the
  raw body against the App Secret) and answer the GET verification handshake.

All network calls use httpx.AsyncClient so they compose with our async stack.

Env vars (see README → "WhatsApp Cloud API setup"):
    WHATSAPP_ACCESS_TOKEN     Permanent/system-user access token
    WHATSAPP_PHONE_NUMBER_ID  The phone-number id (NOT the phone number)
    WHATSAPP_APP_SECRET       App secret — used to verify webhook signatures
    WHATSAPP_VERIFY_TOKEN     Arbitrary string you also paste into Meta's UI
    WHATSAPP_API_VERSION      Graph API version (default v21.0)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.facebook.com"


class WhatsAppConfigError(RuntimeError):
    """Raised when required WhatsApp env vars are missing."""


class WhatsAppClient:
    """Thin async wrapper over the WhatsApp Cloud API."""

    def __init__(
        self,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        app_secret: str | None = None,
        verify_token: str | None = None,
        api_version: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.access_token = access_token or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        self.phone_number_id = phone_number_id or os.getenv(
            "WHATSAPP_PHONE_NUMBER_ID", ""
        )
        self.app_secret = app_secret or os.getenv("WHATSAPP_APP_SECRET", "")
        self.verify_token = verify_token or os.getenv("WHATSAPP_VERIFY_TOKEN", "")
        self.api_version = api_version or os.getenv("WHATSAPP_API_VERSION", "v21.0")
        self.timeout = timeout

    # ── Config / configured checks ────────────────────────────────────────────
    @property
    def is_configured(self) -> bool:
        return bool(self.access_token and self.phone_number_id)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise WhatsAppConfigError(
                "WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID must be set."
            )

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    # ── Webhook verification (GET handshake) ──────────────────────────────────
    def verify_subscription(
        self, mode: str | None, token: str | None, challenge: str | None
    ) -> str | None:
        """Return the challenge string if Meta's GET handshake is valid, else None."""
        if mode == "subscribe" and token and token == self.verify_token:
            return challenge
        return None

    # ── Webhook signature (POST payload authenticity) ─────────────────────────
    def verify_signature(self, payload: bytes, signature_header: str | None) -> bool:
        """Validate the `X-Hub-Signature-256` header against the raw request body.

        If no App Secret is configured we skip verification (dev mode) and warn.
        """
        if not self.app_secret:
            logger.warning(
                "WHATSAPP_APP_SECRET not set — skipping signature verification"
            )
            return True
        if not signature_header or not signature_header.startswith("sha256="):
            return False
        expected = hmac.new(
            self.app_secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        provided = signature_header.split("sha256=", 1)[1]
        return hmac.compare_digest(expected, provided)

    # ── Outbound: send a text message ─────────────────────────────────────────
    async def send_text(self, to: str, body: str, preview_url: bool = False) -> dict:
        """Send a plain-text WhatsApp message to `to` (E.164 number, no '+')."""
        self._require_configured()
        # WhatsApp hard-limits text bodies to 4096 chars.
        body = body[:4096]
        url = f"{_GRAPH_BASE}/{self.api_version}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": preview_url, "body": body},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._auth_headers, json=payload)
            if resp.status_code >= 400:
                logger.error(
                    "WhatsApp send_text failed (%s): %s", resp.status_code, resp.text
                )
            resp.raise_for_status()
            return resp.json()

    # ── Mark an inbound message as read (blue ticks) ──────────────────────────
    async def mark_read(self, message_id: str) -> None:
        self._require_configured()
        url = f"{_GRAPH_BASE}/{self.api_version}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.post(url, headers=self._auth_headers, json=payload)
        except Exception as exc:  # non-critical — never fail the pipeline on this
            logger.debug("mark_read failed for %s: %s", message_id, exc)

    # ── Inbound media: id → url → bytes ───────────────────────────────────────
    async def get_media_url(self, media_id: str) -> tuple[str, str]:
        """Resolve a media id to its temporary CDN url. Returns (url, mime_type)."""
        self._require_configured()
        url = f"{_GRAPH_BASE}/{self.api_version}/{media_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self._auth_headers)
            resp.raise_for_status()
            data = resp.json()
            return data["url"], data.get("mime_type", "application/octet-stream")

    async def download_media(self, media_url: str) -> bytes:
        """Download media bytes from the CDN url (requires the auth header too)."""
        self._require_configured()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(media_url, headers=self._auth_headers)
            resp.raise_for_status()
            return resp.content

    async def fetch_media_bytes(self, media_id: str) -> tuple[bytes, str]:
        """Convenience: media id → (bytes, mime_type) in one call."""
        media_url, mime_type = await self.get_media_url(media_id)
        return await self.download_media(media_url), mime_type


# ── Module-level singleton ────────────────────────────────────────────────────
whatsapp_client = WhatsAppClient()
