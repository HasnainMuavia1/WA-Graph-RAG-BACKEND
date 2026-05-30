"""
Deepgram transcription client.

WhatsApp voice notes arrive as OGG/Opus audio. We send the raw bytes to
Deepgram's pre-recorded transcription API and get text back, which then flows
into the RAG agent exactly like a typed message.

Uses the official `deepgram-sdk` (v3) async client. Falls back to a clear,
typed error if the key is missing so callers can reply gracefully to the user.

Env vars:
    DEEPGRAM_API_KEY   Your Deepgram API key
    DEEPGRAM_MODEL     Model name (default: nova-2)
    DEEPGRAM_LANGUAGE  Language hint (default: en; use "multi" for auto)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be performed or returns nothing."""


class DeepgramTranscriber:
    """Async wrapper over Deepgram pre-recorded transcription."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.model = model or os.getenv("DEEPGRAM_MODEL", "nova-2")
        self.language = language or os.getenv("DEEPGRAM_LANGUAGE", "en")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, audio: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio bytes to text. Raises TranscriptionError on failure.

        Targets the Deepgram SDK v7 async client:
            client.listen.v1.media.transcribe_file(request=<bytes>, ...)
        """
        if not self.is_configured:
            raise TranscriptionError("DEEPGRAM_API_KEY is not configured.")
        if not audio:
            raise TranscriptionError("Empty audio payload.")

        try:
            from deepgram import AsyncDeepgramClient
        except ImportError as exc:  # pragma: no cover
            raise TranscriptionError(
                "deepgram-sdk is not installed. Add it to requirements."
            ) from exc

        client = AsyncDeepgramClient(api_key=self.api_key)
        try:
            response = await client.listen.v1.media.transcribe_file(
                request=audio,
                model=self.model,
                language=self.language,
                smart_format=True,   # punctuation + capitalization
                punctuate=True,
            )
        except Exception as exc:
            raise TranscriptionError(f"Deepgram request failed: {exc}") from exc

        try:
            transcript = (
                response.results.channels[0].alternatives[0].transcript or ""
            ).strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise TranscriptionError(f"Unexpected Deepgram response shape: {exc}") from exc

        if not transcript:
            raise TranscriptionError("Transcription returned empty text.")
        logger.info("Transcribed %d bytes of audio → %d chars", len(audio), len(transcript))
        return transcript


# ── Module-level singleton ────────────────────────────────────────────────────
transcriber = DeepgramTranscriber()
