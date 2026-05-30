"""
LLM and Embedding Model Providers.
Configures and provides access to language and embedding models.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-4o-mini")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Single API key — OPENAI_API_KEY is canonical; legacy aliases still accepted
_OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("EMBEDDING_API_KEY")
    or ""
)

INGESTION_LLM_CHOICE = os.getenv("INGESTION_LLM_CHOICE", LLM_CHOICE)


def get_embedding_model() -> str:
    """Return the embedding model name string."""
    return _EMBEDDING_MODEL_NAME


def get_ingestion_model() -> str:
    """Return the Pydantic AI model string for the ingestion LLM."""
    if LLM_PROVIDER.lower() in ("openai", "openroute"):
        return f"openai:{INGESTION_LLM_CHOICE}"
    return f"openai:{INGESTION_LLM_CHOICE}"


def get_embedding_client():
    """Return an AsyncOpenAI client configured for embedding requests."""
    from openai import AsyncOpenAI

    kwargs: dict = {"api_key": _OPENAI_API_KEY}
    if EMBEDDING_PROVIDER.lower() == "openroute":
        kwargs["base_url"] = EMBEDDING_BASE_URL
    return AsyncOpenAI(**kwargs)


def get_llm_client():
    """Return an AsyncOpenAI client configured for LLM requests."""
    from openai import AsyncOpenAI

    kwargs: dict = {"api_key": _OPENAI_API_KEY}
    if LLM_PROVIDER.lower() == "openroute":
        kwargs["base_url"] = LLM_BASE_URL
    return AsyncOpenAI(**kwargs)


def get_llm_model(model_name: Optional[str] = None):
    """Return a synchronous OpenAI client (kept for backward compatibility)."""
    from openai import OpenAI

    kwargs: dict = {"api_key": _OPENAI_API_KEY}
    if LLM_PROVIDER.lower() == "openroute":
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


_llm_client: Optional[object] = None
_embedding_client: Optional[object] = None


def get_cached_llm():
    """Return the cached async LLM client, initialising if needed."""
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def get_cached_embedding():
    """Return the cached async embedding client, initialising if needed."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_embedding_client()
    return _embedding_client
