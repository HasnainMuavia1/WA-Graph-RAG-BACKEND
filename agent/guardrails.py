"""
Enterprise guardrails for the university RAG assistant.

Guardrails wrap every agent turn (HTTP chat + WhatsApp) with two checkpoints:

  ┌── INPUT ──────────────────────────────────────────────────────────────┐
  │ • length / empty limits          (DoS + junk protection)              │
  │ • prompt-injection / jailbreak   (e.g. "ignore previous instructions")│
  │ • system-prompt / secret probing ("show me your prompt", api keys)    │
  │ • abuse / hate                    (light keyword screen)              │
  │ • PII capture awareness          (don't echo CNIC / card numbers)     │
  └───────────────────────────────────────────────────────────────────────┘
  ┌── OUTPUT ─────────────────────────────────────────────────────────────┐
  │ • system-prompt leak scrubbing                                        │
  │ • PII redaction (CNIC, card, raw API keys)                            │
  │ • LANGUAGE ENFORCEMENT: replies must be Roman Urdu, never Hindi/      │
  │   Devanagari. Devanagari output is transliterated back via the LLM.   │
  └───────────────────────────────────────────────────────────────────────┘

Design notes
------------
* Pure-Python heuristics (regex) so the hot path stays fast and dependency-free.
* All user-facing block messages are in Roman Urdu (matching the product policy).
* Fails OPEN on internal errors for *output* checks (never drop a good answer
  because a guard crashed) but fails CLOSED for *input* safety checks.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tunables (env-overridable) ────────────────────────────────────────────────
MAX_INPUT_CHARS = int(os.getenv("GUARDRAIL_MAX_INPUT_CHARS", "4000"))
MIN_INPUT_CHARS = int(os.getenv("GUARDRAIL_MIN_INPUT_CHARS", "1"))
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
ENFORCE_ROMAN_URDU = os.getenv("ENFORCE_ROMAN_URDU", "true").lower() == "true"

# Polite Roman-Urdu refusals.
_MSG_TOO_LONG = (
    "Aap ka message bohat lamba hai. Baraye meharbani thora mukhtasar likhein."
)
_MSG_EMPTY = "Mujhe aap ka sawal samajh nahi aaya. Baraye meharbani dobara likhein."
_MSG_INJECTION = (
    "Maazrat — main sirf university (admissions, programs, fees, scholarships waghaira) "
    "se mutaliq madad kar sakta hoon. Apna university se mutaliq sawal poochein."
)
_MSG_ABUSE = "Baraye meharbani mohzbana andaaz mein baat karein taake main aap ki behtar madad kar sakoon."

# ── Prompt-injection / jailbreak signatures ───────────────────────────────────
_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts?|rules?)",
    r"\bdisregard\s+(the\s+)?(previous|above|system)\b",
    r"\b(reveal|show|print|repeat|tell)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
    r"\bwhat\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?)",
    r"\byou\s+are\s+now\b.*\b(dan|developer mode|jailbreak|unrestricted)\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bact\s+as\s+(an?\s+)?(unrestricted|uncensored|evil)\b",
    r"\bpretend\s+(you\s+are|to\s+be)\b.*\b(no\s+rules|without\s+restrictions)\b",
    r"\b(bypass|override|disable)\s+(your\s+)?(safety|guardrails?|filters?|restrictions?)",
    r"\bprint\s+the\s+(api|secret|access)\s*key",
    r"\b(system\s*prompt|initial\s+instructions)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# ── Abuse / hate (light screen — not a substitute for a full classifier) ──────
_ABUSE_PATTERNS = [
    r"\bfuck\b",
    r"\bshit\b",
    r"\bbitch\b",
    r"\basshole\b",
    r"\bharaam\s*zada\b",
    r"\bkutt?a\b",
    r"\bgandu\b",
    r"\bmadarchod\b",
    r"\bbehnchod\b",
]
_ABUSE_RE = re.compile("|".join(_ABUSE_PATTERNS), re.IGNORECASE)

# ── PII patterns (output redaction) ───────────────────────────────────────────
# Pakistani CNIC (#####-#######-#), generic 13-digit, long card numbers, API keys.
_PII_PATTERNS = [
    (re.compile(r"\b\d{5}-\d{7}-\d\b"), "[REDACTED-CNIC]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED-CARD]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED-KEY]"),
    (
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
        ),
        "[REDACTED-TOKEN]",
    ),
]

# ── System-prompt leak signatures (output scrubbing) ──────────────────────────
_LEAK_PATTERNS = re.compile(
    r"(you are 'uchenab assistant'|_system_prompt|system prompt|my instructions are)",
    re.IGNORECASE,
)

# Devanagari (Hindi) Unicode block — Roman Urdu must never contain these.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


@dataclass
class GuardrailResult:
    """Outcome of an input check. `allowed=False` → return `user_message`."""

    allowed: bool
    reason: Optional[str] = None
    user_message: Optional[str] = None
    sanitized_input: Optional[str] = None


# ── INPUT guardrails ──────────────────────────────────────────────────────────


def check_input(text: str) -> GuardrailResult:
    """Validate and screen an inbound user message. Fails CLOSED on safety hits."""
    if not GUARDRAILS_ENABLED:
        return GuardrailResult(allowed=True, sanitized_input=text)

    raw = text or ""
    stripped = raw.strip()

    if len(stripped) < MIN_INPUT_CHARS:
        return GuardrailResult(False, reason="empty", user_message=_MSG_EMPTY)

    if len(stripped) > MAX_INPUT_CHARS:
        return GuardrailResult(False, reason="too_long", user_message=_MSG_TOO_LONG)

    if _INJECTION_RE.search(stripped):
        logger.warning("Guardrail: prompt-injection attempt blocked")
        return GuardrailResult(
            False, reason="prompt_injection", user_message=_MSG_INJECTION
        )

    if _ABUSE_RE.search(stripped):
        logger.info("Guardrail: abusive language blocked")
        return GuardrailResult(False, reason="abuse", user_message=_MSG_ABUSE)

    # Defensive: neutralize role-play markers a model might otherwise honor.
    sanitized = re.sub(r"(?i)\b(system|assistant|developer)\s*:", "", stripped)

    return GuardrailResult(allowed=True, sanitized_input=sanitized)


# ── OUTPUT guardrails ─────────────────────────────────────────────────────────


def redact_pii(text: str) -> str:
    """Redact obvious PII / secrets from any outbound text."""
    out = text
    for pattern, replacement in _PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def scrub_leaks(text: str) -> str:
    """Remove accidental system-prompt / internal-config leakage lines."""
    if not _LEAK_PATTERNS.search(text):
        return text
    safe_lines = [ln for ln in text.splitlines() if not _LEAK_PATTERNS.search(ln)]
    cleaned = "\n".join(safe_lines).strip()
    return cleaned or "Maazrat, main is waqt is sawal ka jawab nahi de sakta."


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


async def enforce_roman_urdu(text: str) -> str:
    """If output contains Hindi/Devanagari, transliterate it to Roman Urdu via the LLM.

    Fails open: on any error the original text is returned unchanged.
    """
    if not (ENFORCE_ROMAN_URDU and contains_devanagari(text)):
        return text
    try:
        from .providers import get_cached_llm, LLM_CHOICE

        client = get_cached_llm()
        resp = await client.chat.completions.create(
            model=LLM_CHOICE,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user's text into Roman Urdu (Urdu written in Latin/English "
                        "letters). Keep English words/proper nouns as-is. Output ONLY the converted "
                        "text, no extra commentary. Never use Hindi or Devanagari script."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        converted = (resp.choices[0].message.content or "").strip()
        if converted and not contains_devanagari(converted):
            logger.info("Guardrail: transliterated Devanagari output → Roman Urdu")
            return converted
    except Exception as exc:  # fail open — never lose a good answer
        logger.warning("Roman-Urdu enforcement failed (returning original): %s", exc)
    return text


async def apply_output_guardrails(text: str) -> str:
    """Full outbound pipeline: leak scrub → PII redaction → Roman-Urdu enforcement."""
    if not GUARDRAILS_ENABLED:
        return text
    try:
        out = scrub_leaks(text)
        out = redact_pii(out)
        out = await enforce_roman_urdu(out)
        return out
    except Exception as exc:  # fail open on the output path
        logger.warning("Output guardrail error (returning original): %s", exc)
        return text


async def classify_query_scope(query: str) -> str:
    """Classify if the user's query is within the university's scope or completely out-of-scope.

    If the query is out of scope (e.g. asking about Python, Google, general trivia),
    returns 'out_of_scope'. Otherwise returns 'in_scope'.
    """
    try:
        from .providers import get_cached_llm, LLM_CHOICE

        client = get_cached_llm()
        resp = await client.chat.completions.create(
            model=LLM_CHOICE,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI scope classifier for Uchenab University helpdesk.\n"
                        "Classify if the user's message is asking about university-related matters "
                        "(e.g., admissions, courses, eligibility, fees, campus, hostel, policies) or if it's "
                        "completely out-of-scope (e.g., programming languages like Python/Java, tech companies like Google/Apple, "
                        "general trivia, writing code, politics, recipes, or unrelated chat).\n"
                        "Respond with ONLY one word: 'IN_SCOPE' or 'OUT_OF_SCOPE'."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=10,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        if "OUT_OF_SCOPE" in verdict:
            logger.info("Guardrail scope classifier: OUT_OF_SCOPE query '%s'", query)
            return "out_of_scope"
        return "in_scope"
    except Exception as exc:
        logger.error("Failed to classify scope (defaulting to in_scope): %s", exc)
        return "in_scope"
