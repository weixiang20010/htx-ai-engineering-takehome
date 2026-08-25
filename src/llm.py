"""
Shared LLM factory and resilience layer for Parts 1 and 2.

Task-specific model pairs
-------------------------
Extraction tasks (Part 1, Part 2 date extraction) use a lightweight model
as primary so quota is reserved for reasoning.  Reasoning tasks (Part 2 MCP
tool selection, temporal classification) use the stronger model as primary.
Both roles have a cross-fallback so exhausting one model's quota does not
stall the pipeline:

    build_extraction_llm_pair()  →  (GEMINI_EXTRACTION_MODEL, GEMINI_EXTRACTION_FALLBACK_MODEL)
    build_reasoning_llm_pair()   →  (GEMINI_REASONING_MODEL,  GEMINI_REASONING_FALLBACK_MODEL)

Fallback is only triggered for quota / rate-limit errors (429). Validation
failures, bad PDF extraction, or coding bugs propagate normally — falling
back to a second model would not fix them and would hide the root cause.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError

logger = logging.getLogger(__name__)

# Only these errors justify a model switch. Everything else propagates.
_QUOTA_ERRORS: tuple[type[Exception], ...] = (GoogleRateLimitError,)


@dataclass
class ModelUsage:
    """Records which model was actually used for a single LLM invocation."""

    requested_model: str
    actual_model: str
    fallback_used: bool = False
    fallback_reason: str | None = None


def build_llm(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
    """Construct a ChatGoogleGenerativeAI for a given model name."""
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


def _build_pair(
    primary_env: str,
    fallback_env: str,
    primary_default: str,
    role: str,
) -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI | None]:
    """Read env vars and return (primary, fallback) for a named task role."""
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Provide it via the environment variable or a .env file."
        )
    primary_model = os.environ.get(primary_env, primary_default)
    fallback_model = os.environ.get(fallback_env)

    primary = build_llm(primary_model, api_key)
    fallback = build_llm(fallback_model, api_key) if fallback_model else None

    logger.debug("%s pair: primary=%s, fallback=%s", role, primary_model, fallback_model or "none")
    return primary, fallback


def build_extraction_llm_pair() -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI | None]:
    """
    Return the LLM pair for extraction tasks (Part 1, Part 2 date extraction).

    Primary is gemini-3.5-flash-lite (GA, optimised for document parsing and
    structured output). Fallback is gemini-3.1-flash-lite — the previous
    stable lightweight model — keeping the fallback within the same task
    class rather than consuming reasoning-model quota.

    Env vars: GEMINI_EXTRACTION_MODEL, GEMINI_EXTRACTION_FALLBACK_MODEL
    """
    return _build_pair(
        "GEMINI_EXTRACTION_MODEL",
        "GEMINI_EXTRACTION_FALLBACK_MODEL",
        primary_default="gemini-3.5-flash-lite",
        role="extraction",
    )


def build_reasoning_llm_pair() -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI | None]:
    """
    Return the LLM pair for reasoning tasks (Part 2 MCP tool selection, classification).

    Primary is gemini-3.6-flash. Fallback is gemini-3.5-flash-lite — the
    lightweight extraction primary — so reasoning degrades gracefully without
    falling all the way back to the oldest model tier.

    Env vars: GEMINI_REASONING_MODEL, GEMINI_REASONING_FALLBACK_MODEL
    """
    return _build_pair(
        "GEMINI_REASONING_MODEL",
        "GEMINI_REASONING_FALLBACK_MODEL",
        primary_default="gemini-3.6-flash",
        role="reasoning",
    )


def invoke_with_fallback(
    make_chain: Callable[[ChatGoogleGenerativeAI], Runnable],
    inputs: Any,
    primary: ChatGoogleGenerativeAI,
    fallback: ChatGoogleGenerativeAI | None,
) -> tuple[Any, ModelUsage]:
    """
    Build and invoke make_chain(primary). On quota/rate-limit error only,
    rebuild the chain with the fallback model and retry once.

    Parameters
    ----------
    make_chain:
        Callable that accepts an LLM and returns a Runnable chain.
        Called for primary first; called again for fallback only if needed.
    inputs:
        Input passed to chain.invoke().
    primary:
        Primary LLM (used normally).
    fallback:
        Fallback LLM. If None, quota errors propagate unhandled.
    """
    try:
        result = make_chain(primary).invoke(inputs)
        return result, ModelUsage(
            requested_model=primary.model,
            actual_model=primary.model,
        )
    except _QUOTA_ERRORS:
        if fallback is None:
            raise
        logger.warning(
            "Primary model quota exceeded (%s) — retrying with fallback (%s)",
            primary.model,
            fallback.model,
        )
        result = make_chain(fallback).invoke(inputs)
        return result, ModelUsage(
            requested_model=primary.model,
            actual_model=fallback.model,
            fallback_used=True,
            fallback_reason="quota_exceeded",
        )


async def ainvoke_with_fallback(
    make_chain: Callable[[ChatGoogleGenerativeAI], Runnable],
    inputs: Any,
    primary: ChatGoogleGenerativeAI,
    fallback: ChatGoogleGenerativeAI | None,
) -> tuple[Any, ModelUsage]:
    """Async variant of invoke_with_fallback."""
    try:
        result = await make_chain(primary).ainvoke(inputs)
        return result, ModelUsage(
            requested_model=primary.model,
            actual_model=primary.model,
        )
    except _QUOTA_ERRORS:
        if fallback is None:
            raise
        logger.warning(
            "Primary model quota exceeded (%s) — retrying with fallback (%s)",
            primary.model,
            fallback.model,
        )
        result = await make_chain(fallback).ainvoke(inputs)
        return result, ModelUsage(
            requested_model=primary.model,
            actual_model=fallback.model,
            fallback_used=True,
            fallback_reason="quota_exceeded",
        )
