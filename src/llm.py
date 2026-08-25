"""
Shared LLM factory and resilience layer for Parts 1 and 2.

Both parts build their LLMs via build_llm_pair() and call the LLM through
invoke_with_fallback / ainvoke_with_fallback rather than directly. This
centralises the fallback logic so it does not leak into extraction code.

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


def build_llm_pair() -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI | None]:
    """
    Read GEMINI_PRIMARY_MODEL and GEMINI_FALLBACK_MODEL from the environment.

    Returns (primary_llm, fallback_llm).  fallback_llm is None when
    GEMINI_FALLBACK_MODEL is not set, in which case quota errors propagate.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Provide it via the environment variable or a .env file."
        )
    primary_model = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash")
    fallback_model = os.environ.get("GEMINI_FALLBACK_MODEL")

    primary = build_llm(primary_model, api_key)
    fallback = build_llm(fallback_model, api_key) if fallback_model else None

    logger.debug("LLM pair: primary=%s, fallback=%s", primary_model, fallback_model or "none")
    return primary, fallback


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
