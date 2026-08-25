"""
Unit tests for the LLM resilience layer in src/llm.py.

All tests are mock-based — zero API quota consumed. They verify that:
- Primary succeeds → fallback is never called.
- Primary raises a quota error → fallback is called and result returned.
- Primary raises a non-quota error → propagates immediately, fallback not called.
- No fallback is configured + primary quota error → error propagates.
- Async variant mirrors the sync contract.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_google_genai.chat_models import GoogleRateLimitError

from src.llm import ModelUsage, ainvoke_with_fallback, invoke_with_fallback


def _make_llm(model_name: str) -> MagicMock:
    """Return a mock that looks like a ChatGoogleGenerativeAI instance."""
    llm = MagicMock()
    llm.model = model_name
    return llm


def _chain_returning(value: str) -> MagicMock:
    """Return a mock chain whose .invoke() returns *value*."""
    chain = MagicMock()
    chain.invoke.return_value = value
    return chain


def _chain_raising(exc: Exception) -> MagicMock:
    """Return a mock chain whose .invoke() raises *exc*."""
    chain = MagicMock()
    chain.invoke.side_effect = exc
    return chain


class TestInvokeWithFallback:
    def test_primary_succeeds_fallback_not_called(self) -> None:
        """When the primary chain succeeds, the fallback factory is never invoked."""
        primary = _make_llm("primary-model")
        fallback = _make_llm("fallback-model")

        call_log: list[str] = []

        def make_chain(llm: MagicMock) -> MagicMock:
            call_log.append(llm.model)
            return _chain_returning("ok")

        result, usage = invoke_with_fallback(make_chain, {}, primary, fallback)

        assert result == "ok"
        assert usage == ModelUsage(
            requested_model="primary-model",
            actual_model="primary-model",
            fallback_used=False,
            fallback_reason=None,
        )
        assert call_log == ["primary-model"], "Fallback must not be called on success"

    def test_primary_quota_error_calls_fallback(self) -> None:
        """A quota error on the primary triggers exactly one fallback invocation."""
        primary = _make_llm("primary-model")
        fallback = _make_llm("fallback-model")

        calls: list[str] = []

        def make_chain(llm: MagicMock) -> MagicMock:
            calls.append(llm.model)
            if llm.model == "primary-model":
                return _chain_raising(GoogleRateLimitError("quota"))
            return _chain_returning("fallback-ok")

        result, usage = invoke_with_fallback(make_chain, {}, primary, fallback)

        assert result == "fallback-ok"
        assert usage == ModelUsage(
            requested_model="primary-model",
            actual_model="fallback-model",
            fallback_used=True,
            fallback_reason="quota_exceeded",
        )
        assert calls == ["primary-model", "fallback-model"]

    def test_primary_non_quota_error_propagates_no_fallback(self) -> None:
        """A non-quota error (e.g. ValueError) must propagate without touching fallback."""
        primary = _make_llm("primary-model")
        fallback = _make_llm("fallback-model")

        calls: list[str] = []

        def make_chain(llm: MagicMock) -> MagicMock:
            calls.append(llm.model)
            return _chain_raising(ValueError("bad schema"))

        with pytest.raises(ValueError, match="bad schema"):
            invoke_with_fallback(make_chain, {}, primary, fallback)

        assert calls == ["primary-model"], "Fallback must not be called for non-quota errors"

    def test_primary_quota_error_no_fallback_propagates(self) -> None:
        """When no fallback is configured a quota error propagates unhandled."""
        primary = _make_llm("primary-model")

        def make_chain(llm: MagicMock) -> MagicMock:
            return _chain_raising(GoogleRateLimitError("quota"))

        with pytest.raises(GoogleRateLimitError):
            invoke_with_fallback(make_chain, {}, primary, None)


class TestAInvokeWithFallback:
    async def test_primary_succeeds_fallback_not_called(self) -> None:
        primary = _make_llm("primary-model")
        fallback = _make_llm("fallback-model")

        calls: list[str] = []

        def make_chain(llm: MagicMock) -> MagicMock:
            calls.append(llm.model)
            chain = MagicMock()
            chain.ainvoke = AsyncMock(return_value="async-ok")
            return chain

        result, usage = await ainvoke_with_fallback(make_chain, {}, primary, fallback)

        assert result == "async-ok"
        assert usage.actual_model == "primary-model"
        assert not usage.fallback_used
        assert calls == ["primary-model"]

    async def test_primary_quota_error_calls_fallback(self) -> None:
        primary = _make_llm("primary-model")
        fallback = _make_llm("fallback-model")

        calls: list[str] = []

        def make_chain(llm: MagicMock) -> MagicMock:
            calls.append(llm.model)
            chain = MagicMock()
            if llm.model == "primary-model":
                chain.ainvoke = AsyncMock(side_effect=GoogleRateLimitError("quota"))
            else:
                chain.ainvoke = AsyncMock(return_value="async-fallback-ok")
            return chain

        result, usage = await ainvoke_with_fallback(make_chain, {}, primary, fallback)

        assert result == "async-fallback-ok"
        assert usage == ModelUsage(
            requested_model="primary-model",
            actual_model="fallback-model",
            fallback_used=True,
            fallback_reason="quota_exceeded",
        )
        assert calls == ["primary-model", "fallback-model"]
