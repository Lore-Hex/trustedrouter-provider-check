"""Tier 3 checks isolate every non-streaming response and request defect."""

from __future__ import annotations

import asyncio

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.chat import run_chat_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult

_CHAT_IDS = {
    "chat.non-empty",
    "chat.pong",
    "chat.usage",
    "chat.model",
    "chat.finish-reason",
    "chat.temperature-zero",
    "chat.forwarded-fields",
    "chat.max-token-spelling",
}


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    model: str = "mock/model",
    provider: str | None = None,
) -> list[CheckResult]:
    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        provider=provider,
        headers={"X-Mock-Mode": mode},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results = await run_chat_checks(client, model)
    assert len(results) == len(_CHAT_IDS)
    assert {result.id for result in results} == _CHAT_IDS
    return results


def _statuses(results: list[CheckResult]) -> dict[str, str]:
    rows: dict[str, str] = {result.id: result.status for result in results}
    assert len(rows) == len(_CHAT_IDS)
    return rows


def _assert_only(
    results: list[CheckResult],
    check_id: str,
    status: str,
    *,
    allowed_skip: str | None = None,
) -> None:
    rows = _statuses(results)
    assert rows[check_id] == status
    expected = {name: "pass" for name in _CHAT_IDS}
    expected[check_id] = status
    if allowed_skip is not None:
        expected[allowed_skip] = "skip"
    assert rows == expected


@pytest.mark.asyncio
async def test_chat_conforming_control_is_green(mock_server: MockOpenAIServer) -> None:
    results = await _run(mock_server, "conforming")
    assert _statuses(results) == {check_id: "pass" for check_id in _CHAT_IDS}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "check_id", "status", "allowed_skip"),
    [
        ("empty_content", "chat.non-empty", "fail", "chat.pong"),
        ("wrong_pong", "chat.pong", "fail", None),
        ("bad_usage", "chat.usage", "fail", None),
        ("wrong_model", "chat.model", "warn", None),
        ("unknown_finish_reason", "chat.finish-reason", "warn", None),
        ("rejects_temperature_zero", "chat.temperature-zero", "fail", None),
        # SHOULD-tolerate, so advisory: OpenAI itself rejects top_k/thinking,
        # and failing a provider for matching the reference implementation
        # would make the verdict useless.
        ("strict_extra_fields", "chat.forwarded-fields", "warn", None),
        ("rejects_max_tokens", "chat.max-token-spelling", "fail", None),
    ],
)
async def test_chat_mode_isolates_its_named_check_and_has_green_control(
    mock_server: MockOpenAIServer,
    mode: str,
    check_id: str,
    status: str,
    allowed_skip: str | None,
) -> None:
    isolated = await _run(mock_server, mode)
    control = await _run(mock_server, "conforming")

    _assert_only(isolated, check_id, status, allowed_skip=allowed_skip)
    assert _statuses(control) == {name: "pass" for name in _CHAT_IDS}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "content-parts",
        "content-parts-content",
        "reasoning-content-string",
        "reasoning-content-parts",
        "reasoning-string",
        "reasoning-parts",
    ],
)
async def test_chat_accepts_every_vendored_content_and_reasoning_shape(
    mock_server: MockOpenAIServer,
    shape: str,
) -> None:
    results = await _run(
        mock_server,
        "chat_reasoning_shapes",
        model=f"mock/{shape}",
    )
    assert _statuses(results) == {name: "pass" for name in _CHAT_IDS}


@pytest.mark.asyncio
async def test_chat_uses_max_completion_tokens_for_openai_reasoning_models(
    mock_server: MockOpenAIServer,
) -> None:
    results = await _run(
        mock_server,
        "rejects_max_tokens",
        model="openai/gpt-5.1",
        provider="openai",
    )
    rows = _statuses(results)
    assert rows["chat.max-token-spelling"] == "pass"
    assert rows["chat.temperature-zero"] == "skip"
    assert sum(status == "skip" for status in rows.values()) == 1
    assert sum(status == "pass" for status in rows.values()) == len(_CHAT_IDS) - 1
    bodies = [record.body for record in mock_server.request_log if record.body]
    cap_bodies = [body for body in bodies if "max_completion_tokens" in body]
    assert len(cap_bodies) == 1
    assert "max_tokens" not in cap_bodies[0]


@pytest.mark.asyncio
async def test_every_tier_three_429_is_inconclusive(
    mock_server: MockOpenAIServer,
) -> None:
    statuses = _statuses(await _run(mock_server, "queue_then_429"))

    assert "fail" not in statuses.values()
    assert statuses["chat.pong"] == "warn"
    assert statuses["chat.temperature-zero"] == "warn"
    assert statuses["chat.forwarded-fields"] == "warn"
    assert statuses["chat.max-token-spelling"] == "warn"
    assert statuses["chat.non-empty"] == "skip"
