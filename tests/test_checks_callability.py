"""Tier 2 checks use the vendored route classifier against every mock route."""

from __future__ import annotations

import asyncio

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.callability import run_callability_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult


async def _run(
    server: MockOpenAIServer,
    mode: str,
    models: list[str],
    *,
    declared_chat_models: bool = False,
) -> CheckResult:
    async def no_delay(_: float) -> None:
        return None

    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
        sleep=no_delay,
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results = await run_callability_checks(
                client,
                models,
                delay_seconds=0,
                sleep=no_delay,
                declared_chat_models=declared_chat_models,
            )
    assert len(results) == 1
    assert results[0].id == "callability.advertised-models"
    return results[0]


@pytest.mark.asyncio
async def test_callability_conforming_control_is_green(
    mock_server: MockOpenAIServer,
) -> None:
    result = await _run(mock_server, "conforming", ["mock/model"])
    assert result.status == "pass"
    assert result.measured["dead_count"] == 0
    assert result.measured["flaky_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "model", "http_status"),
    [
        ("catalog_lists_403_model", "mock/catalog-forbidden", 403),
        ("catalog_lists_404_model", "mock/catalog-missing", 404),
    ],
)
async def test_callability_rejects_dead_routes_without_rejecting_live_routes(
    mock_server: MockOpenAIServer,
    mode: str,
    model: str,
    http_status: int,
) -> None:
    rejected = await _run(mock_server, mode, [model], declared_chat_models=True)
    undeclared = await _run(mock_server, mode, [model])
    accepted = await _run(mock_server, "conforming", ["mock/model"])

    # A declared chat model that will not serve chat is a broken promise.
    assert rejected.status == "fail"
    # Without a declaration the tool cannot know an advertised id is meant to
    # serve chat -- OpenAI's /models lists embeddings and speech endpoints that
    # 404 here by design -- so the same evidence is only advisory.
    assert undeclared.status == "warn"
    assert undeclared.measured["dead_count"] == 1
    assert rejected.measured["dead_count"] == 1
    model_rows = rejected.measured["models"]
    assert isinstance(model_rows, list) and len(model_rows) == 1
    assert model_rows[0]["http_status"] == http_status
    assert accepted.status == "pass"


@pytest.mark.asyncio
async def test_callability_marks_transient_capacity_flaky_not_dead(
    mock_server: MockOpenAIServer,
) -> None:
    flaky = await _run(mock_server, "queue_then_429", ["mock/model"])
    healthy = await _run(mock_server, "conforming", ["mock/model"])

    assert flaky.status == "warn"
    assert flaky.measured["dead_count"] == 0
    assert flaky.measured["flaky_count"] == 1
    assert healthy.status == "pass"


@pytest.mark.asyncio
async def test_callability_skips_explicitly_when_discovery_is_empty(
    mock_server: MockOpenAIServer,
) -> None:
    result = await _run(mock_server, "conforming", [])
    assert result.status == "skip"
    assert result.measured["reason"]
