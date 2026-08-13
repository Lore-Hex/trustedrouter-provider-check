"""Tier 5 structured checks separate field rejection from broken output."""

from __future__ import annotations

import asyncio

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.structured import run_structured_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    declared: bool | None = True,
) -> dict[str, CheckResult]:
    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results = await run_structured_checks(
                client,
                "mock/model",
                declared=declared,
            )
    assert len(results) == 2
    rows = {result.id: result for result in results}
    assert len(rows) == 2
    assert set(rows) == {"structured.json-object", "structured.json-schema"}
    return rows


@pytest.mark.asyncio
async def test_structured_control_parses_and_validates(
    mock_server: MockOpenAIServer,
) -> None:
    rows = await _run(mock_server, "conforming")

    assert rows["structured.json-object"].status == "pass"
    assert rows["structured.json-object"].measured["body_parsed_as_json"] is True
    assert rows["structured.json-schema"].status == "pass"
    assert rows["structured.json-schema"].measured["schema_valid"] is True


@pytest.mark.asyncio
async def test_declared_response_format_rejection_warns_configuration_with_control(
    mock_server: MockOpenAIServer,
) -> None:
    rejected = await _run(mock_server, "rejects_response_format", declared=True)
    mock_server.clear_requests()
    control = await _run(mock_server, "conforming", declared=True)

    for check_id in ("structured.json-object", "structured.json-schema"):
        assert rejected[check_id].status == "warn"
        assert rejected[check_id].owner == "configuration"
        assert rejected[check_id].measured["http_status"] == 400
        assert control[check_id].status == "pass"


@pytest.mark.asyncio
async def test_undeclared_response_format_rejection_skips_not_fails(
    mock_server: MockOpenAIServer,
) -> None:
    rejected = await _run(mock_server, "rejects_response_format", declared=None)
    mock_server.clear_requests()
    discovered = await _run(mock_server, "conforming", declared=None)

    assert rejected["structured.json-object"].status == "skip"
    assert rejected["structured.json-schema"].status == "skip"
    assert discovered["structured.json-object"].status == "pass"
    assert discovered["structured.json-schema"].status == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "failed_id", "passing_id"),
    [
        (
            "structured_non_json",
            "structured.json-object",
            "structured.json-schema",
        ),
        (
            "structured_schema_violation",
            "structured.json-schema",
            "structured.json-object",
        ),
    ],
)
async def test_accepted_nonconforming_structured_output_fails_provider_only(
    mock_server: MockOpenAIServer,
    mode: str,
    failed_id: str,
    passing_id: str,
) -> None:
    broken = await _run(mock_server, mode)
    mock_server.clear_requests()
    control = await _run(mock_server, "conforming")

    assert broken[failed_id].status == "fail"
    assert broken[failed_id].owner == "provider"
    assert broken[passing_id].status == "pass"
    assert control[failed_id].status == "pass"
    assert control[passing_id].status == "pass"


@pytest.mark.asyncio
async def test_declared_false_structured_output_skips_without_requests(
    mock_server: MockOpenAIServer,
) -> None:
    rows = await _run(mock_server, "conforming", declared=False)

    assert rows["structured.json-object"].status == "skip"
    assert rows["structured.json-schema"].status == "skip"
    assert mock_server.request_log == []
