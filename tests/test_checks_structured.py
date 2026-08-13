"""Tier 5 structured checks separate field rejection from broken output."""

from __future__ import annotations

import asyncio

import httpx
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
async def test_declared_response_format_rejection_fails_configuration_with_control(
    mock_server: MockOpenAIServer,
) -> None:
    rejected = await _run(mock_server, "rejects_response_format", declared=True)
    mock_server.clear_requests()
    control = await _run(mock_server, "conforming", declared=True)

    for check_id in ("structured.json-object", "structured.json-schema"):
        assert rejected[check_id].status == "fail"
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


@pytest.mark.asyncio
async def test_backend_outage_during_probe_is_inconclusive_not_a_rejection(
    mock_server: MockOpenAIServer,
) -> None:
    # A live endpoint with intermittent 502s was told response_format was
    # "rejected" and that it rejected temperature: 0 -- a confident diagnosis
    # the evidence never supported. A 5xx means the request was never read.
    probes = await _run(mock_server, "capability_probe_backend_down")

    structured = probes["structured.json-object"]
    assert structured.status == "warn"
    assert structured.measured["inconclusive"] is True
    assert "unknown" in structured.measured["reason"]

    # Negative control: a real 4xx refusal must still read as a refusal.
    refused = (await _run(mock_server, "rejects_response_format"))[
        "structured.json-object"
    ]
    assert refused.measured["inconclusive"] is False
    assert "rejected" in refused.measured["reason"]


@pytest.mark.asyncio
async def test_reasoning_prose_is_not_parsed_as_the_structured_answer(
    mock_server: MockOpenAIServer,
) -> None:
    # Two live endpoint shapes returned exact JSON in
    # content while reasoning_content held prose. Parsing the reasoning
    # fallback reported both conformant providers as emitting malformed JSON.
    probes = await _run(mock_server, "structured_json_in_content_prose_in_reasoning")

    assert probes["structured.json-object"].status == "pass"
    assert probes["structured.json-object"].measured["body_parsed_as_json"] is True

    # Negative control: genuinely malformed content still fails, so dropping
    # the reasoning channel did not make the check unfalsifiable.
    broken = await _run(mock_server, "structured_non_json")
    assert broken["structured.json-object"].status == "fail"


@pytest.mark.asyncio
async def test_undeclared_prose_does_not_create_or_fail_a_capability(
    mock_server: MockOpenAIServer,
) -> None:
    rows = await _run(mock_server, "structured_non_json", declared=None)
    result = rows["structured.json-object"]

    assert result.status == "skip"
    assert result.measured["capability_discovered"] is False
    assert result.measured["body_parsed_as_json"] is False


@pytest.mark.asyncio
async def test_schema_valid_is_unknown_when_content_never_parses(
    mock_server: MockOpenAIServer,
) -> None:
    rows = await _run(mock_server, "structured_non_json", declared=True)
    # Exercise the schema check with non-JSON too by making both formats prose.
    object_result = rows["structured.json-object"]
    assert object_result.measured["body_parsed_as_json"] is False

    async with GatewayClient(
        f"{mock_server.base_url}/v1",
        "test-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"message": {"content": "not json"}}],
                },
            )
        ),
    ) as client:
        schema_rows = await run_structured_checks(client, "mock/model", declared=True)
    schema_result = next(
        row for row in schema_rows if row.id == "structured.json-schema"
    )
    assert schema_result.measured["body_parsed_as_json"] is False
    assert schema_result.measured["schema_valid"] is None


@pytest.mark.asyncio
async def test_structured_transport_error_preserves_its_real_type() -> None:
    async def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with GatewayClient(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(fail),
    ) as client:
        rows = await run_structured_checks(client, "mock/model", declared=True)

    assert {row.status for row in rows} == {"warn"}
    assert {row.measured["error"] for row in rows} == {"ConnectError"}


@pytest.mark.asyncio
async def test_json_schema_probe_does_not_grade_prompt_wording(
    mock_server: MockOpenAIServer,
) -> None:
    rows = await _run(mock_server, "conforming", declared=True)
    schema_requests = [
        record.body
        for record in mock_server.request_log
        if isinstance(record.body, dict)
        and (record.body.get("response_format") or {}).get("type") == "json_schema"
    ]

    assert rows["structured.json-schema"].status == "pass"
    assert len(schema_requests) == 1
    schema = schema_requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"] == {
        "answer": {"type": "string"},
        "count": {"type": "integer"},
    }

    mock_server.clear_requests()
    near_miss = await _run(mock_server, "structured_wording_near_miss", declared=True)
    assert near_miss["structured.json-schema"].status == "pass"
