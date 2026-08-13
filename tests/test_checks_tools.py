"""Tier 5 tool checks fire on each enclave-breaking wire mutation."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.tools import (
    _ToolTurn,
    run_parallel_tool_delta_check,
    run_tool_checks,
    run_tool_round_trip_check,
)
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    declared: bool | None = True,
) -> list[CheckResult]:
    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results = await run_tool_checks(client, "mock/model", declared=declared)
    assert len(results) == 2
    assert [result.id for result in results] == [
        "tools.parallel-deltas",
        "tools.round-trip",
    ]
    return results


def _by_id(results: list[CheckResult]) -> dict[str, CheckResult]:
    rows = {result.id: result for result in results}
    assert len(rows) == 2
    return rows


@pytest.mark.asyncio
async def test_parallel_tools_and_empty_string_round_trip_control_are_green(
    mock_server: MockOpenAIServer,
) -> None:
    rows = _by_id(await _run(mock_server, "conforming"))

    assert rows["tools.parallel-deltas"].status == "pass"
    assert rows["tools.parallel-deltas"].measured["tool_call_count"] == 2
    assert rows["tools.parallel-deltas"].measured["tool_delta_count"] == 4
    assert rows["tools.round-trip"].status == "pass"

    completion_requests = [
        record
        for record in mock_server.request_log
        if record.path == "/v1/chat/completions"
    ]
    assert len(completion_requests) == 2
    replay = completion_requests[1].body
    assert isinstance(replay, dict)
    messages = replay.get("messages")
    assert isinstance(messages, list) and len(messages) == 4
    assistant = messages[1]
    assert isinstance(assistant, dict)
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ""
    assert assistant["content"] is not None
    tool_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    assert len(tool_messages) == 2
    assert {message["tool_call_id"] for message in tool_messages} == {
        "call_weather",
        "call_time",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "measured_key"),
    [
        ("tool_deltas_missing_index", "missing_index_count"),
        ("tool_name_late", "late_name_indices"),
        ("tool_arguments_invalid_json", "argument_errors"),
    ],
)
async def test_parallel_tool_delta_mutations_fail_but_control_passes(
    mock_server: MockOpenAIServer,
    mode: str,
    measured_key: str,
) -> None:
    broken = _by_id(await _run(mock_server, mode))
    mock_server.clear_requests()
    control = _by_id(await _run(mock_server, "conforming"))

    failure = broken["tools.parallel-deltas"]
    assert failure.status == "fail"
    assert failure.owner == "provider"
    measured = failure.measured[measured_key]
    if mode == "tool_single_call":
        assert measured == 1
    elif isinstance(measured, int):
        assert measured >= 1
    else:
        assert len(measured) >= 1
    assert broken["tools.round-trip"].status == "skip"
    assert control["tools.parallel-deltas"].status == "pass"
    assert control["tools.round-trip"].status == "pass"


@pytest.mark.asyncio
async def test_undeclared_rejected_tools_skip_but_declared_rejection_fails(
    mock_server: MockOpenAIServer,
) -> None:
    undeclared = _by_id(await _run(mock_server, "rejects_tools", declared=None))
    mock_server.clear_requests()
    declared = _by_id(await _run(mock_server, "rejects_tools", declared=True))
    mock_server.clear_requests()
    control = _by_id(await _run(mock_server, "conforming", declared=True))

    assert undeclared["tools.parallel-deltas"].status == "skip"
    assert undeclared["tools.round-trip"].status == "skip"
    assert declared["tools.parallel-deltas"].status == "fail"
    assert declared["tools.parallel-deltas"].owner == "configuration"
    assert control["tools.parallel-deltas"].status == "pass"


@pytest.mark.asyncio
async def test_round_trip_rejects_empty_string_shape_with_negative_control(
    mock_server: MockOpenAIServer,
) -> None:
    rejected = _by_id(await _run(mock_server, "rejects_empty_tool_content"))
    rejected_requests = list(mock_server.request_log)
    mock_server.clear_requests()
    accepted = _by_id(await _run(mock_server, "conforming"))

    assert rejected["tools.parallel-deltas"].status == "pass"
    assert rejected["tools.round-trip"].status == "fail"
    assert rejected["tools.round-trip"].owner == "configuration"
    assert accepted["tools.round-trip"].status == "pass"

    replay_bodies: list[dict[str, Any]] = [
        record.body
        for record in rejected_requests
        if isinstance(record.body, dict) and record.body.get("stream") is False
    ]
    assert len(replay_bodies) == 1
    replay_messages = replay_bodies[0]["messages"]
    assert isinstance(replay_messages, list)
    assert replay_messages[1]["content"] == ""


@pytest.mark.asyncio
async def test_declared_false_tools_skip_without_a_billed_completion(
    mock_server: MockOpenAIServer,
) -> None:
    rows = _by_id(await _run(mock_server, "conforming", declared=False))

    assert rows["tools.parallel-deltas"].status == "skip"
    assert rows["tools.round-trip"].status == "skip"
    assert mock_server.request_log == []


@pytest.mark.asyncio
async def test_single_well_formed_tool_call_warns_rather_than_fails(
    mock_server: MockOpenAIServer,
) -> None:
    # Cerebras/gpt-oss-120b answered the forced-parallel prompt with ONE
    # perfectly-formed call. The enclave breaks on a missing index or a late
    # function.name, never on how many calls a model chose to make, so this
    # cannot be a conformance failure.
    single = _by_id(await _run(mock_server, "tool_single_call"))[
        "tools.parallel-deltas"
    ]

    assert single.status == "warn"
    assert single.measured["tool_call_count"] == 1
    assert single.measured["missing_index_count"] == 0
    assert not single.measured["late_name_indices"]
    assert "not exercised" in single.measured["reason"]

    # Negative control: a malformed delta at the same call count still fails.
    malformed = _by_id(await _run(mock_server, "tool_deltas_missing_index"))[
        "tools.parallel-deltas"
    ]
    assert malformed.status == "fail"


@pytest.mark.asyncio
async def test_transient_tool_probe_and_round_trip_are_inconclusive(
    mock_server: MockOpenAIServer,
) -> None:
    probe = _by_id(await _run(mock_server, "tools_probe_backend_down"))
    mock_server.clear_requests()
    replay = _by_id(await _run(mock_server, "tool_round_trip_backend_down"))

    assert probe["tools.parallel-deltas"].status == "warn"
    assert probe["tools.parallel-deltas"].measured["inconclusive"] is True
    assert probe["tools.round-trip"].status == "skip"
    assert replay["tools.parallel-deltas"].status == "pass"
    assert replay["tools.round-trip"].status == "warn"
    assert "unknown" in replay["tools.round-trip"].measured["reason"]


@pytest.mark.asyncio
async def test_round_trip_model_selected_tool_call_warns(
    mock_server: MockOpenAIServer,
) -> None:
    rows = _by_id(await _run(mock_server, "tool_round_trip_calls_tool"))
    result = rows["tools.round-trip"]

    assert rows["tools.parallel-deltas"].status == "pass"
    assert result.status == "warn"
    assert result.measured["alternative_tool_call_count"] == 1
    assert "model chose another tool call" in result.measured["reason"]


@pytest.mark.asyncio
async def test_tool_transport_errors_are_inconclusive() -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async def no_delay(_: float) -> None:
        return None

    async with GatewayClient(
        "https://example.test/v1",
        "test-key",
        transport=httpx.MockTransport(offline),
        sleep=no_delay,
    ) as client:
        parallel, turn = await run_parallel_tool_delta_check(
            client, "mock/model", declared=True
        )
        replay = await run_tool_round_trip_check(
            client,
            "mock/model",
            _ToolTurn(
                [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup_time", "arguments": "{}"},
                    }
                ]
            ),
            unavailable_reason="unused",
        )

    assert turn is None
    assert parallel.status == "warn"
    assert parallel.measured["inconclusive"] is True
    assert replay.status == "warn"
    assert "unknown" in replay.measured["reason"]
