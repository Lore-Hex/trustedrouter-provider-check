"""Tier 4 checks isolate framing, timing, accounting, and error signaling."""

from __future__ import annotations

import asyncio

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.streaming import run_streaming_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult

_STREAM_IDS = {
    "stream.response-status",
    "stream.sse-framing",
    "stream.error-signaling",
    "stream.done",
    "stream.usage",
    "stream.first-delta",
    "stream.incremental-delivery",
}


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    model: str = "mock/model",
    first_token_budget_seconds: float | None = None,
) -> list[CheckResult]:
    async def no_retry_delay(_: float) -> None:
        return None

    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
        sleep=no_retry_delay,
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results = await run_streaming_checks(
                client,
                model,
                first_token_budget_seconds=first_token_budget_seconds,
            )
    assert len(results) == len(_STREAM_IDS)
    assert {result.id for result in results} == _STREAM_IDS
    return results


def _statuses(results: list[CheckResult]) -> dict[str, str]:
    rows: dict[str, str] = {result.id: result.status for result in results}
    assert len(rows) == len(_STREAM_IDS)
    return rows


def _green() -> dict[str, str]:
    return {check_id: "pass" for check_id in _STREAM_IDS}


@pytest.mark.asyncio
async def test_streaming_conforming_control_is_green(
    mock_server: MockOpenAIServer,
) -> None:
    assert _statuses(await _run(mock_server, "conforming")) == _green()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["no_space_framing", "non_sse_200"])
async def test_streaming_rejects_unreadable_framing_without_rejecting_sse(
    mock_server: MockOpenAIServer,
    mode: str,
) -> None:
    rejected = _statuses(await _run(mock_server, mode))
    accepted = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.sse-framing"] = "fail"
    for check_id in (
        "stream.done",
        "stream.usage",
        "stream.first-delta",
        "stream.incremental-delivery",
    ):
        expected[check_id] = "skip"
    assert rejected == expected
    assert accepted == _green()


@pytest.mark.asyncio
async def test_streaming_warns_on_missing_done_without_rejecting_complete_stream(
    mock_server: MockOpenAIServer,
) -> None:
    missing = _statuses(await _run(mock_server, "no_done_sentinel"))
    complete = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.done"] = "warn"
    assert missing == expected
    assert complete == _green()


@pytest.mark.asyncio
async def test_streaming_requires_requested_usage_but_accepts_usage_in_any_chunk(
    mock_server: MockOpenAIServer,
) -> None:
    missing = _statuses(await _run(mock_server, "ignores_include_usage"))
    present = _statuses(await _run(mock_server, "bad_usage"))

    expected = _green()
    expected["stream.usage"] = "fail"
    assert missing == expected
    assert present == _green()


@pytest.mark.asyncio
async def test_streaming_first_delta_ignores_early_role_and_ping(
    mock_server: MockOpenAIServer,
) -> None:
    late = _statuses(
        await _run(
            mock_server,
            "role_then_late_content",
            first_token_budget_seconds=(
                mock_server.knobs.late_content_delay_seconds / 2
            ),
        )
    )
    generous = _statuses(
        await _run(
            mock_server,
            "role_then_late_content",
            first_token_budget_seconds=HANG_GUARD_SECONDS,
        )
    )

    expected = _green()
    expected["stream.first-delta"] = "fail"
    assert late == expected
    assert generous == _green()


@pytest.mark.asyncio
async def test_streaming_empty_success_fails_first_delta_only(
    mock_server: MockOpenAIServer,
) -> None:
    empty = _statuses(await _run(mock_server, "finish_reason_only"))
    non_empty = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.first-delta"] = "fail"
    expected["stream.incremental-delivery"] = "skip"
    assert empty == expected
    assert non_empty == _green()


@pytest.mark.asyncio
async def test_streaming_warns_when_whole_stream_is_buffered(
    mock_server: MockOpenAIServer,
) -> None:
    buffered = _statuses(await _run(mock_server, "buffered_stream"))
    incremental = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.incremental-delivery"] = "warn"
    assert buffered == expected
    assert incremental == _green()


@pytest.mark.asyncio
async def test_streaming_midstream_error_is_failure_not_warning(
    mock_server: MockOpenAIServer,
) -> None:
    truncated = _statuses(await _run(mock_server, "midstream_error"))
    successful = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.error-signaling"] = "fail"
    for check_id in (
        "stream.done",
        "stream.usage",
        "stream.incremental-delivery",
    ):
        expected[check_id] = "skip"
    assert truncated == expected
    assert successful == _green()


@pytest.mark.asyncio
async def test_streaming_http_error_is_visible_before_stream_bytes(
    mock_server: MockOpenAIServer,
) -> None:
    capacity = _statuses(await _run(mock_server, "queue_then_429"))
    successful = _statuses(await _run(mock_server, "conforming"))

    expected = _green()
    expected["stream.response-status"] = "warn"
    for check_id in (
        "stream.sse-framing",
        "stream.done",
        "stream.usage",
        "stream.first-delta",
        "stream.incremental-delivery",
    ):
        expected[check_id] = "skip"
    assert capacity == expected
    assert successful == _green()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "delta-content",
        "delta-reasoning_content",
        "delta-reasoning",
        "delta-thinking",
        "delta-text",
        "delta-output_text",
        "message-content",
        "message-reasoning_content",
        "message-reasoning",
        "message-thinking",
        "message-text",
        "choice-text",
        "tool-call",
    ],
)
async def test_streaming_accepts_every_vendored_nonempty_delta_shape(
    mock_server: MockOpenAIServer,
    shape: str,
) -> None:
    results = await _run(mock_server, "delta_shapes", model=f"mock/{shape}")
    assert _statuses(results) == _green()
