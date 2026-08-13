"""Join the mock wire stream to the exact parser and route classifier used in production."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import httpx
import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check import contract, snapshot


async def _observe(server: MockOpenAIServer, mode: str) -> contract._StreamObservation:
    body = {
        "model": "mock/model",
        "messages": [{"role": "user", "content": "reply exactly PONG"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with httpx.AsyncClient(timeout=HANG_GUARD_SECONDS) as client:
        started = time.perf_counter()
        async with client.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            headers={"X-Mock-Mode": mode},
            json=body,
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            return await contract._observe_provider_stream(response, started=started)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "has_content", "finish_reason", "stream_error", "usage"),
    [
        ("conforming", True, "stop", None, (4, 2, 0)),
        (
            "midstream_error",
            True,
            None,
            ("provider_error", None, "upstream reset"),
            (0, 0, 0),
        ),
        ("no_done_sentinel", True, "stop", None, (4, 2, 0)),
        ("no_space_framing", True, "stop", None, (4, 2, 0)),
        ("ignores_include_usage", True, "stop", None, (0, 0, 0)),
        ("finish_reason_only", False, "stop", None, (4, 2, 0)),
        ("buffered_stream", True, "stop", None, (4, 2, 0)),
    ],
)
async def test_production_stream_observer_reads_mock_wire_modes(
    mock_server: MockOpenAIServer,
    mode: str,
    has_content: bool,
    finish_reason: str | None,
    stream_error: tuple[str, int | None, str | None] | None,
    usage: tuple[int, int, int],
) -> None:
    observation = await _observe(mock_server, mode)

    assert observation.ttfb_milliseconds is not None
    assert observation.elapsed_milliseconds >= observation.ttfb_milliseconds
    if has_content:
        assert observation.first_token_milliseconds is not None
        assert observation.last_token_milliseconds is not None
        assert observation.ttfb_milliseconds <= observation.first_token_milliseconds
        assert (
            observation.first_token_milliseconds
            <= observation.last_token_milliseconds
            <= observation.elapsed_milliseconds
        )
    else:
        assert observation.first_token_milliseconds is None
        assert observation.last_token_milliseconds is None
    assert observation.finish_reason == finish_reason
    assert observation.stream_error == stream_error
    assert (
        observation.usage.input_tokens,
        observation.usage.output_tokens,
        observation.usage.reasoning_tokens,
    ) == usage

    if mode == "buffered_stream":
        assert observation.ttfb_milliseconds >= (
            mock_server.knobs.buffered_stream_delay_seconds * 1000 * 0.8
        )
        assert (
            observation.first_token_milliseconds == observation.last_token_milliseconds
        )


_CONTRACT_DATA: dict[str, Any] = snapshot.load_contract()
_CLASSIFICATION_ROWS = [
    *[
        pytest.param(status, "healthy response", "dead", id=f"dead-status-{status}")
        for status in _CONTRACT_DATA["markers"]["dead_statuses"]
    ],
    *[
        pytest.param(
            None, f"provider says: {marker}", "dead", id=f"dead-marker-{marker}"
        )
        for marker in _CONTRACT_DATA["markers"]["dead_markers"]
    ],
    pytest.param(200, "model not found", "ok", id="http-200-wins"),
    pytest.param(429, "rate limited", "flaky", id="capacity-is-flaky"),
    pytest.param(503, "internal error", "flaky", id="server-error-is-flaky"),
    pytest.param(None, "network error", "flaky", id="network-is-flaky"),
]


@pytest.mark.parametrize(("status", "body", "expected"), _CLASSIFICATION_ROWS)
def test_route_classifier_replays_dead_statuses_and_markers(
    status: int | None, body: str, expected: str
) -> None:
    assert contract._classify(status, body) == expected


def test_stream_observation_defaults_remain_explicit() -> None:
    assert asdict(contract._StreamObservation()) == {
        "ttfb_milliseconds": None,
        "first_token_milliseconds": None,
        "last_token_milliseconds": None,
        "elapsed_milliseconds": 0,
        "finish_reason": None,
        "stream_error": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
    }
