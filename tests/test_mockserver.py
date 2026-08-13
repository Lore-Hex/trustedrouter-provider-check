"""Verify each mock mode reproduces its named provider wire-contract behavior."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from tests.mockserver.app import (
    HANG_GUARD_SECONDS,
    MOCK_MODE_GROUPS,
    MOCK_MODES,
    MockOpenAIServer,
)


def _post(
    server: MockOpenAIServer,
    mode: str,
    *,
    stream: bool = True,
    extra: dict[str, Any] | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {
        "model": "mock/model",
        "messages": [{"role": "user", "content": "reply exactly PONG"}],
        "stream": stream,
    }
    if extra:
        body.update(extra)
    return httpx.post(
        f"{server.base_url}/v1/chat/completions",
        headers={"X-Mock-Mode": mode},
        json=body,
        timeout=HANG_GUARD_SECONDS,
    )


def _json_events(response: httpx.Response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response.content.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if payload == b"[DONE]":
            continue
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        events.append(parsed)
    return events


def _visible_content(events: list[dict[str, Any]]) -> list[str]:
    return [
        event["choices"][0]["delta"]["content"]
        for event in events
        if event.get("choices") and event["choices"][0].get("delta", {}).get("content")
    ]


def _stream_frame_arrivals(server: MockOpenAIServer, mode: str) -> list[float]:
    body = {
        "model": "mock/model",
        "messages": [{"role": "user", "content": "reply exactly PONG"}],
        "stream": True,
    }
    arrivals: list[float] = []
    with httpx.stream(
        "POST",
        f"{server.base_url}/v1/chat/completions",
        headers={"X-Mock-Mode": mode},
        json=body,
        timeout=HANG_GUARD_SECONDS,
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                arrivals.append(time.perf_counter())
    return arrivals


def test_conforming_models_and_non_stream_completion(
    mock_server: MockOpenAIServer,
) -> None:
    models = httpx.get(f"{mock_server.base_url}/v1/models", timeout=HANG_GUARD_SECONDS)
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mock/model"

    response = _post(mock_server, "conforming", stream=False)
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "mock/model"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "PONG"}
    usage = payload["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
    assert all(isinstance(value, int) for value in usage.values())


def test_conforming_sse_wire(mock_server: MockOpenAIServer) -> None:
    response = _post(
        mock_server,
        "conforming",
        extra={"stream_options": {"include_usage": True}},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 5
    common = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "mock/model",
    }
    for event in events:
        assert {key: event[key] for key in common} == common
        assert set(event) <= {*common, "choices", "usage"}
    assert events[0]["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }
    ]
    content = [
        event["choices"][0]["delta"]["content"]
        for event in events
        if event.get("choices") and event["choices"][0].get("delta", {}).get("content")
    ]
    assert content == ["PO", "NG"]
    for event in events[1:3]:
        assert set(event["choices"][0]) == {"index", "delta", "finish_reason"}
        assert event["choices"][0]["index"] == 0
        assert event["choices"][0]["finish_reason"] is None
    assert events[3]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert events[4]["choices"] == []
    assert events[4]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


def test_conforming_stream_arrives_incrementally_but_buffered_stream_does_not(
    mock_server: MockOpenAIServer,
) -> None:
    delay = mock_server.knobs.stream_chunk_delay_seconds
    conforming = _stream_frame_arrivals(mock_server, "conforming")
    buffered = _stream_frame_arrivals(mock_server, "buffered_stream")

    assert len(conforming) == 5
    assert len(buffered) == 5
    # Buffering is defined by WHEN the first byte arrives, not by how tightly
    # the frames cluster afterwards. Asserting that the buffered frames land
    # within one chunk delay of each other measures the reader's scheduling, not
    # the server's: a descheduled client spreads genuinely simultaneous frames
    # past any fixed threshold, which is how this test flaked under load. The
    # withheld first byte is the property that actually defeats a first-token
    # measurement, and no amount of jitter can make it arrive early.
    assert conforming[-1] - conforming[0] > delay * 2
    assert buffered[0] >= mock_server.knobs.buffered_stream_delay_seconds * 0.8


def test_unknown_mock_mode_fails_loudly(mock_server: MockOpenAIServer) -> None:
    unknown = "confroming"
    response = _post(mock_server, unknown)
    assert response.status_code == 400
    error = response.json()["error"]
    assert unknown in error["message"]
    assert error["valid_modes"] == list(MOCK_MODES)


def test_mock_mode_registry_is_grouped_unique_and_documented() -> None:
    assert set(MOCK_MODE_GROUPS) == {
        "catalog_and_serving_disagree",
        "request_rejected_before_completion",
        "completion_and_shared_shapes",
        "stream_transport_and_deltas",
    }
    registered = [mode for modes in MOCK_MODE_GROUPS.values() for mode in modes]
    assert registered == list(MOCK_MODES)
    assert len(registered) == len(set(registered))
    descriptions = [
        description
        for modes in MOCK_MODE_GROUPS.values()
        for description in modes.values()
    ]
    assert len(descriptions) == len(registered)
    assert all(description.endswith(".") for description in descriptions)


@pytest.mark.parametrize(
    ("mode", "model", "status"),
    [
        ("catalog_lists_403_model", "mock/catalog-forbidden", 403),
        ("catalog_lists_404_model", "mock/catalog-missing", 404),
    ],
)
def test_catalog_advertises_unservable_models(
    mock_server: MockOpenAIServer, mode: str, model: str, status: int
) -> None:
    models = httpx.get(
        f"{mock_server.base_url}/v1/models",
        headers={"X-Mock-Mode": mode},
        timeout=HANG_GUARD_SECONDS,
    )
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == model
    response = _post(mock_server, mode, stream=False, extra={"model": model})
    assert response.status_code == status
    if status == 403:
        assert response.headers["content-type"] == "text/html"
        assert response.content.startswith(b"<html>")
    else:
        assert response.json()["error"]["type"] == "model_not_found"


def test_empty_content_is_really_empty(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "empty_content", stream=False)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ""

    stream_response = _post(mock_server, "empty_content")
    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"] == "text/event-stream"
    assert stream_response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(stream_response)
    assert [event["choices"][0]["delta"].get("content") for event in events] == [
        "",
        "",
        None,
    ]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_finish_reason_only_has_no_content_delta(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "finish_reason_only")
    events = _json_events(response)
    assert response.status_code == 200
    assert events == [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "mock/model",
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        }
    ]
    assert response.content.endswith(b"data: [DONE]\n\n")


def test_queue_then_429_withholds_first_byte(mock_server: MockOpenAIServer) -> None:
    delay = mock_server.knobs.queue_delay_seconds
    started = time.perf_counter()
    response = _post(mock_server, "queue_then_429", stream=False)
    elapsed = time.perf_counter() - started
    assert response.status_code == 429
    assert elapsed >= delay * 0.8
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_no_space_framing_is_exact(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "no_space_framing")
    assert response.status_code == 200
    assert response.content.startswith(b"data:{")
    assert b"data: {" not in response.content
    assert response.content.endswith(b"data:[DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 4
    assert _visible_content(events) == ["PO", "NG"]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_no_done_sentinel_really_omits_done(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "no_done_sentinel")
    assert response.status_code == 200
    assert b"[DONE]" not in response.content
    events = _json_events(response)
    assert len(events) == 4
    assert _visible_content(events) == ["PO", "NG"]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_only_named_missing_done_and_empty_body_modes_lack_done(
    mock_server: MockOpenAIServer,
) -> None:
    successful_sse_modes: set[str] = set()
    missing_done: list[str] = []
    for mode in MOCK_MODES:
        response = _post(mock_server, mode)
        if (
            response.status_code != 200
            or response.headers["content-type"] != "text/event-stream"
        ):
            continue
        events = _json_events(response)
        if any("error" in event for event in events):
            continue
        successful_sse_modes.add(mode)
        if b"[DONE]" not in response.content:
            missing_done.append(mode)
    assert successful_sse_modes == set(MOCK_MODES) - {
        "catalog_lists_403_model",
        "catalog_lists_404_model",
        "midstream_error",
        "non_sse_200",
        "queue_then_429",
    }
    assert missing_done == ["no_done_sentinel", "empty_stream_200"]


def test_ignores_include_usage_really_has_no_usage(
    mock_server: MockOpenAIServer,
) -> None:
    response = _post(
        mock_server,
        "ignores_include_usage",
        extra={"stream_options": {"include_usage": True}},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 4
    assert events[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
    assert [events[index]["choices"][0]["delta"]["content"] for index in (1, 2)] == [
        "PO",
        "NG",
    ]
    assert events[3]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in event for event in events)


def test_tool_deltas_missing_index(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "tool_deltas_missing_index")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 2
    tool_call = events[0]["choices"][0]["delta"]["tool_calls"][0]
    assert "index" not in tool_call
    assert tool_call["function"]["name"] == "lookup"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_tool_name_arrives_late(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "tool_name_late")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 3
    first = events[0]["choices"][0]["delta"]["tool_calls"][0]
    second = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert first["index"] == second["index"] == 0
    assert "name" not in first["function"]
    assert second["function"]["name"] == "lookup"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_midstream_error_follows_content_on_200(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "midstream_error")
    events = _json_events(response)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert len(events) == 3
    assert events[1]["choices"][0]["delta"]["content"] == "PO"
    assert events[2]["error"]["type"] == "provider_error"
    assert b"[DONE]" not in response.content


def test_non_sse_200_is_plain_json(mock_server: MockOpenAIServer) -> None:
    response = _post(mock_server, "non_sse_200")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content.startswith(b"{")
    assert b"data:" not in response.content
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "PONG"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_bad_usage_is_non_integer_and_inconsistent(
    mock_server: MockOpenAIServer,
) -> None:
    response = _post(mock_server, "bad_usage")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    assert len(events) == 5
    assert _visible_content(events) == ["PO", "NG"]
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    usage = events[-1]["usage"]
    assert not isinstance(usage["prompt_tokens"], int)
    assert usage["completion_tokens"] != usage["total_tokens"]


def test_rejects_temperature_zero_only_at_zero(mock_server: MockOpenAIServer) -> None:
    rejected = _post(
        mock_server,
        "rejects_temperature_zero",
        stream=False,
        extra={"temperature": 0},
    )
    accepted = _post(
        mock_server,
        "rejects_temperature_zero",
        stream=False,
        extra={"temperature": 0.1},
    )
    assert rejected.status_code == 400
    assert accepted.status_code == 200
    assert rejected.json()["error"]["type"] == "invalid_request"
    assert accepted.json()["choices"][0]["message"]["content"] == "PONG"


@pytest.mark.parametrize("field", ["top_k", "thinking", "reasoning_effort"])
def test_strict_extra_fields_rejects_only_the_extra_fields(
    mock_server: MockOpenAIServer, field: str
) -> None:
    rejected = _post(
        mock_server,
        "strict_extra_fields",
        stream=False,
        extra={field: "unexpected"},
    )
    accepted = _post(mock_server, "strict_extra_fields", stream=False)
    assert rejected.status_code == 400
    assert accepted.status_code == 200
    assert rejected.json()["error"]["type"] == "invalid_request"
    assert accepted.json()["choices"][0]["message"]["content"] == "PONG"


def test_buffered_stream_withholds_all_content_deltas(
    mock_server: MockOpenAIServer,
) -> None:
    delay = mock_server.knobs.buffered_stream_delay_seconds
    started = time.perf_counter()
    response = _post(mock_server, "buffered_stream")
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    events = _json_events(response)
    content_deltas = [
        event["choices"][0]["delta"]["content"]
        for event in events
        if event.get("choices") and event["choices"][0].get("delta", {}).get("content")
    ]
    assert elapsed >= delay * 0.8
    assert content_deltas == _visible_content(events) == ["PO", "NG"]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_request_log_captures_method_path_headers_and_body(
    mock_server: MockOpenAIServer,
) -> None:
    assert mock_server.request_log == []
    body = {
        "model": "mock/logged",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    response = httpx.post(
        f"{mock_server.base_url}/v1/chat/completions",
        headers={"X-Mock-Mode": "conforming", "X-Test-Sentinel": "captured"},
        json=body,
        timeout=HANG_GUARD_SECONDS,
    )
    assert response.status_code == 200
    record = mock_server.request_log[-1]
    assert record.method == "POST"
    assert record.path == "/v1/chat/completions"
    assert record.headers["x-mock-mode"] == "conforming"
    assert record.headers["x-test-sentinel"] == "captured"
    assert record.body == body
    assert json.loads(record.raw_body) == body


def test_request_log_preserves_unparseable_raw_body(
    mock_server: MockOpenAIServer,
) -> None:
    response = httpx.post(
        f"{mock_server.base_url}/v1/chat/completions",
        headers={"X-Mock-Mode": "conforming", "Content-Type": "application/json"},
        content=b"{not-json",
        timeout=HANG_GUARD_SECONDS,
    )
    assert response.status_code == 200
    record = mock_server.request_log[-1]
    assert record.body is None
    assert record.raw_body == b"{not-json"


def test_request_log_distinguishes_no_body(mock_server: MockOpenAIServer) -> None:
    response = httpx.get(
        f"{mock_server.base_url}/v1/models", timeout=HANG_GUARD_SECONDS
    )
    assert response.status_code == 200
    record = mock_server.request_log[-1]
    assert record.body is None
    assert record.raw_body == b""


def test_close_before_start_returns() -> None:
    server = MockOpenAIServer()
    server.close()
