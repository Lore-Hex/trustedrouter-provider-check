"""A configurable, dependency-free OpenAI-compatible mock HTTP server."""

from __future__ import annotations

import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

# A hang guard, never a latency assertion. Deliberate fixture delays are far
# below this value so an overloaded test machine cannot turn it into a tight
# performance bound.
HANG_GUARD_SECONDS = 30


@dataclass(frozen=True)
class RequestRecord:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None
    raw_body: bytes


@dataclass
class ModeKnobs:
    queue_delay_seconds: float = 0.08
    stream_chunk_delay_seconds: float = 0.025
    buffered_stream_delay_seconds: float = 0.08
    late_content_delay_seconds: float = 0.08


# Modes are grouped by the request-handler branch that implements them. Each
# description records the production behavior that made the fixture relevant.
MOCK_MODE_GROUPS: dict[str, dict[str, str]] = {
    "catalog_and_serving_disagree": {
        "catalog_lists_403_model": "A listed route serves an HTML authorization error.",
        "catalog_lists_404_model": "A listed route is absent from chat serving.",
        "invalid_declared_catalog": "A marketplace declaration uses a native id where owner/model is required.",
        "native_models_empty": "Native discovery returns an empty model list.",
        "native_noncanonical_ids": "Native discovery returns valid engine-native ids without owner prefixes.",
        "models_without_object_envelope": "Native discovery omits the top-level object:list envelope (pearlresearch.ai).",
        "models_without_data_array": "Native discovery returns an object with no data[] array at all.",
        "capability_probe_backend_down": "A capability probe hits a 502 rather than a parameter refusal.",
        "structured_json_in_content_prose_in_reasoning": "Exact JSON in content while reasoning_content holds prose (Fireworks glm-5p2).",
        "gzipped_sse": "A conformant SSE stream compressed with gzip (Nebius).",
        "models_bare_array": "Native discovery returns a top-level JSON array, not {data: [...]} (Together).",
    },
    "request_rejected_before_completion": {
        "queue_then_429": "Queueing consumes the probe budget before capacity rejects.",
        "rejects_empty_tool_content": "A backend rejects the enclave's empty-string assistant tool turn.",
        "rejects_max_tokens": "A backend rejects the selected max_tokens spelling.",
        "rejects_response_format": "A backend rejects a forwarded response_format field.",
        "rejects_temperature_zero": "A backend rejects deterministic temperature=0.",
        "rejects_tools": "A backend rejects forwarded tool definitions.",
        "strict_extra_fields": "A strict schema rejects translator-only request fields.",
    },
    "completion_and_shared_shapes": {
        "conforming": "The OpenAI-compatible control response.",
        "empty_content": "A successful response carries no visible assistant text.",
        "bad_usage": "Token counts are non-integer and internally inconsistent.",
        "chat_reasoning_shapes": "A completion uses a tolerated content or reasoning shape.",
        "structured_non_json": "An accepted JSON-object request returns plain text.",
        "structured_schema_violation": "An accepted JSON-Schema request returns schema-invalid JSON.",
        "unknown_finish_reason": "A completion reports a provider-specific terminal reason.",
        "wrong_model": "A completion reports a different native model id.",
        "wrong_pong": "A non-empty completion misses the deterministic PONG marker.",
    },
    "stream_transport_and_deltas": {
        "buffered_stream": "A provider withholds every answer frame until completion.",
        "delta_shapes": "A stream uses a content-delta shape tolerated by production.",
        "finish_reason_only": "A stream finishes successfully without a visible token.",
        "ignores_include_usage": "A provider suppresses requested stream usage.",
        "midstream_error": "A 200 stream changes into an in-band provider error.",
        "no_done_sentinel": "A stream closes without the terminal [DONE] sentinel.",
        "no_space_framing": "SSE data fields omit the optional space after the colon.",
        "non_sse_200": "stream=true receives a plain JSON completion.",
        "perf_insufficient_output": "A long throughput probe reports fewer than 128 output tokens.",
        "role_then_late_content": "A role chunk and ping arrive before late real content.",
        "tool_deltas_missing_index": "Tool-call fragments omit their correlation index.",
        "tool_arguments_invalid_json": "Tool-call argument fragments do not concatenate to valid JSON.",
        "tool_name_late": "A tool name arrives after the first delta for its index.",
        "tool_single_call": "A forced parallel request emits only one tool call.",
    },
}
MOCK_MODES = tuple(
    mode
    for modes_by_provenance in MOCK_MODE_GROUPS.values()
    for mode in modes_by_provenance
)

_BAD_USAGE = {
    "prompt_tokens": "four",
    "completion_tokens": 2,
    "total_tokens": 1,
}


@dataclass
class _ServerState:
    request_log: list[RequestRecord] = field(default_factory=list)
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    knobs: ModeKnobs = field(default_factory=ModeKnobs)


class _MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        self.state = _ServerState()
        super().__init__(server_address, _MockHandler)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def _sse_frame(payload: Any, *, space: bool = True) -> bytes:
    value = payload if isinstance(payload, str) else _json_bytes(payload).decode()
    separator = b"data: " if space else b"data:"
    return separator + value.encode() + b"\n\n"


def _completion(
    model: str,
    *,
    content: Any = "PONG",
    usage: Any = None,
    response_model: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": response_model or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    return result


def _catalog(*, invalid: bool = False) -> dict[str, Any]:
    return {
        "object": "list",
        "contract_version": "2.0",
        "provider": {
            "id": "mock",
            "status_url": "https://status.example.test",
            "support_contact": "mailto:support@example.test",
            "incident_contact": "mailto:incident@example.test",
            "regions": ["test-1"],
            "request_id_header": "X-Request-ID",
            "error_contract": {
                "rate_limit_status": 429,
                "overload_status": 503,
                "retry_after_header": "Retry-After",
                "account_quota_error_codes": ["account_quota"],
            },
        },
        "data": [
            {
                "id": "deepseek-chat" if invalid else "mock/model",
                "object": "model",
                "owned_by": "mock",
                "name": "Mock model",
                "type": "chat",
                "context_length": 8192,
                "max_output_tokens": 2048,
                "endpoints": ["chat/completions"],
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "capabilities": {
                    "streaming": True,
                    "tools": True,
                    "structured_output": True,
                    "reasoning": True,
                    "prompt_caching": False,
                },
                "pricing": {
                    "currency": "USD",
                    "unit": "per_1m_tokens",
                    "input": "1.0",
                    "output": "2.0",
                    "cached_input": None,
                    "cache_write": None,
                    "minimum_request": "0",
                },
                "lifecycle": {
                    "status": "active",
                    "deprecation_at": None,
                    "retirement_at": None,
                    "replacement_model_id": None,
                },
                "reliability": {
                    "first_token_timeout_seconds": 20,
                    "completion_timeout_seconds": 80,
                    "stream_idle_timeout_seconds": 30,
                    "capacity_scope": "model",
                },
            }
        ],
    }


def _chat_shape_completion(model: str) -> dict[str, Any]:
    payload = _completion(model, content="")
    choice = payload["choices"][0]
    assert isinstance(choice, dict)
    message = choice["message"]
    assert isinstance(message, dict)
    shape = model.rsplit("/", 1)[-1]
    if shape == "content-parts":
        message["content"] = [{"type": "text", "text": "PONG"}]
    elif shape == "content-parts-content":
        message["content"] = [{"type": "text", "content": "PONG"}]
    elif shape == "reasoning-content-string":
        message["reasoning_content"] = "PONG"
    elif shape == "reasoning-content-parts":
        message["reasoning_content"] = [{"type": "text", "text": "PONG"}]
    elif shape == "reasoning-string":
        message["reasoning"] = "PONG"
    elif shape == "reasoning-parts":
        message["reasoning"] = [{"type": "text", "text": "PONG"}]
    else:
        message["content"] = "PONG"
    return payload


def _chunk(
    model: str,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    choices: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
        "choices": choices
        if choices is not None
        else [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def _shape_chunk(model: str) -> dict[str, Any]:
    shape = model.rsplit("/", 1)[-1]
    if shape == "tool-call":
        return _chunk(
            model,
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_mock",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            },
        )
    if shape.startswith("delta-"):
        return _chunk(model, delta={shape.removeprefix("delta-"): "PONG"})
    if shape.startswith("message-"):
        return _chunk(
            model,
            choices=[
                {
                    "index": 0,
                    "message": {shape.removeprefix("message-"): "PONG"},
                    "finish_reason": None,
                }
            ],
        )
    if shape == "choice-text":
        return _chunk(
            model,
            choices=[{"index": 0, "text": "PONG", "finish_reason": None}],
        )
    return _chunk(model, delta={"content": "PONG"})


def _tool_stream_frames(
    model: str,
    *,
    mode: str,
    include_usage: bool,
) -> list[bytes]:
    first_calls: list[dict[str, Any]] = [
        {
            "index": 0,
            "id": "call_weather",
            "type": "function",
            "function": {"name": "lookup_weather", "arguments": '{"city":'},
        },
        {
            "index": 1,
            "id": "call_time",
            "type": "function",
            "function": {"name": "lookup_time", "arguments": '{"city":'},
        },
    ]
    second_calls: list[dict[str, Any]] = [
        {"index": 0, "function": {"arguments": '"Athens"}'}},
        {"index": 1, "function": {"arguments": '"Athens"}'}},
    ]
    if mode == "tool_deltas_missing_index":
        for call in first_calls:
            call.pop("index")
        for call in second_calls:
            call.pop("index")
    elif mode == "tool_name_late":
        first_calls[0]["function"].pop("name")
        second_calls[0]["function"]["name"] = "lookup_weather"
    elif mode == "tool_arguments_invalid_json":
        second_calls[0]["function"]["arguments"] = '"Athens"'
    elif mode == "tool_single_call":
        first_calls = first_calls[:1]
        second_calls = second_calls[:1]

    frames = [
        _sse_frame(_chunk(model, delta={"tool_calls": first_calls})),
        _sse_frame(_chunk(model, delta={"tool_calls": second_calls})),
        _sse_frame(_chunk(model, finish_reason="tool_calls")),
    ]
    if include_usage:
        frames.append(
            _sse_frame(
                _chunk(
                    model,
                    choices=[],
                    usage={
                        "prompt_tokens": 24,
                        "completion_tokens": 12,
                        "total_tokens": 36,
                    },
                )
            )
        )
    frames.append(_sse_frame("[DONE]"))
    return frames


def _is_tool_round_trip(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    assistant_turns = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
    ]
    tool_results = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    return len(assistant_turns) == 1 and len(tool_results) >= 1


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _MockHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> tuple[dict[str, Any] | None, bytes]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None, raw
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None, raw
        return (parsed if isinstance(parsed, dict) else None), raw

    def _record(self, body: dict[str, Any] | None, raw_body: bytes = b"") -> None:
        record = RequestRecord(
            method=self.command,
            path=self.path,
            headers={key.casefold(): value for key, value in self.headers.items()},
            body=body,
            raw_body=raw_body,
        )
        with self.server.state.request_lock:
            self.server.state.request_log.append(record)

    def _send(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        delay_seconds: float = 0.0,
    ) -> None:
        if delay_seconds:
            time.sleep(delay_seconds)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_json(
        self, status: int, payload: Any, *, delay_seconds: float = 0.0
    ) -> None:
        self._send(
            status,
            _json_bytes(payload),
            content_type="application/json",
            delay_seconds=delay_seconds,
        )

    def _send_sse(
        self,
        frames: list[bytes],
        *,
        incremental: bool,
        initial_delay_seconds: float = 0.0,
        delays_after: list[float] | None = None,
    ) -> None:
        if initial_delay_seconds:
            time.sleep(initial_delay_seconds)
        # Nebius gzips its SSE. A recorder that wraps the raw transport sees
        # compressed bytes with no data: lines in them.
        gzipped = self.headers.get("X-Mock-Mode") == "gzipped_sse"
        if gzipped:
            frames = [gzip.compress(b"".join(frames))]
            incremental = False
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        if gzipped:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Connection", "close")
        self.end_headers()
        if not incremental:
            self.wfile.write(b"".join(frames))
            self.wfile.flush()
            return
        for index, frame in enumerate(frames):
            self.wfile.write(frame)
            self.wfile.flush()
            if index != len(frames) - 1:
                delay = (
                    delays_after[index]
                    if delays_after is not None
                    else self.server.state.knobs.stream_chunk_delay_seconds
                )
                time.sleep(delay)

    def _mode_or_send_error(self) -> str | None:
        """Return a valid mode, or write the terminal 400 response for the caller."""
        mode = self.headers.get("X-Mock-Mode", "conforming")
        if mode in MOCK_MODES:
            return mode
        self._send_json(
            HTTPStatus.BAD_REQUEST,
            {
                "error": {
                    "type": "unknown_mock_mode",
                    "message": f"unknown X-Mock-Mode {mode!r}",
                    "valid_modes": list(MOCK_MODES),
                }
            },
        )
        return None

    def do_GET(self) -> None:
        self._record(None)
        parsed_url = urlsplit(self.path)
        if parsed_url.path == "/catalog.v2.json":
            requested_mode = parse_qs(parsed_url.query).get("mode", ["conforming"])[0]
            if requested_mode not in MOCK_MODES:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"type": "unknown_mock_mode"}},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                _catalog(invalid=requested_mode == "invalid_declared_catalog"),
            )
            return
        if parsed_url.path != "/v1/models":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"type": "not_found"}})
            return
        mode = self._mode_or_send_error()
        if mode is None:
            return
        if mode == "native_models_empty":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": []})
            return
        if mode == "models_bare_array":
            self._send_json(
                HTTPStatus.OK,
                [{"id": "mock/model", "object": "model", "type": "chat"}],
            )
            return
        if mode == "models_without_object_envelope":
            # pearlresearch.ai's real shape: valid ids, no object envelope.
            self._send_json(
                HTTPStatus.OK,
                {
                    "data": [{"id": "mock/model", "owned_by": "mock"}],
                    "pricing_source": "upstream",
                },
            )
            return
        if mode == "models_without_data_array":
            self._send_json(HTTPStatus.OK, {"object": "list"})
            return
        if mode == "native_noncanonical_ids":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {"id": "deepseek-chat", "object": "model", "owned_by": "mock"},
                        {
                            "id": "qwen3-4b:latest",
                            "object": "model",
                            "owned_by": "mock",
                        },
                    ],
                },
            )
            return
        model = {
            # Production denylist case: catalog advertises a route that returns HTML 403.
            "catalog_lists_403_model": "mock/catalog-forbidden",
            # Production denylist case: catalog advertises a route missing from chat serving.
            "catalog_lists_404_model": "mock/catalog-missing",
        }.get(mode, "mock/model")
        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "mock"}],
            },
        )

    def do_POST(self) -> None:
        body, raw_body = self._read_json_body()
        self._record(body, raw_body)
        if self.path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"type": "not_found"}})
            return
        mode = self._mode_or_send_error()
        if mode is None:
            return
        payload = body or {}
        model = str(payload.get("model") or "mock/model")

        # Production denylist case: an advertised model returns a provider HTML auth page.
        if mode == "catalog_lists_403_model":
            self._send(
                HTTPStatus.FORBIDDEN,
                b"<html><body>Forbidden</body></html>",
                content_type="text/html",
            )
            return
        # Production denylist case: an advertised model is absent from the chat API.
        if mode == "catalog_lists_404_model":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"type": "model_not_found", "message": "model not found"}},
            )
            return
        # Production probe case: queueing consumes the deadline, then capacity rejects.
        if mode == "queue_then_429":
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": {"type": "rate_limit_error", "message": "queue full"}},
                delay_seconds=self.server.state.knobs.queue_delay_seconds,
            )
            return
        # Production compatibility case: deterministic temperature is rejected.
        if mode == "rejects_temperature_zero" and payload.get("temperature") == 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "temperature=0 unsupported",
                    }
                },
            )
            return
        if mode == "rejects_max_tokens" and "max_tokens" in payload:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "max_tokens unsupported",
                    }
                },
            )
            return
        if mode == "rejects_tools" and payload.get("tools"):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "tools unsupported",
                    }
                },
            )
            return
        if mode == "structured_json_in_content_prose_in_reasoning" and (
            "response_format" in payload
        ):
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1_700_000_000,
                    "model": payload.get("model", "mock/model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"city": "Paris", "temperature_c": 15}',
                                "reasoning_content": "1. Analyze the request. 2. Emit JSON.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 12,
                        "total_tokens": 16,
                    },
                },
            )
            return
        if mode == "capability_probe_backend_down" and (
            "response_format" in payload or payload.get("temperature") == 0
        ):
            # The applicant case: an intermittently dead backend answers the
            # capability probe with 502. Nothing was learned about support.
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": {"type": "api_error"}})
            return
        if mode == "rejects_response_format" and "response_format" in payload:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "response_format unsupported",
                    }
                },
            )
            return
        messages = payload.get("messages")
        tool_round_trip = _is_tool_round_trip(messages)
        if mode == "rejects_empty_tool_content" and tool_round_trip:
            assert isinstance(messages, list)
            assistant = next(
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            if assistant.get("content") == "":
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": {
                            "type": "invalid_request",
                            "message": "assistant content may not be empty",
                        }
                    },
                )
                return
        # Production compatibility case: strict schemas reject translator-only extras.
        strict_fields = {"top_k", "thinking", "reasoning", "reasoning_effort"}
        if mode == "strict_extra_fields" and strict_fields.intersection(payload):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request",
                        "message": "extra fields rejected",
                    }
                },
            )
            return

        stream = payload.get("stream") is True
        if not stream:
            if tool_round_trip:
                self._send_json(
                    HTTPStatus.OK,
                    _completion(model, content="TOOLS COMPLETE"),
                )
                return
            response_format = payload.get("response_format")
            if isinstance(response_format, dict):
                format_type = response_format.get("type")
                if mode == "structured_non_json" and format_type == "json_object":
                    content = "not json"
                elif (
                    mode == "structured_schema_violation"
                    and format_type == "json_schema"
                ):
                    content = '{"answer":"NOPE","count":"two"}'
                else:
                    content = '{"answer":"PONG","count":2}'
                self._send_json(HTTPStatus.OK, _completion(model, content=content))
                return
            if mode == "empty_content":
                self._send_json(HTTPStatus.OK, _completion(model, content=""))
                return
            if mode == "bad_usage":
                self._send_json(HTTPStatus.OK, _completion(model, usage=_BAD_USAGE))
                return
            if mode == "wrong_pong":
                self._send_json(HTTPStatus.OK, _completion(model, content="NOPE"))
                return
            if mode == "wrong_model":
                self._send_json(
                    HTTPStatus.OK,
                    _completion(model, response_model="mock/other-model"),
                )
                return
            if mode == "unknown_finish_reason":
                self._send_json(
                    HTTPStatus.OK,
                    _completion(model, finish_reason="complete"),
                )
                return
            if mode == "chat_reasoning_shapes":
                self._send_json(HTTPStatus.OK, _chat_shape_completion(model))
                return
            self._send_json(HTTPStatus.OK, _completion(model))
            return

        if mode == "non_sse_200":
            # Production failure: stream=true ignored and a plain JSON completion is returned.
            self._send_json(HTTPStatus.OK, _completion(model))
            return

        role = _chunk(model, delta={"role": "assistant", "content": ""})
        finish = _chunk(model, finish_reason="stop")
        content_parts = [
            _chunk(model, delta={"content": "PO"}),
            _chunk(model, delta={"content": "NG"}),
        ]
        perf_request = (
            payload.get("max_tokens") == 512
            or payload.get("max_completion_tokens") == 512
        )
        output_tokens = 127 if mode == "perf_insufficient_output" else 256
        if not perf_request:
            output_tokens = 2
        usage = _chunk(
            model,
            choices=[],
            usage={
                "prompt_tokens": 18 if perf_request else 4,
                "completion_tokens": output_tokens,
                "total_tokens": output_tokens + (18 if perf_request else 4),
            },
        )
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        if payload.get("tools"):
            self._send_sse(
                _tool_stream_frames(
                    model,
                    mode=mode,
                    include_usage=include_usage,
                ),
                incremental=True,
            )
            return

        frames: list[bytes]
        space = mode != "no_space_framing"
        delay = 0.0
        delays_after: list[float] | None = None
        if mode == "finish_reason_only":
            # Production failure: successful SSE terminates without any visible token.
            frames = [_sse_frame(finish)]
            if include_usage:
                frames.append(_sse_frame(usage))
            frames.append(_sse_frame("[DONE]"))
        elif mode == "empty_content":
            frames = [
                _sse_frame(role),
                _sse_frame(_chunk(model, delta={"content": ""})),
                _sse_frame(finish),
                _sse_frame("[DONE]"),
            ]
        elif mode == "tool_deltas_missing_index":
            # Production failure: tool-call fragments cannot be correlated without index.
            frames = [
                _sse_frame(
                    _chunk(
                        model,
                        delta={
                            "tool_calls": [
                                {
                                    "id": "call_mock",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                    )
                ),
                _sse_frame(finish),
                _sse_frame("[DONE]"),
            ]
        elif mode == "tool_name_late":
            # Production failure: function.name arrives after the first delta for index 0.
            frames = [
                _sse_frame(
                    _chunk(
                        model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_mock",
                                    "type": "function",
                                    "function": {"arguments": "{"},
                                }
                            ]
                        },
                    )
                ),
                _sse_frame(
                    _chunk(
                        model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "lookup", "arguments": "}"},
                                }
                            ]
                        },
                    )
                ),
                _sse_frame(finish),
                _sse_frame("[DONE]"),
            ]
        elif mode == "midstream_error":
            # Production failure: a 200 stream changes into an in-band provider error.
            frames = [
                _sse_frame(role),
                _sse_frame(content_parts[0]),
                _sse_frame(
                    {"error": {"type": "provider_error", "message": "upstream reset"}}
                ),
            ]
        elif mode == "buffered_stream":
            # Production failure: all valid frames are withheld until completion.
            frames = [
                _sse_frame(role),
                *map(_sse_frame, content_parts),
                _sse_frame(finish),
            ]
            if include_usage:
                frames.append(_sse_frame(usage))
            frames.append(_sse_frame("[DONE]"))
            delay = self.server.state.knobs.buffered_stream_delay_seconds
        elif mode == "role_then_late_content":
            frames = [
                _sse_frame(role),
                b": ping\n\n",
                _sse_frame(_chunk(model, delta={"content": "PONG"})),
                _sse_frame(finish),
            ]
            if include_usage:
                frames.append(_sse_frame(usage))
            frames.append(_sse_frame("[DONE]"))
            standard_delay = self.server.state.knobs.stream_chunk_delay_seconds
            delays_after = [
                standard_delay,
                self.server.state.knobs.late_content_delay_seconds,
                standard_delay,
                standard_delay,
                standard_delay,
            ][: len(frames) - 1]
        elif mode == "delta_shapes":
            frames = [
                _sse_frame(role),
                _sse_frame(_shape_chunk(model)),
                _sse_frame(finish),
            ]
            if include_usage:
                frames.append(_sse_frame(usage))
            frames.append(_sse_frame("[DONE]"))
        elif mode == "bad_usage":
            # Production failure: token counts are non-integer and internally inconsistent.
            frames = [
                _sse_frame(role),
                *map(_sse_frame, content_parts),
                _sse_frame(finish),
                _sse_frame(_chunk(model, choices=[], usage=_BAD_USAGE)),
                _sse_frame("[DONE]"),
            ]
        else:
            frames = [_sse_frame(role, space=space)]
            frames.extend(_sse_frame(part, space=space) for part in content_parts)
            frames.append(_sse_frame(finish, space=space))
            if include_usage and mode != "ignores_include_usage":
                # `ignores_include_usage` deliberately omits this requested frame.
                frames.append(_sse_frame(usage, space=space))
            if mode != "no_done_sentinel":
                # `no_done_sentinel` deliberately closes before this marker.
                frames.append(_sse_frame("[DONE]", space=space))

        self._send_sse(
            frames,
            incremental=mode != "buffered_stream",
            initial_delay_seconds=delay,
            delays_after=delays_after,
        )


class MockOpenAIServer:
    """Own a background server and expose its URL, knobs, and request log."""

    def __init__(self) -> None:
        self._httpd = _MockHTTPServer(("127.0.0.1", 0))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._started = False

    @property
    def base_url(self) -> str:
        address = self._httpd.server_address
        host = address[0]
        if isinstance(host, bytes):
            host = host.decode()
        return f"http://{host}:{address[1]}"

    @property
    def request_log(self) -> list[RequestRecord]:
        return self._httpd.state.request_log

    @property
    def knobs(self) -> ModeKnobs:
        return self._httpd.state.knobs

    def clear_requests(self) -> None:
        with self._httpd.state.request_lock:
            self._httpd.state.request_log.clear()

    def reset(self) -> None:
        self.clear_requests()
        self._httpd.state.knobs = ModeKnobs()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("mock server already started")
        self._thread.start()
        self._started = True

    def close(self) -> None:
        if self._started:
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._started:
            self._thread.join(timeout=2)
        self._started = False
