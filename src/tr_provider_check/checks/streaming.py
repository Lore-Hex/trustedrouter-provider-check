"""Tier 4: streaming behavior at the enclave's OpenAI-compatible boundary."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

import httpx

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.checks.catalog import CatalogEvidence
from tr_provider_check.contract import (
    PONG_PROMPT,
    _classify,
    _elapsed_ms_with_clock,
    _observe_provider_stream,
    _response_error,
    _sse_line_payload,
    model_deadlines,
    model_deadlines_declared,
)
from tr_provider_check.http import GatewayClient, probe_verdict
from tr_provider_check.report import CheckResult, CheckStatus, check_result

Clock = Callable[[], float]


class _DecodedRecordingStream(httpx.AsyncByteStream):
    """Decode once with httpx, then feed identical bytes to every consumer."""

    def __init__(
        self, response: httpx.Response, *, started: float, clock: Clock
    ) -> None:
        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise TypeError("async streaming response exposed a synchronous body")
        self._decoded_response = httpx.Response(
            status_code=response.status_code,
            headers=httpx.Headers(response.headers),
            stream=response.stream,
            request=response.request,
        )
        self._started = started
        self._clock = clock
        self.records: list[tuple[int, bytes]] = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._decoded_response.aiter_bytes():
            self.records.append(
                (_elapsed_ms_with_clock(self._started, self._clock), chunk)
            )
            yield chunk

    async def aclose(self) -> None:
        await self._decoded_response.aclose()


def _record_decoded_stream(
    response: httpx.Response, *, started: float, clock: Clock
) -> _DecodedRecordingStream:
    """Install one decoded stream shared by the observer and timing checks."""

    recorder = _DecodedRecordingStream(response, started=started, clock=clock)
    response.stream = recorder
    # The replacement stream is already decoded. Prevent response.aiter_bytes(),
    # which the vendored observer calls, from applying the content codec twice.
    for header in ("content-encoding", "content-length"):
        if header in response.headers:
            del response.headers[header]
    return recorder


def _skip(check_id: str, assertion: str, reason: str) -> CheckResult:
    return check_result(
        id=check_id,
        tier=4,
        status="skip",
        assertion=assertion,
        measured={"reason": reason},
        contract_ref=(
            "enclave-go/internal/llm/byok.go; "
            "enclave-go/internal/llm/stream_translate.go; "
            "enclave-go/cmd/enclave/provider_stream.go"
        ),
        marketplace_bullet=assertion,
        remediation="Fix the prerequisite streaming assertion, then rerun this check.",
    )


def _wire_framing(body: bytes, content_type: str) -> tuple[bool, int, list[str]]:
    text = body.decode("utf-8", "replace")
    lines = text.splitlines()
    data_lines = [line for line in lines if line.startswith("data: ")]
    invalid_data_lines = [
        line
        for line in lines
        if line.startswith("data:") and not line.startswith("data: ")
    ]
    media_type = content_type.partition(";")[0].strip().casefold()
    return (
        media_type == "text/event-stream"
        and bool(data_lines)
        and not invalid_data_lines,
        len(data_lines),
        invalid_data_lines,
    )


def _first_tool_delta_milliseconds(
    records: list[tuple[int, bytes]],
) -> int | None:
    """Find a tool-call delta using the vendored SSE JSON payload extractor."""

    tail = ""
    for elapsed_milliseconds, chunk in records:
        tail += chunk.decode("utf-8", "ignore")
        lines = tail.split("\n")
        tail = lines.pop()
        for line in lines:
            payload = _sse_line_payload(line)
            if payload is None:
                continue
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("tool_calls"):
                    return elapsed_milliseconds
    if tail:
        payload = _sse_line_payload(tail)
        if payload is not None:
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("tool_calls"):
                    return records[-1][0] if records else None
    return None


def _dependent_stream_results(reason: str) -> list[CheckResult]:
    return [
        _skip(
            "stream.sse-framing",
            assertion_for("stream.sse-framing"),
            reason,
        ),
        _skip(
            "stream.done",
            assertion_for("stream.done"),
            reason,
        ),
        _skip(
            "stream.usage",
            assertion_for("stream.usage"),
            reason,
        ),
        _skip(
            "stream.first-delta",
            assertion_for("stream.first-delta"),
            reason,
        ),
        _skip(
            "stream.incremental-delivery",
            assertion_for("stream.incremental-delivery"),
            reason,
        ),
    ]


async def run_streaming_checks(
    client: GatewayClient,
    model: str,
    *,
    first_token_budget_seconds: float | None = None,
    evidence: CatalogEvidence | None = None,
    clock: Clock = time.perf_counter,
) -> list[CheckResult]:
    """Run Tier 4 against the vendored production stream observer.

    ``enclave-go/internal/llm/byok.go`` requires HTTP 200 before translating,
    requests ``stream_options.include_usage``, and exposes the response body to
    ``stream_translate.go``. That translator recognizes only ``data: `` lines,
    consumes content/reasoning/tool deltas incrementally, and otherwise turns a
    non-SSE 200 into a silent empty completion. ``cmd/enclave/provider_stream.go``
    starts its first-byte failover clock only when translated output is written;
    role-only chunks and SSE comments therefore cannot satisfy the budget. A
    200 stream error is already downstream of that byte gate and cannot be
    retried safely.
    """

    reliability = evidence.reliability(model) if evidence is not None else None
    raw_declared_budget = (
        reliability.get("first_token_timeout_seconds")
        if isinstance(reliability, dict)
        else None
    )
    declared_budget = (
        float(raw_declared_budget)
        if isinstance(raw_declared_budget, (int, float))
        and not isinstance(raw_declared_budget, bool)
        else None
    )
    if first_token_budget_seconds is not None:
        budget_seconds = first_token_budget_seconds
        budget_source = "argument"
    elif declared_budget is not None:
        budget_seconds = model_deadlines_declared(
            model, declared_first_token_seconds=declared_budget
        ).first_token_seconds
        budget_source = "declared"
    else:
        budget_seconds = model_deadlines(model).first_token_seconds
        budget_source = "fallback"
    if budget_seconds <= 0:
        raise ValueError("first_token_budget_seconds must be positive")

    started = clock()
    try:
        stream_context = client.stream_chat(
            model=model,
            prompt=PONG_PROMPT,
            temperature=None,
        )
        async with stream_context as response:
            status_code = response.status_code
            if status_code != 200:
                await response.aread()
                error_type, error_status, error_message = _response_error(response)
                verdict = _classify(status_code, response.text[:4096])
                status, reason = probe_verdict(status_code, declared=True)
                return [
                    check_result(
                        id="stream.response-status",
                        tier=4,
                        status=status,
                        assertion=assertion_for("stream.response-status"),
                        measured={
                            "http_status": status_code,
                            "route_verdict": verdict,
                            "reason": reason,
                        },
                        contract_ref="enclave-go/internal/llm/byok.go (non-200 is upstreamHTTPError before translation)",
                        marketplace_bullet="Streaming requests are accepted for the advertised route.",
                        remediation="Make this model available on POST /chat/completions; use 429/503 with Retry-After only for genuinely transient capacity, and a precise 4xx for permanent request errors.",
                        error_type=error_type,
                        error_status=error_status,
                        error_message=error_message,
                    ),
                    check_result(
                        id="stream.error-signaling",
                        tier=4,
                        status="pass",
                        assertion=assertion_for("stream.error-signaling"),
                        measured={
                            "http_status": status_code,
                            "error_before_stream": True,
                        },
                        contract_ref="enclave-go/internal/llm/byok.go; enclave-go/cmd/enclave/provider_stream.go (retry only before bytes)",
                        marketplace_bullet="Provider failures are visible before the gateway commits a successful stream.",
                        remediation="No error-signaling change is required; fix the separately reported route failure.",
                    ),
                    *_dependent_stream_results("stream request returned non-200"),
                ]

            recorder = _record_decoded_stream(response, started=started, clock=clock)
            observation = await _observe_provider_stream(
                response,
                started=started,
                clock=clock,
            )
            wire = b"".join(chunk for _, chunk in recorder.records)
            first_tool_ms = _first_tool_delta_milliseconds(recorder.records)
            transport_chunk_count = len(recorder.records)
            content_type = response.headers.get("content-type", "")
    except httpx.HTTPError as request_error:
        status, reason = probe_verdict(None, declared=True)
        return [
            check_result(
                id="stream.response-status",
                tier=4,
                status=status,
                assertion=assertion_for("stream.response-status"),
                measured={
                    "http_status": None,
                    "error_type": request_error.__class__.__name__,
                    "reason": reason,
                },
                contract_ref="enclave-go/internal/llm/byok.go (http client failure before translation)",
                marketplace_bullet="Streaming requests are accepted for the advertised route.",
                remediation="Make the endpoint reachable over HTTP and keep the connection open for the SSE response.",
                error_type=request_error.__class__.__name__,
            ),
            check_result(
                id="stream.error-signaling",
                tier=4,
                status=status,
                assertion=assertion_for("stream.error-signaling"),
                measured={"http_status": None, "error_before_stream": False},
                contract_ref="enclave-go/internal/llm/byok.go; enclave-go/cmd/enclave/provider_stream.go",
                marketplace_bullet="Provider failures are visible before the gateway commits a successful stream.",
                remediation="Return a non-2xx OpenAI-style HTTP error instead of resetting or abandoning the connection.",
                error_type=request_error.__class__.__name__,
            ),
            *_dependent_stream_results(
                "stream transport failed before response headers"
            ),
        ]

    results = [
        check_result(
            id="stream.response-status",
            tier=4,
            status="pass",
            assertion=assertion_for("stream.response-status"),
            measured={"http_status": 200},
            contract_ref="enclave-go/internal/llm/byok.go (HTTP 200 enters stream translation)",
            marketplace_bullet="Streaming requests are accepted for the advertised route.",
            remediation="Return HTTP 200 only when an SSE completion can be streamed.",
        )
    ]

    framing_ok, data_line_count, invalid_data_lines = _wire_framing(wire, content_type)
    wire_unobserved = not wire
    genuinely_empty = (
        wire_unobserved
        and observation.ttfb_milliseconds is None
        and observation.first_token_milliseconds is None
        and first_tool_ms is None
    )
    capture_failed = wire_unobserved and not genuinely_empty
    framing_status: CheckStatus = (
        "pass"
        if framing_ok
        else "fail"
        if genuinely_empty or not capture_failed
        else "warn"
    )
    results.append(
        check_result(
            id="stream.sse-framing",
            tier=4,
            status=framing_status,
            assertion=assertion_for("stream.sse-framing"),
            measured={
                "content_type": content_type,
                "data_line_count": data_line_count,
                "invalid_data_line_count": len(invalid_data_lines),
                "wire_unobserved": wire_unobserved,
                "genuinely_empty": genuinely_empty,
                "capture_failed": capture_failed,
                **(
                    {
                        "reason": (
                            "HTTP 200 carried no response bytes or content deltas"
                            if genuinely_empty
                            else "the observer populated from the response but no "
                            "decoded bytes reached the recorder; framing is unknown"
                        )
                    }
                    if wire_unobserved
                    else {}
                ),
            },
            contract_ref="enclave-go/internal/llm/stream_translate.go (accepts only lines prefixed 'data: ')",
            marketplace_bullet="Streaming responses use text/event-stream and enclave-readable data lines.",
            remediation="For stream=true, return Content-Type: text/event-stream and frame each JSON event exactly as 'data: <json>\\n\\n'. Do not return a plain JSON 200 body.",
            error_type="empty_stream" if framing_status == "fail" else None,
        )
    )

    stream_error = observation.stream_error
    results.append(
        check_result(
            id="stream.error-signaling",
            tier=4,
            status="fail" if stream_error is not None else "pass",
            assertion=assertion_for("stream.error-signaling"),
            measured={
                "http_status": 200,
                "midstream_error_type": stream_error[0] if stream_error else None,
                "midstream_error_status": stream_error[1] if stream_error else None,
            },
            contract_ref="enclave-go/internal/llm/byok.go; enclave-go/cmd/enclave/provider_stream.go (a failure after bytes cannot fail over)",
            marketplace_bullet="Provider failures are visible before the gateway commits a successful stream.",
            remediation='Validate and reserve capacity before sending HTTP 200 or any SSE bytes; return the OpenAI-style error with a non-2xx HTTP status instead of embedding data: {"error": ...} in the stream.',
            error_type="stream_error" if stream_error is not None else None,
            error_status=stream_error[1] if stream_error else None,
            error_message=stream_error[2] if stream_error else None,
        )
    )

    if not framing_ok:
        results.extend(
            [
                _skip(
                    "stream.done",
                    assertion_for("stream.done"),
                    "SSE framing is not enclave-readable",
                ),
                _skip(
                    "stream.usage",
                    assertion_for("stream.usage"),
                    "SSE framing is not enclave-readable",
                ),
                _skip(
                    "stream.first-delta",
                    assertion_for("stream.first-delta"),
                    "SSE framing is not enclave-readable",
                ),
                _skip(
                    "stream.incremental-delivery",
                    assertion_for("stream.incremental-delivery"),
                    "SSE framing is not enclave-readable",
                ),
            ]
        )
        return results

    content_or_reasoning_ms = observation.first_token_milliseconds
    first_token_ms = (
        first_tool_ms
        if content_or_reasoning_ms is None
        else content_or_reasoning_ms
        if first_tool_ms is None
        else min(content_or_reasoning_ms, first_tool_ms)
    )
    budget_ms = int(budget_seconds * 1000)
    first_delta_ok = first_token_ms is not None and first_token_ms <= budget_ms
    results.append(
        check_result(
            id="stream.first-delta",
            tier=4,
            status="pass" if first_delta_ok else "fail",
            assertion=assertion_for("stream.first-delta"),
            measured={
                "ttfb_milliseconds": observation.ttfb_milliseconds,
                "first_delta_milliseconds": first_token_ms,
                "first_content_or_reasoning_delta_milliseconds": (
                    content_or_reasoning_ms
                ),
                "first_tool_delta_milliseconds": first_tool_ms,
                "budget_milliseconds": budget_ms,
                "budget_source": budget_source,
            },
            contract_ref="enclave-go/internal/llm/stream_translate.go; enclave-go/cmd/enclave/provider_stream.go (translated first-byte budget)",
            marketplace_bullet="A role-only chunk or SSE ping does not mask late model output.",
            remediation="Flush a non-empty content, reasoning, or tool delta before the advertised first-token deadline. Role-only chunks and ': ping' comments are keepalives, not model output.",
            error_type=None if first_delta_ok else "synthetic_deadline_failure",
        )
    )

    if stream_error is not None:
        results.extend(
            [
                _skip(
                    "stream.done",
                    assertion_for("stream.done"),
                    "the HTTP 200 stream carried an in-band error",
                ),
                _skip(
                    "stream.usage",
                    assertion_for("stream.usage"),
                    "the HTTP 200 stream carried an in-band error",
                ),
                _skip(
                    "stream.incremental-delivery",
                    assertion_for("stream.incremental-delivery"),
                    "the HTTP 200 stream carried an in-band error",
                ),
            ]
        )
        return results

    done_ok = any(line == b"data: [DONE]" for line in wire.splitlines())
    results.append(
        check_result(
            id="stream.done",
            tier=4,
            status="pass" if done_ok else "warn",
            assertion=assertion_for("stream.done"),
            measured={"done_sentinel": done_ok},
            contract_ref="enclave-go/internal/llm/stream_translate.go (DONE terminates scanning; EOF is tolerated)",
            marketplace_bullet="Successful streams carry the conventional terminal [DONE] sentinel.",
            remediation="After the final completion and usage chunks, emit exactly 'data: [DONE]\\n\\n' before closing the response.",
            error_type=None if done_ok else "incomplete_stream",
        )
    )

    usage = observation.usage
    usage_ok = any(
        value > 0
        for value in (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens)
    )
    results.append(
        check_result(
            id="stream.usage",
            tier=4,
            status="pass" if usage_ok else "fail",
            assertion=assertion_for("stream.usage"),
            measured={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
            contract_ref="enclave-go/internal/llm/byok.go (always requests include_usage); enclave-go/internal/llm/stream_translate.go (usage in any chunk)",
            marketplace_bullet="Successful streams return provider-measured token usage for settlement.",
            remediation="Honor stream_options.include_usage=true and include a usage object in any SSE chunk, conventionally a final choices: [] chunk before [DONE].",
            error_type=None if usage_ok else "stream_error",
        )
    )

    if first_token_ms is None:
        results.append(
            _skip(
                "stream.incremental-delivery",
                assertion_for("stream.incremental-delivery"),
                "no non-empty content or reasoning delta arrived",
            )
        )
    else:
        incremental_ok = (
            transport_chunk_count > 1
            and first_token_ms < observation.elapsed_milliseconds
        )
        results.append(
            check_result(
                id="stream.incremental-delivery",
                tier=4,
                status="pass" if incremental_ok else "warn",
                assertion=assertion_for("stream.incremental-delivery"),
                measured={
                    "first_delta_milliseconds": first_token_ms,
                    "last_delta_milliseconds": observation.last_token_milliseconds,
                    "elapsed_milliseconds": observation.elapsed_milliseconds,
                    "transport_chunk_count": transport_chunk_count,
                },
                contract_ref="enclave-go/internal/llm/stream_translate.go; enclave-go/cmd/enclave/provider_stream.go (translated writes drive first-byte behavior)",
                marketplace_bullet="SSE output is delivered incrementally instead of as a completed buffered body.",
                remediation="Disable reverse-proxy and application response buffering, flush each SSE event, and avoid accumulating the full model answer before the first write.",
                error_type=None if incremental_ok else "stream_error",
            )
        )
    return results
