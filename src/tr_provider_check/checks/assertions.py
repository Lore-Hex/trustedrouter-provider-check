"""Stable public assertion text keyed by report check id."""

from __future__ import annotations

CHECK_ASSERTIONS: dict[str, str] = {
    "catalog.native-model-discovery": (
        "GET /models advertises at least one unique, non-empty native model id"
    ),
    "catalog.declared-v2": (
        "declared marketplace catalog conforms to the published v2 schema and "
        "vendored id/decimal/field rules"
    ),
    "callability.advertised-models": (
        "advertised native chat models are callable; permanent route failures are "
        "DEAD and transient health failures are FLAKY"
    ),
    "chat.unavailable": "Tier 3 has a selected native model to inspect",
    "chat.non-empty": (
        "a successful response contains non-empty assistant or reasoning text"
    ),
    "chat.pong": (
        "PONG_PROMPT produces a response matched by the vendored extractor and matcher"
    ),
    "chat.usage": (
        "usage counts are non-negative integers and total equals input plus output"
    ),
    "chat.model": "the response model field identifies the requested run",
    "chat.finish-reason": (
        "finish_reason truthfully uses stop, length, tool_calls, or content_filter"
    ),
    "chat.temperature-zero": ("temperature: 0 is accepted when the gateway sends it"),
    "chat.forwarded-fields": (
        "gateway-forwarded top_k, thinking, reasoning, and reasoning_effort fields "
        "are tolerated"
    ),
    "chat.max-token-spelling": (
        "the provider accepts the max-token spelling the enclave selects for this "
        "provider/model"
    ),
    "stream.unavailable": "Tier 4 has a selected native model to inspect",
    "stream.response-status": "the selected model accepts a streaming request",
    "stream.sse-framing": ("the HTTP 200 body uses enclave-readable SSE data framing"),
    "stream.error-signaling": (
        "errors are HTTP failures before stream bytes, never data: {error} after HTTP 200"
    ),
    "stream.done": "the successful SSE stream ends with data: [DONE]",
    "stream.usage": ("stream_options.include_usage produces an actual usage chunk"),
    "stream.first-delta": (
        "the first non-empty content, reasoning, or tool delta arrives within budget"
    ),
    "stream.incremental-delivery": (
        "the provider does not withhold the whole successful stream before delivery"
    ),
    "tools.parallel-deltas": (
        "a forced parallel tool stream emits well-formed indexed calls whose first "
        "deltas have function.name and whose concatenated arguments are valid JSON"
    ),
    "tools.round-trip": (
        'a replayed assistant tool turn uses content:"" plus role:"tool" results and '
        "receives a normal non-empty assistant completion"
    ),
    "structured.json-object": (
        "response_format type=json_object returns assistant content that parses as JSON"
    ),
    "structured.json-schema": (
        "response_format type=json_schema returns JSON that validates against the "
        "supplied schema"
    ),
    "perf.production-benchmark": (
        "advisory samples expose TTFB, TTFT, and effective throughput using "
        "production's request-start denominator and vendored leaderboard score"
    ),
}


def assertion_for(check_id: str) -> str:
    """Return the one public assertion string for ``check_id``."""

    return CHECK_ASSERTIONS[check_id]


__all__ = ["CHECK_ASSERTIONS", "assertion_for"]
