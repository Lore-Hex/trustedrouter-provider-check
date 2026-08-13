"""Vendored TrustedRouter provider contract.

Every symbol below is copied from TrustedRouter's control plane
(https://github.com/Lore-Hex/quill-router) rather than reimplemented, so that
this suite asserts the same contract production asserts. A reimplementation
that merely looked equivalent would drift, and a provider would pass here and
fail in production -- which is the one outcome that makes the tool worthless.

Copies are byte-for-byte. ``tests/test_contract_snapshot.py`` recomputes the
normalized hashes exported by production, accounts explicitly for every
provenance-claimed symbol, and compares behavioral replays against the
snapshot. The single source-hash exception is ``model_deadlines``, whose
production body reaches into a private catalog module; it is split into the
two functions below and is listed, with its reason, in that test's ADAPTED
allowlist.

WHICH production version these correspond to
--------------------------------------------
The authority is ``contract_version`` in ``data/contract_snapshot.json``: a
hash over the exported contract, so it identifies the symbols by content and
cannot go stale or dangle. ``UPSTREAM_COMMIT`` below is a convenience pointer
for a human who wants to read the originals; per-symbol comments name the
upstream path. quill-router contains a parity gate that recomputes the snapshot
from live production symbols. Before this repository is published and wired
into automatic CI, run that gate explicitly with
``PROVIDER_CHECK_REPO_PATH=<checkout>`` as documented in the README.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from dataclasses import field as dataclass_field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

import httpx

#: Convenience pointer for a human reading the originals. The binding
#: identity is ``contract_version`` in the snapshot, not this string: a
#: content hash cannot dangle, whereas a branch commit disappears the moment
#: it is squash-merged -- which is exactly how the previous pin rotted.
UPSTREAM_REPOSITORY = "https://github.com/Lore-Hex/quill-router"
UPSTREAM_COMMIT = "e3c4930fc2b0b5d10bd9bc6f4a22fd0a0e60952a"


# Public-package projection of the fields consumed by the vendored leaderboard.
@dataclass
class ProviderBenchmarkSample:
    provider: str
    model: str
    source: str
    status: str
    error_type: str | None
    error_status: int | None
    error_message: str | None
    created_at: str
    output_tokens: int
    elapsed_milliseconds: int | None
    speed_tokens_per_second: float | None
    first_token_milliseconds: int | None
    ttfb_milliseconds: int | None


# Source: src/trusted_router/synthetic/components.py
MONITOR_CONFIGURATION_ERROR_TYPES = frozenset(
    {
        "monitor_account_unavailable",
        "monitor_workspace_paused",
    }
)


# Source: src/trusted_router/synthetic/components.py
def is_router_origin_error(error_type: str | None) -> bool:
    """Return whether a benchmark failure happened before provider invocation."""
    return bool(
        error_type
        and (error_type in MONITOR_CONFIGURATION_ERROR_TYPES or error_type.startswith("router_"))
    )


# Source: src/trusted_router/provider_reliability.py
class FailureOwner(StrEnum):
    NONE = "none"
    PROVIDER = "provider"
    TRUSTEDROUTER = "trustedrouter"
    CONFIGURATION = "configuration"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


# Source: src/trusted_router/provider_reliability.py
class FailureClass(StrEnum):
    SUCCESS = "success"
    PROVIDER_CAPACITY = "provider_capacity"
    PROVIDER_INTERNAL = "provider_internal"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_STREAM = "provider_stream"
    TRUSTEDROUTER_CAPACITY = "trustedrouter_capacity"
    ROUTER_FAULT = "router_fault"
    CUSTOMER_QUOTA = "customer_quota"
    PROVIDER_AUTH_CONFIG = "provider_auth_config"
    UNSUPPORTED_ROUTE = "unsupported_route"
    PROBE_CONFIG = "probe_config_error"
    UNKNOWN = "unknown"


# Source: src/trusted_router/provider_reliability.py
@dataclass(frozen=True)
class FailureAttribution:
    owner: FailureOwner
    failure_class: FailureClass
    counts_toward_provider_availability: bool
    capacity_rejected: bool = False


# Source: src/trusted_router/provider_reliability.py
_ACCOUNT_QUOTA_MARKERS = (
    "account quota",
    "billing quota",
    "credit balance",
    "credits exhausted",
    "insufficient credit",
    "insufficient funds",
    "monthly spend",
    "payment required",
    "quota exceeded for quota metric",
)

# Source: src/trusted_router/provider_reliability.py
_CUSTOMER_QUOTA_TYPES = frozenset(
    {
        "credit_limit_exceeded",
        "insufficient_credits",
        "key_limit_exceeded",
        "rate_limit_exceeded",
        "workspace_limit_exceeded",
    }
)

# Source: src/trusted_router/provider_reliability.py
_CONFIG_TYPES = frozenset(
    {
        "bad_request",
        "invalid_request",
        "invalid_request_error",
        "model_not_available",
        "model_not_found",
        "not_found",
        "not_supported",
        "probe_config_error",
        "provider_auth_config",
        "unsupported",
        "unsupported_model",
        "unsupported_provider",
        "unsupported_route",
    }
)

# Source: src/trusted_router/provider_reliability.py
_TIMEOUT_MARKERS = ("timeout", "timed_out", "deadline", "readtimeout", "connecttimeout")

# Source: src/trusted_router/provider_reliability.py
_STREAM_MARKERS = (
    "empty_stream",
    "stream_error",
    "stream_interrupted",
    "incomplete_stream",
)


# Source: src/trusted_router/provider_reliability.py
def classify_provider_failure(
    *,
    status: str,
    error_type: str | None,
    error_status: int | None,
    error_message: str | None = None,
) -> FailureAttribution:
    """Classify one metadata-only provider observation.

    An upstream account quota is owned by TrustedRouter because provisioning
    enough provider capacity is our responsibility. A generic 429/503 remains
    provider capacity unless the provider explicitly identifies an account
    quota. Router-origin errors never lower a provider's availability.
    """

    if status == "success":
        return FailureAttribution(FailureOwner.NONE, FailureClass.SUCCESS, True)

    kind = (error_type or "").strip().casefold()
    message = (error_message or "").strip().casefold()

    if kind in _CUSTOMER_QUOTA_TYPES:
        return FailureAttribution(
            FailureOwner.CUSTOMER,
            FailureClass.CUSTOMER_QUOTA,
            False,
        )
    if is_router_origin_error(error_type):
        return FailureAttribution(
            FailureOwner.TRUSTEDROUTER,
            FailureClass.ROUTER_FAULT,
            False,
        )
    if status == "unsupported" or kind in _CONFIG_TYPES or error_status in {400, 404, 422}:
        failure_class = (
            FailureClass.PROVIDER_AUTH_CONFIG
            if error_status in {401, 403} or kind == "provider_auth_config"
            else FailureClass.PROBE_CONFIG
            if kind == "probe_config_error"
            else FailureClass.UNSUPPORTED_ROUTE
        )
        return FailureAttribution(FailureOwner.CONFIGURATION, failure_class, False)
    if error_status in {401, 403}:
        return FailureAttribution(
            FailureOwner.CONFIGURATION,
            FailureClass.PROVIDER_AUTH_CONFIG,
            False,
        )
    if error_status == 402 or any(marker in message for marker in _ACCOUNT_QUOTA_MARKERS):
        return FailureAttribution(
            FailureOwner.TRUSTEDROUTER,
            FailureClass.TRUSTEDROUTER_CAPACITY,
            False,
            capacity_rejected=True,
        )
    if error_status in {429, 503, 529} or "overload" in kind or "capacity" in kind:
        return FailureAttribution(
            FailureOwner.PROVIDER,
            FailureClass.PROVIDER_CAPACITY,
            True,
            capacity_rejected=True,
        )
    if any(marker in kind for marker in _TIMEOUT_MARKERS):
        return FailureAttribution(
            FailureOwner.PROVIDER,
            FailureClass.PROVIDER_TIMEOUT,
            True,
        )
    if any(marker in kind for marker in _STREAM_MARKERS):
        return FailureAttribution(
            FailureOwner.PROVIDER,
            FailureClass.PROVIDER_STREAM,
            True,
        )
    if error_status is not None and 500 <= error_status <= 599:
        return FailureAttribution(
            FailureOwner.PROVIDER,
            FailureClass.PROVIDER_INTERNAL,
            True,
        )
    return FailureAttribution(FailureOwner.UNKNOWN, FailureClass.UNKNOWN, False)


# Source: src/trusted_router/provider_reliability.py
@dataclass(frozen=True)
class ModelDeadlines:
    first_token_seconds: float
    completion_seconds: float


# Source: src/trusted_router/provider_reliability.py
_SLOW_REASONING_MARKERS = (
    "reasoning",
    "thinking",
    "deep-research",
    "deepseek-r",
    "gpt-5.5",
    "gpt-5.6",
    "glm-5",
    "opus",
    "/o1",
    "/o3",
    "/o4",
)

# Source: src/trusted_router/provider_reliability.py
_FAST_MARKERS = ("flash-lite", "haiku", "cerebras/", "groq/", "fast")


# Adapted from: src/trusted_router/provider_reliability.py
def model_deadlines_fallback(
    model_id: str,
    *,
    default_first_token_seconds: float = 20.0,
) -> ModelDeadlines:
    """Apply the byte-identical non-catalog fallback deadline policy."""
    if default_first_token_seconds <= 0:
        raise ValueError("default_first_token_seconds must be positive")
    normalized = model_id.casefold()
    first_token = float(default_first_token_seconds)
    if any(marker in normalized for marker in _SLOW_REASONING_MARKERS):
        first_token = max(first_token, 45.0)
    elif any(marker in normalized for marker in _FAST_MARKERS):
        first_token = min(first_token, 15.0)
    first_token = min(max(first_token, 5.0), 90.0)
    return ModelDeadlines(
        first_token_seconds=first_token,
        completion_seconds=min(max(first_token * 4, 30.0), 300.0),
    )


# Adapted from: src/trusted_router/provider_reliability.py
def model_deadlines_declared(
    model: str,
    *,
    declared_first_token_seconds: float | None = None,
    declared_completion_seconds: float | None = None,
) -> ModelDeadlines:
    """Apply production clamps to Provider Contract v2 declared budgets."""
    if declared_first_token_seconds is None:
        return model_deadlines_fallback(model)
    return ModelDeadlines(
        first_token_seconds=min(max(declared_first_token_seconds, 5.0), 300.0),
        completion_seconds=min(
            max(
                declared_completion_seconds or declared_first_token_seconds * 4,
                30.0,
            ),
            900.0,
        ),
    )


# Adapted from: src/trusted_router/provider_reliability.py
def model_deadlines(
    model_id: str,
    *,
    default_first_token_seconds: float = 20.0,
) -> ModelDeadlines:
    """Use fallback policy; declared provider budgets require the explicit API."""
    return model_deadlines_fallback(
        model_id,
        default_first_token_seconds=default_first_token_seconds,
    )


# Source: src/trusted_router/synthetic/probes.py
PONG_PROMPT = "reply exactly PONG"


# Source: src/trusted_router/synthetic/probes.py
@dataclass
class _StreamUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


# Source: src/trusted_router/synthetic/probes.py
@dataclass
class _StreamObservation:
    ttfb_milliseconds: int | None = None
    first_token_milliseconds: int | None = None
    last_token_milliseconds: int | None = None
    elapsed_milliseconds: int = 0
    finish_reason: str | None = None
    stream_error: tuple[str, int | None, str | None] | None = None
    usage: _StreamUsage = dataclass_field(default_factory=_StreamUsage)


# Source: src/trusted_router/synthetic/probes.py
def _sse_line_payload(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# Source: src/trusted_router/synthetic/probes.py
def _sse_line_has_content(line: str) -> bool:
    """True if an SSE `data:` line carries a visible content/reasoning delta."""
    data = _sse_line_payload(line)
    if data is None:
        return False
    for choice in data.get("choices") or []:
        delta = choice.get("delta") or {}
        if (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thinking")
            or delta.get("text")
            or delta.get("output_text")
        ):
            return True
        message = choice.get("message") or {}
        if (
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
            or message.get("text")
        ):
            return True
        if choice.get("text"):
            return True
    return False


# Source: src/trusted_router/synthetic/probes.py
def _sse_line_error(line: str) -> tuple[str, int | None, str | None] | None:
    """Return an OpenAI-style SSE error if the data line carries one."""
    data = _sse_line_payload(line)
    if data is None:
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    error_type = str(error.get("type") or "provider_error")
    message = str(error.get("message") or "") or None
    source = str(error.get("source") or "") or None
    status_raw = error.get("status") or error.get("code") or error.get("status_code")
    status: int | None
    try:
        status = int(status_raw) if status_raw is not None else None
    except (TypeError, ValueError):
        status = None
    return _rotation_error_type(error_type, status, message, source=source), status, message


# Source: src/trusted_router/synthetic/probes.py
def _sse_line_finish_reason(line: str) -> str | None:
    """Return the first choice finish reason from an SSE data line, if any."""
    data = _sse_line_payload(line)
    if data is None:
        return None
    for choice in data.get("choices") or []:
        reason = choice.get("finish_reason")
        if reason:
            return str(reason)
    return None


# Source: src/trusted_router/synthetic/probes.py
def _sse_line_usage(line: str) -> _StreamUsage | None:
    data = _sse_line_payload(line)
    if data is None:
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    return _StreamUsage(
        input_tokens=_first_int(usage, "prompt_tokens", "input_tokens"),
        output_tokens=_first_int(usage, "completion_tokens", "output_tokens"),
        reasoning_tokens=_first_int(completion_details, "reasoning_tokens"),
    )


# Source: src/trusted_router/synthetic/probes.py
def _first_int(values: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


# Source: src/trusted_router/synthetic/probes.py
async def _observe_provider_stream(
    response: httpx.Response,
    *,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> _StreamObservation:
    observation = _StreamObservation()
    tail = ""

    def observe_line(line: str, now_milliseconds: int) -> None:
        finish_reason = _sse_line_finish_reason(line)
        if finish_reason is not None:
            observation.finish_reason = finish_reason
        stream_error = _sse_line_error(line)
        if stream_error is not None:
            observation.stream_error = stream_error
            return
        if _sse_line_has_content(line):
            if observation.first_token_milliseconds is None:
                observation.first_token_milliseconds = now_milliseconds
            observation.last_token_milliseconds = now_milliseconds
        usage = _sse_line_usage(line)
        if usage is not None:
            observation.usage = usage

    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        now_milliseconds = _elapsed_ms_with_clock(started, clock)
        if observation.ttfb_milliseconds is None:
            observation.ttfb_milliseconds = now_milliseconds
        tail += chunk.decode("utf-8", "ignore")
        lines = tail.split("\n")
        tail = lines.pop()
        for line in lines:
            observe_line(line, now_milliseconds)
            if observation.stream_error is not None:
                break
        if observation.stream_error is not None:
            break
    if tail and observation.stream_error is None:
        observe_line(tail, _elapsed_ms_with_clock(started, clock))
    observation.elapsed_milliseconds = _elapsed_ms_with_clock(started, clock)
    return observation


# Source: src/trusted_router/synthetic/probes.py
def _response_error(response: httpx.Response) -> tuple[str, int | None, str | None]:
    try:
        payload = response.json()
    except ValueError:
        return f"http_{response.status_code}", response.status_code, None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) and isinstance(payload, dict):
        detail = payload.get("detail")
        error = detail.get("error") if isinstance(detail, dict) else None
    if not isinstance(error, dict):
        return f"http_{response.status_code}", response.status_code, None
    error_type = str(error.get("type") or f"http_{response.status_code}")
    message = str(error.get("message") or "") or None
    source = str(error.get("source") or "") or None
    status_raw = error.get("status") or error.get("code") or error.get("status_code")
    try:
        status = int(status_raw) if status_raw is not None else response.status_code
    except (TypeError, ValueError):
        status = response.status_code
    return _rotation_error_type(error_type, status, message, source=source), status, message


# Source: src/trusted_router/synthetic/probes.py
_UNSUPPORTED_ROUTE_ERROR_TYPES = frozenset(
    {
        "model_not_found",
        "model_not_available",
        "not_found",
        "not_supported",
        "unsupported",
        "unsupported_model",
        "unsupported_provider",
        "unsupported_route",
    }
)

# Source: src/trusted_router/synthetic/probes.py
_PROBE_CONFIG_ERROR_TYPES = frozenset(
    {
        "bad_request",
        "invalid_request",
        "invalid_request_error",
        "invalid_request_error_type",
    }
)

# Source: src/trusted_router/synthetic/probes.py
_UNSUPPORTED_ROUTE_MESSAGE_MARKERS = (
    "model not found",
    "model_not_found",
    "unknown model",
    "invalid model",
    "no such model",
    "model does not exist",
    "does not exist",
    "not available",
    "unavailable",
    "not enabled",
    "not authorized",
    "not permitted",
    "does not support",
    "not supported",
    "unsupported",
    "no endpoint",
    "no route",
)

# Source: src/trusted_router/synthetic/probes.py
_PROBE_CONFIG_MESSAGE_MARKERS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
)


# Source: src/trusted_router/synthetic/probes.py
def _rotation_error_type(
    error_type: str,
    status: int | None,
    message: str | None,
    *,
    source: str | None = None,
) -> str:
    raw_type = error_type.casefold()
    raw_message = (message or "").casefold()
    raw_source = (source or "").casefold()
    if "workspace billing is paused" in raw_message:
        return "monitor_workspace_paused"
    if "database contention" in raw_message or "deadlock" in raw_message:
        return "router_database_contention"
    if "read-only mode" in raw_message or "planned maintenance" in raw_message:
        return "router_maintenance"
    if raw_source == "router" and any(
        marker in raw_message
        for marker in (
            "insufficient credits",
            "api key is disabled",
            "api key expired",
            "invalid api key",
            "api key not found",
        )
    ):
        return "monitor_account_unavailable"
    if raw_type in _UNSUPPORTED_ROUTE_ERROR_TYPES or any(
        marker in raw_message for marker in _UNSUPPORTED_ROUTE_MESSAGE_MARKERS
    ):
        return "unsupported_route"
    if raw_type in _PROBE_CONFIG_ERROR_TYPES or (
        status in {400, 422}
        and any(marker in raw_message for marker in _PROBE_CONFIG_MESSAGE_MARKERS)
    ):
        return "probe_config_error"
    if raw_source == "router":
        return "router_error"
    if status in {401, 403}:
        return "provider_auth_config"
    return error_type


# Source: src/trusted_router/synthetic/probes.py
_THROUGHPUT_PROMPT = (
    "Continue writing the lowercase word benchmark separated by single spaces "
    "until the response token limit stops you. Do not count, explain, use "
    "punctuation, or stop early."
)


# Source: src/trusted_router/synthetic/probes.py
def _rotation_max_tokens(provider: str, model: str) -> int:
    provider_l = provider.lower()
    model_l = model.lower()
    if provider_l == "openai" and (
        "/o1" in model_l or "/o3" in model_l or "/o4" in model_l or "/gpt-5" in model_l
    ):
        return 512
    if "gemini-2.5" in model_l or "gemini-3" in model_l:
        # Gemini thinks before visible content; hidden thinking consumes the
        # budget but is absent from usage, so 16 yields empty_stream. Live
        # verification on 2026-07-19 showed 2048 works; keep generous headroom.
        return 2048
    if (
        "gpt-oss" in model_l
        or "glm-4.6" in model_l
        or "glm-4.7" in model_l
        or "glm-5" in model_l
        or "nemotron" in model_l
        or "claude-fable-5" in model_l
        or "claude-sonnet-5" in model_l
        or "reasoning" in model_l
        or "thinking" in model_l
    ):
        # Some models reason before emitting visible content: at 16 tokens they
        # finish=length with zero streamed content and register as
        # probe_config_error. Models that omit those reasoning deltas need the
        # larger budget to emit a visible token.
        return 512
    if "kimi-k2" in model_l or "grok" in model_l or "claude-opus" in model_l:
        return 128
    return 16


# Source: src/trusted_router/synthetic/probes.py
def _rotation_omits_temperature(provider: str, model: str) -> bool:
    provider_l = provider.lower()
    model_l = model.lower()
    return (
        (provider_l == "kimi" and "kimi-k2." in model_l)
        or (
            provider_l == "openai"
            and ("/o1" in model_l or "/o3" in model_l or "/o4" in model_l or "/gpt-5" in model_l)
        )
        or (
            provider_l == "anthropic"
            and ("claude-opus-4.7" in model_l or "claude-opus-4.8" in model_l)
        )
    )


# Source: src/trusted_router/synthetic/probes.py
def _rotation_error_excluded_from_uptime(error_type: str | None) -> bool:
    return is_router_origin_error(error_type) or error_type in {
        "unsupported_route",
        "probe_config_error",
        "provider_auth_config",
        "insufficient_throughput_sample",
    }


# Source: src/trusted_router/synthetic/probes.py
def _pong_matches(text: str) -> bool:
    """Accept any output that contains the literal word PONG (case
    insensitive). LLMs reliably emit the word but sometimes wrap it in
    quotes, append punctuation, or prefix a token of whitespace. We only
    want to flag a hard miss (model returned something unrelated, empty
    body, or wrong language)."""
    return "pong" in text.casefold()


# Source: src/trusted_router/synthetic/probes.py
def _chat_text(response: httpx.Response) -> str:
    """Extract assistant-visible text from a /chat/completions reply.

    Handles three shapes the catalog actually returns:
      * Plain string content (OpenAI canonical)
      * List-of-parts content (Anthropic, multimodal adapters):
        [{"type":"text", "text":"…"}, …]
      * Reasoning-content split (kimi-k2.6, glm-4.6, deepseek-v4):
        message.content is empty while message.reasoning_content (or
        message.reasoning) carries the actual answer.

    Concatenates anything we find so the pong matcher sees the full
    answer regardless of which path the upstream took. Before this
    was reasoning-aware, the probe flagged `pong_mismatch` on every
    reasoning model whose visible content arrived empty.
    """
    if response.status_code != 200:
        return ""
    try:
        choices = response.json().get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        parts: list[str] = []
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
        # Reasoning shapes: some providers expose the thinking trace,
        # some emit the answer only inside it when max_tokens caps
        # the visible content. Treat both as fair game for the
        # output_match check.
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        text = item.get("text") or ""
                        if isinstance(text, str):
                            parts.append(text)
        return " ".join(p for p in parts if p)
    except (ValueError, AttributeError):
        return ""


# Source: src/trusted_router/synthetic/probes.py
def _responses_text(response: httpx.Response) -> str:
    """Extract text from a /responses reply, walking the full output[].

    OpenAI's Responses API emits an ordered output[] array; for
    reasoning models the first item is a `reasoning` block and the
    visible answer is further down in a `message`-type item. The
    previous extractor read output[0].content[0].text exclusively,
    so reasoning models showed up as empty → pong_mismatch.
    """
    if response.status_code != 200:
        return ""
    try:
        output = response.json().get("output") or []
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for piece in content:
                    if isinstance(piece, dict):
                        text = piece.get("text") or ""
                        if isinstance(text, str):
                            parts.append(text)
            elif isinstance(content, str):
                parts.append(content)
            # Reasoning summary blocks
            summary = item.get("summary")
            if isinstance(summary, list):
                for piece in summary:
                    if isinstance(piece, dict):
                        text = piece.get("text") or ""
                        if isinstance(text, str):
                            parts.append(text)
        return " ".join(p for p in parts if p)
    except (ValueError, AttributeError):
        return ""


# Source: src/trusted_router/synthetic/probes.py
def _elapsed_ms_with_clock(
    started: float,
    clock: Callable[[], float],
) -> int:
    return max(1, int(round((clock() - started) * 1000)))


# Source: src/trusted_router/synthetic/leaderboard.py
def _percentile(values: Sequence[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # Nearest-rank: smallest value at or above the percentile position.
    rank = max(1, -(-percentile * len(ordered) // 100))  # ceil(p*n/100)
    return ordered[min(rank, len(ordered)) - 1]


# Source: src/trusted_router/synthetic/leaderboard.py
def _median_float(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# Source: src/trusted_router/synthetic/leaderboard.py
def _effective_throughput(sample: ProviderBenchmarkSample) -> float | None:
    """Return a buffering-safe output rate for a long synthetic probe.

    Rows written before the metric correction stored post-first-chunk delivery
    speed, which can be wildly inflated when an upstream buffers SSE events.
    All long-probe rows already carry provider-reported output tokens and total
    elapsed time, so derive the honest end-to-end rate at read time. The stored
    speed remains a compatibility fallback for older tests or partial rows.
    """
    if (
        sample.output_tokens > 0
        and sample.elapsed_milliseconds is not None
        and sample.elapsed_milliseconds > 0
    ):
        return sample.output_tokens * 1000 / sample.elapsed_milliseconds
    if sample.speed_tokens_per_second is not None and sample.speed_tokens_per_second > 0:
        return sample.speed_tokens_per_second
    return None


# Source: src/trusted_router/synthetic/leaderboard.py
@dataclass
class ProviderModelStats:
    provider: str
    model: str
    sample_count: int = 0
    success_count: int = 0
    error_count: int = 0
    effective_attempt_count: int = 0
    effective_success_count: int = 0
    excluded_count: int = 0
    provider_sample_count: int = 0
    provider_success_count: int = 0
    deadline_sample_count: int = 0
    within_deadline_count: int = 0
    capacity_attempt_count: int = 0
    capacity_rejection_count: int = 0
    ttft_sample_count: int = 0
    throughput_sample_count: int = 0
    rank: int | None = None
    rank_eligible: bool = False
    p50_ttft_ms: int | None = None
    p95_ttft_ms: int | None = None
    p50_ttfb_ms: int | None = None
    p95_ttfb_ms: int | None = None
    p50_tokens_per_second: float | None = None
    last_seen: str | None = None
    errors: Counter[str] = field(default_factory=Counter)
    excluded_reasons: Counter[str] = field(default_factory=Counter)
    failure_owners: Counter[str] = field(default_factory=Counter)
    failure_classes: Counter[str] = field(default_factory=Counter)

    @property
    def uptime(self) -> float:
        return self.success_count / self.sample_count if self.sample_count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.sample_count if self.sample_count else 0.0

    @property
    def provider_availability(self) -> float:
        return (
            self.provider_success_count / self.provider_sample_count
            if self.provider_sample_count
            else 0.0
        )

    @property
    def completion_rate(self) -> float:
        return (
            self.effective_success_count / self.effective_attempt_count
            if self.effective_attempt_count
            else 0.0
        )

    @property
    def availability_within_deadline(self) -> float:
        return (
            self.within_deadline_count / self.deadline_sample_count
            if self.deadline_sample_count
            else 0.0
        )

    @property
    def capacity_acceptance_rate(self) -> float:
        return (
            1 - self.capacity_rejection_count / self.capacity_attempt_count
            if self.capacity_attempt_count
            else 0.0
        )

    @property
    def top_error(self) -> str | None:
        common = self.errors.most_common(1)
        return common[0][0] if common else None

    @property
    def top_excluded(self) -> str | None:
        common = self.excluded_reasons.most_common(1)
        return common[0][0] if common else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "sample_count": self.sample_count,
            "uptime": round(self.uptime, 4) if self.sample_count else None,
            "error_rate": round(self.error_rate, 4),
            "effective_attempt_count": self.effective_attempt_count,
            "completion_rate": (
                round(self.completion_rate, 4) if self.effective_attempt_count else None
            ),
            "provider_sample_count": self.provider_sample_count,
            "provider_availability": (
                round(self.provider_availability, 4)
                if self.provider_sample_count
                else None
            ),
            "deadline_sample_count": self.deadline_sample_count,
            "availability_within_deadline": (
                round(self.availability_within_deadline, 4)
                if self.deadline_sample_count
                else None
            ),
            "first_token_deadline_ms": int(
                model_deadlines(self.model).first_token_seconds * 1000
            ),
            "capacity_attempt_count": self.capacity_attempt_count,
            "capacity_acceptance_rate": (
                round(self.capacity_acceptance_rate, 4)
                if self.capacity_attempt_count
                else None
            ),
            "excluded_count": self.excluded_count,
            "ttft_sample_count": self.ttft_sample_count,
            "throughput_sample_count": self.throughput_sample_count,
            "rank": self.rank,
            "rank_eligible": self.rank_eligible,
            "top_error": self.top_error,
            "top_excluded": self.top_excluded,
            "errors": dict(self.errors),
            "excluded_reasons": dict(self.excluded_reasons),
            "failure_owners": dict(self.failure_owners),
            "failure_classes": dict(self.failure_classes),
            "p50_ttft_ms": self.p50_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "p50_ttfb_ms": self.p50_ttfb_ms,
            "p95_ttfb_ms": self.p95_ttfb_ms,
            "p50_tokens_per_second": (
                round(self.p50_tokens_per_second, 2)
                if self.p50_tokens_per_second is not None
                else None
            ),
            "last_seen": self.last_seen,
        }


# Source: src/trusted_router/synthetic/leaderboard.py
@dataclass
class ProviderStats:
    provider: str
    model_count: int = 0
    sample_count: int = 0
    success_count: int = 0
    error_count: int = 0
    effective_attempt_count: int = 0
    effective_success_count: int = 0
    excluded_count: int = 0
    provider_sample_count: int = 0
    provider_success_count: int = 0
    deadline_sample_count: int = 0
    within_deadline_count: int = 0
    capacity_attempt_count: int = 0
    capacity_rejection_count: int = 0
    ttft_sample_count: int = 0
    throughput_sample_count: int = 0
    rank: int | None = None
    rank_eligible: bool = False
    p50_ttft_ms: int | None = None
    p95_ttft_ms: int | None = None
    p50_ttfb_ms: int | None = None
    p95_ttfb_ms: int | None = None
    p50_tokens_per_second: float | None = None
    errors: Counter[str] = field(default_factory=Counter)
    excluded_reasons: Counter[str] = field(default_factory=Counter)
    failure_owners: Counter[str] = field(default_factory=Counter)
    failure_classes: Counter[str] = field(default_factory=Counter)

    @property
    def uptime(self) -> float:
        return self.success_count / self.sample_count if self.sample_count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.sample_count if self.sample_count else 0.0

    @property
    def provider_availability(self) -> float:
        return (
            self.provider_success_count / self.provider_sample_count
            if self.provider_sample_count
            else 0.0
        )

    @property
    def completion_rate(self) -> float:
        return (
            self.effective_success_count / self.effective_attempt_count
            if self.effective_attempt_count
            else 0.0
        )

    @property
    def availability_within_deadline(self) -> float:
        return (
            self.within_deadline_count / self.deadline_sample_count
            if self.deadline_sample_count
            else 0.0
        )

    @property
    def capacity_acceptance_rate(self) -> float:
        return (
            1 - self.capacity_rejection_count / self.capacity_attempt_count
            if self.capacity_attempt_count
            else 0.0
        )

    @property
    def top_error(self) -> str | None:
        common = self.errors.most_common(1)
        return common[0][0] if common else None

    @property
    def top_excluded(self) -> str | None:
        common = self.excluded_reasons.most_common(1)
        return common[0][0] if common else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_count": self.model_count,
            "sample_count": self.sample_count,
            "uptime": round(self.uptime, 4) if self.sample_count else None,
            "error_rate": round(self.error_rate, 4),
            "effective_attempt_count": self.effective_attempt_count,
            "completion_rate": (
                round(self.completion_rate, 4) if self.effective_attempt_count else None
            ),
            "provider_sample_count": self.provider_sample_count,
            "provider_availability": (
                round(self.provider_availability, 4)
                if self.provider_sample_count
                else None
            ),
            "deadline_sample_count": self.deadline_sample_count,
            "availability_within_deadline": (
                round(self.availability_within_deadline, 4)
                if self.deadline_sample_count
                else None
            ),
            "capacity_attempt_count": self.capacity_attempt_count,
            "capacity_acceptance_rate": (
                round(self.capacity_acceptance_rate, 4)
                if self.capacity_attempt_count
                else None
            ),
            "excluded_count": self.excluded_count,
            "ttft_sample_count": self.ttft_sample_count,
            "throughput_sample_count": self.throughput_sample_count,
            "rank": self.rank,
            "rank_eligible": self.rank_eligible,
            "top_error": self.top_error,
            "top_excluded": self.top_excluded,
            "errors": dict(self.errors),
            "excluded_reasons": dict(self.excluded_reasons),
            "failure_owners": dict(self.failure_owners),
            "failure_classes": dict(self.failure_classes),
            "p50_ttft_ms": self.p50_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "p50_ttfb_ms": self.p50_ttfb_ms,
            "p95_ttfb_ms": self.p95_ttfb_ms,
            "p50_tokens_per_second": (
                round(self.p50_tokens_per_second, 2)
                if self.p50_tokens_per_second is not None
                else None
            ),
        }


# Source: src/trusted_router/synthetic/leaderboard.py
def _sort_key(p50_ttft_ms: int | None) -> tuple[int, int]:
    # Latency breaks reliability ties; un-measured (None) sinks to the bottom.
    return (0 if p50_ttft_ms is not None else 1, p50_ttft_ms or 0)


# Source: src/trusted_router/synthetic/leaderboard.py
def aggregate_leaderboard(
    samples: Iterable[ProviderBenchmarkSample],
    *,
    min_samples: int = 1,
    model_rank_min_samples: int = 1,
    provider_rank_min_samples: int = 1,
    rank_min_ttft_samples: int = 1,
) -> dict[str, Any]:
    """Aggregate samples into ranked per-model and per-provider stats.

    Models/providers with fewer than ``min_samples`` are excluded from the
    ranked lists (callers surface a "limited data" note for thin coverage).
    """
    by_model: dict[tuple[str, str], ProviderModelStats] = {}
    ttft: dict[tuple[str, str], list[int]] = {}
    ttfb: dict[tuple[str, str], list[int]] = {}
    sustained_tps: dict[tuple[str, str], list[float]] = {}
    provider_ttft: dict[str, list[int]] = {}
    provider_ttfb: dict[str, list[int]] = {}

    for sample in samples:
        key = (sample.provider, sample.model)
        stats = by_model.get(key)
        if stats is None:
            stats = ProviderModelStats(provider=sample.provider, model=sample.model)
            by_model[key] = stats
            ttft[key] = []
            ttfb[key] = []
            sustained_tps[key] = []
            provider_ttft.setdefault(sample.provider, [])
            provider_ttfb.setdefault(sample.provider, [])
        if sample.source == "synthetic_throughput":
            effective_tps = _effective_throughput(sample)
            if sample.status == "success" and effective_tps is not None:
                sustained_tps[key].append(effective_tps)
                stats.throughput_sample_count += 1
            if stats.last_seen is None or sample.created_at > stats.last_seen:
                stats.last_seen = sample.created_at
            # Long probes are intentionally excluded from availability and
            # TTFT. The short PONG probe already measures both without making a
            # slow 512-token completion look like provider downtime.
            continue
        label = sample.error_type or (
            f"http_{sample.error_status}" if sample.error_status else "error"
        )
        attribution = classify_provider_failure(
            status=sample.status,
            error_type=sample.error_type,
            error_status=sample.error_status,
            error_message=sample.error_message,
        )
        if attribution.owner not in {FailureOwner.NONE, FailureOwner.CONFIGURATION}:
            stats.failure_owners[attribution.owner.value] += 1
            stats.failure_classes[attribution.failure_class.value] += 1
        if attribution.owner not in {FailureOwner.CONFIGURATION, FailureOwner.CUSTOMER}:
            stats.effective_attempt_count += 1
            if sample.status == "success":
                stats.effective_success_count += 1
        if _excluded_from_uptime(sample):
            stats.excluded_count += 1
            stats.excluded_reasons[label] += 1
            continue
        stats.sample_count += 1
        if sample.status == "success":
            stats.success_count += 1
        else:
            stats.error_count += 1
            stats.errors[label] += 1
        if attribution.counts_toward_provider_availability:
            stats.provider_sample_count += 1
            stats.capacity_attempt_count += 1
            if sample.status == "success":
                stats.provider_success_count += 1
            if attribution.capacity_rejected:
                stats.capacity_rejection_count += 1
            # Failed provider attempts missed the first-token deadline even
            # when no byte arrived. Successful non-streaming observations may
            # not carry TTFT at all; excluding those unmeasured successes keeps
            # the denominator honest instead of reporting them as late.
            if sample.status != "success" or sample.first_token_milliseconds is not None:
                stats.deadline_sample_count += 1
        if sample.first_token_milliseconds is not None:
            ttft[key].append(sample.first_token_milliseconds)
            provider_ttft[sample.provider].append(sample.first_token_milliseconds)
            if attribution.counts_toward_provider_availability:
                deadline_ms = (
                    model_deadlines(sample.model).first_token_seconds * 1000
                )
                if sample.status == "success" and sample.first_token_milliseconds <= deadline_ms:
                    stats.within_deadline_count += 1
        if sample.ttfb_milliseconds is not None:
            ttfb[key].append(sample.ttfb_milliseconds)
            provider_ttfb[sample.provider].append(sample.ttfb_milliseconds)
        if stats.last_seen is None or sample.created_at > stats.last_seen:
            stats.last_seen = sample.created_at

    for key, stats in by_model.items():
        stats.ttft_sample_count = len(ttft[key])
        stats.p50_ttft_ms = _percentile(ttft[key], 50)
        stats.p95_ttft_ms = _percentile(ttft[key], 95)
        stats.p50_ttfb_ms = _percentile(ttfb[key], 50)
        stats.p95_ttfb_ms = _percentile(ttfb[key], 95)
        stats.p50_tokens_per_second = _median_float(sustained_tps[key])
        stats.rank_eligible = (
            stats.sample_count >= model_rank_min_samples
            and stats.ttft_sample_count >= rank_min_ttft_samples
        )

    models = [
        stats
        for stats in by_model.values()
        if stats.sample_count >= min_samples or stats.throughput_sample_count >= min_samples
    ]
    models.sort(
        key=lambda stats: (
            0 if stats.rank_eligible else 1,
            -stats.provider_availability,
            *_sort_key(stats.p50_ttft_ms),
            stats.model,
            stats.provider,
        )
    )
    next_rank = 1
    for stats in models:
        if stats.rank_eligible:
            stats.rank = next_rank
            next_rank += 1

    providers = _aggregate_providers(
        models,
        provider_rank_min_samples=provider_rank_min_samples,
        rank_min_ttft_samples=rank_min_ttft_samples,
        provider_ttft=provider_ttft,
        provider_ttfb=provider_ttfb,
    )
    return {
        "models": [s.as_dict() for s in models],
        "providers": [s.as_dict() for s in providers],
        "model_count": len(models),
        "provider_count": len(providers),
        "total_samples": sum(s.sample_count for s in models),
        "total_throughput_samples": sum(s.throughput_sample_count for s in models),
        "excluded_samples": sum(s.excluded_count for s in by_model.values()),
    }


# Source: src/trusted_router/synthetic/leaderboard.py
def _aggregate_providers(
    model_stats: list[ProviderModelStats],
    *,
    provider_rank_min_samples: int,
    rank_min_ttft_samples: int,
    provider_ttft: dict[str, list[int]],
    provider_ttfb: dict[str, list[int]],
) -> list[ProviderStats]:
    by_provider: dict[str, ProviderStats] = {}
    ttft: dict[str, list[int]] = {}
    tps: dict[str, list[float]] = {}
    for model_stat in model_stats:
        agg = by_provider.get(model_stat.provider)
        if agg is None:
            agg = ProviderStats(provider=model_stat.provider)
            by_provider[model_stat.provider] = agg
            ttft[model_stat.provider] = []
            tps[model_stat.provider] = []
        agg.model_count += 1
        agg.sample_count += model_stat.sample_count
        agg.success_count += model_stat.success_count
        agg.error_count += model_stat.error_count
        agg.effective_attempt_count += model_stat.effective_attempt_count
        agg.effective_success_count += model_stat.effective_success_count
        agg.excluded_count += model_stat.excluded_count
        agg.provider_sample_count += model_stat.provider_sample_count
        agg.provider_success_count += model_stat.provider_success_count
        agg.deadline_sample_count += model_stat.deadline_sample_count
        agg.within_deadline_count += model_stat.within_deadline_count
        agg.capacity_attempt_count += model_stat.capacity_attempt_count
        agg.capacity_rejection_count += model_stat.capacity_rejection_count
        agg.ttft_sample_count += model_stat.ttft_sample_count
        agg.throughput_sample_count += model_stat.throughput_sample_count
        agg.errors.update(model_stat.errors)
        agg.excluded_reasons.update(model_stat.excluded_reasons)
        agg.failure_owners.update(model_stat.failure_owners)
        agg.failure_classes.update(model_stat.failure_classes)
        # Weight each model's p50 by its sample count for the provider median.
        if model_stat.p50_ttft_ms is not None:
            ttft[model_stat.provider].extend([model_stat.p50_ttft_ms] * model_stat.sample_count)
        if model_stat.p50_tokens_per_second is not None:
            weight = model_stat.throughput_sample_count
            tps[model_stat.provider].extend([model_stat.p50_tokens_per_second] * weight)
    providers = list(by_provider.values())
    for agg in providers:
        agg.p50_ttft_ms = _percentile(ttft[agg.provider], 50)
        agg.p95_ttft_ms = _percentile(provider_ttft.get(agg.provider, []), 95)
        agg.p50_ttfb_ms = _percentile(provider_ttfb.get(agg.provider, []), 50)
        agg.p95_ttfb_ms = _percentile(provider_ttfb.get(agg.provider, []), 95)
        agg.p50_tokens_per_second = _median_float(tps[agg.provider])
        agg.rank_eligible = (
            agg.sample_count >= provider_rank_min_samples
            and agg.ttft_sample_count >= rank_min_ttft_samples
        )
    providers.sort(
        key=lambda stats: (
            0 if stats.rank_eligible else 1,
            -stats.provider_availability,
            *_sort_key(stats.p50_ttft_ms),
            stats.provider,
        )
    )
    next_rank = 1
    for provider_stat in providers:
        if provider_stat.rank_eligible:
            provider_stat.rank = next_rank
            next_rank += 1
    return providers


# Source: src/trusted_router/synthetic/leaderboard.py
def _excluded_from_uptime(sample: ProviderBenchmarkSample) -> bool:
    attribution = classify_provider_failure(
        status=sample.status,
        error_type=sample.error_type,
        error_status=sample.error_status,
        error_message=sample.error_message,
    )
    return attribution.owner in {
        FailureOwner.CONFIGURATION,
        FailureOwner.CUSTOMER,
        FailureOwner.TRUSTEDROUTER,
    }


# Source: scripts/classify_provider_routes.py
_DEAD_STATUSES = frozenset({400, 401, 403, 404, 422})

# Source: scripts/classify_provider_routes.py
_DEAD_MARKERS = (
    "not found",
    "does not exist",
    "deployment",
    "not available",
    "unavailable",
    "not supported",
    "no endpoint",
    "unknown model",
)


# Source: scripts/classify_provider_routes.py
def _classify(status: int | None, body: str) -> str:
    low = body.casefold()
    if status == 200:
        return "ok"
    if status in _DEAD_STATUSES or any(m in low for m in _DEAD_MARKERS):
        return "dead"
    return "flaky"  # 429 / 5xx / timeout / network — real provider health


# Source: scripts/pricing/provider_contract_catalog.py
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$")

# Source: scripts/pricing/provider_contract_catalog.py
_OWNER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Source: scripts/pricing/provider_contract_catalog.py
_TOP_LEVEL_FIELDS = frozenset({"object", "data"})

# Source: scripts/pricing/provider_contract_catalog.py
_TOP_LEVEL_V2_FIELDS = frozenset({"object", "contract_version", "provider", "data"})

# Source: scripts/pricing/provider_contract_catalog.py
_MODEL_FIELDS = frozenset(
    {
        "id",
        "object",
        "owned_by",
        "name",
        "type",
        "context_length",
        "max_output_tokens",
        "endpoints",
        "input_modalities",
        "output_modalities",
        "capabilities",
        "pricing",
        "lifecycle",
    }
)

# Source: scripts/pricing/provider_contract_catalog.py
_MODEL_V2_FIELDS = _MODEL_FIELDS | {"reliability"}

# Source: scripts/pricing/provider_contract_catalog.py
_CAPABILITY_FIELDS = frozenset(
    {"streaming", "tools", "structured_output", "reasoning", "prompt_caching"}
)

# Source: scripts/pricing/provider_contract_catalog.py
_PRICING_FIELDS = frozenset(
    {
        "currency",
        "unit",
        "input",
        "output",
        "cached_input",
        "cache_write",
        "minimum_request",
    }
)

# Source: scripts/pricing/provider_contract_catalog.py
_LIFECYCLE_FIELDS = frozenset(
    {"status", "deprecation_at", "retirement_at", "replacement_model_id"}
)

# Source: scripts/pricing/provider_contract_catalog.py
_ENDPOINTS = frozenset({"chat/completions", "responses"})

# Source: scripts/pricing/provider_contract_catalog.py
_INPUT_MODALITIES = frozenset({"text", "image", "audio", "video", "file"})

# Source: scripts/pricing/provider_contract_catalog.py
_OUTPUT_MODALITIES = frozenset({"text", "image", "audio"})

# Source: scripts/pricing/provider_contract_catalog.py
_LIFECYCLE_STATUSES = frozenset({"active", "deprecated", "retired"})

# Source: scripts/pricing/provider_contract_catalog.py
_PROVIDER_V2_FIELDS = frozenset(
    {
        "id",
        "status_url",
        "support_contact",
        "incident_contact",
        "regions",
        "request_id_header",
        "error_contract",
    }
)

# Source: scripts/pricing/provider_contract_catalog.py
_ERROR_CONTRACT_FIELDS = frozenset(
    {
        "rate_limit_status",
        "overload_status",
        "retry_after_header",
        "account_quota_error_codes",
    }
)

# Source: scripts/pricing/provider_contract_catalog.py
_RELIABILITY_FIELDS = frozenset(
    {
        "first_token_timeout_seconds",
        "completion_timeout_seconds",
        "stream_idle_timeout_seconds",
        "capacity_scope",
    }
)

# Source: scripts/pricing/provider_contract_catalog.py
_CAPACITY_SCOPES = frozenset({"global", "region", "model", "model_region"})


# Source: scripts/pricing/provider_contract_catalog.py
def _decimal(value: object, *, label: str, nullable: bool = False) -> Decimal | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"(0|[1-9][0-9]*)(\.[0-9]+)?", value):
        raise RuntimeError(f"{label} must be a non-negative decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError(f"{label} is not a valid decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RuntimeError(f"{label} must be finite and non-negative")
    return parsed
