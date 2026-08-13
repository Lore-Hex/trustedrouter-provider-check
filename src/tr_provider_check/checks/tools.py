"""Tier 5: streamed parallel tools and the enclave's tool-result replay shape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from tr_provider_check.contract import _chat_text, _response_error
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult, CheckStatus, check_result

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Return the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_time",
            "description": "Return the local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class _ToolTurn:
    calls: list[dict[str, Any]]


def _sse_payloads(body: bytes) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in body.splitlines():
        if not line.startswith(b"data: "):
            continue
        raw = line[len(b"data: ") :]
        if raw == b"[DONE]":
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _error_result(
    *,
    check_id: str,
    assertion: str,
    declared: bool | None,
    status_code: int | None,
    error_type: str | None,
    error_message: str | None,
) -> CheckResult:
    status: CheckStatus = "fail" if declared is True else "skip"
    reason = (
        "the declared tools capability rejected the enclave request"
        if declared is True
        else "tools were neither declared nor discoverable: the probe was rejected"
    )
    return check_result(
        id=check_id,
        tier=5,
        status=status,
        assertion=assertion,
        measured={"reason": reason, "http_status": status_code},
        contract_ref="enclave-go/internal/llm/byok.go",
        marketplace_bullet="Tool requests use the OpenAI-compatible enclave wire shape.",
        remediation="Declare tools only for models that accept tools, tool_choice, and parallel_tool_calls on /chat/completions.",
        error_type=error_type,
        error_status=status_code,
        error_message=error_message,
    )


async def run_parallel_tool_delta_check(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> tuple[CheckResult, _ToolTurn | None]:
    """Check the delta invariants consumed by ``llm/stream_translate.go``."""

    assertion = (
        "a forced parallel tool stream emits at least two calls; every tool delta "
        "has index, each index's first delta has function.name, and concatenated "
        "arguments are valid JSON"
    )
    if declared is False:
        return (
            check_result(
                id="tools.parallel-deltas",
                tier=5,
                status="skip",
                assertion=assertion,
                measured={"reason": "the selected model declares tools=false"},
                contract_ref="enclave-go/internal/llm/stream_translate.go",
                marketplace_bullet="Parallel tool-call deltas can be translated without data loss.",
                remediation="Declare tools=true only after the model supports streamed parallel calls.",
            ),
            None,
        )

    extra = {
        "tools": _TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": True,
    }
    try:
        async with client.stream_chat(
            model=model,
            prompt=(
                "Call lookup_weather once and lookup_time once for Athens. "
                "Make both calls in parallel and do not answer with prose."
            ),
            temperature=0,
            extra=extra,
        ) as response:
            body = await response.aread()
            if response.status_code != 200:
                error_type, error_status, error_message = _response_error(response)
                return (
                    _error_result(
                        check_id="tools.parallel-deltas",
                        assertion=assertion,
                        declared=declared,
                        status_code=error_status,
                        error_type=error_type,
                        error_message=error_message,
                    ),
                    None,
                )
    except httpx.HTTPError as error:
        return (
            _error_result(
                check_id="tools.parallel-deltas",
                assertion=assertion,
                declared=declared,
                status_code=None,
                error_type=error.__class__.__name__,
                error_message=str(error),
            ),
            None,
        )

    delta_count = 0
    missing_index_count = 0
    late_name_indices: list[int] = []
    arguments_by_index: dict[int, list[str]] = {}
    calls_by_index: dict[int, dict[str, Any]] = {}
    invalid_delta_count = 0
    for payload in _sse_payloads(body):
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice, dict) else None
        tool_deltas = delta.get("tool_calls") if isinstance(delta, dict) else None
        if not isinstance(tool_deltas, list):
            continue
        for tool_delta in tool_deltas:
            delta_count += 1
            if not isinstance(tool_delta, dict):
                invalid_delta_count += 1
                continue
            index = tool_delta.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                missing_index_count += 1
                continue
            function = tool_delta.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name")
            arguments = function.get("arguments")
            if index not in calls_by_index:
                if not isinstance(name, str) or not name:
                    late_name_indices.append(index)
                calls_by_index[index] = {
                    "id": tool_delta.get("id"),
                    "type": tool_delta.get("type") or "function",
                    "function": {"name": name if isinstance(name, str) else ""},
                }
                arguments_by_index[index] = []
            else:
                call_function = calls_by_index[index]["function"]
                if not call_function["name"] and isinstance(name, str) and name:
                    call_function["name"] = name
                if not calls_by_index[index]["id"] and tool_delta.get("id"):
                    calls_by_index[index]["id"] = tool_delta["id"]
            if isinstance(arguments, str):
                arguments_by_index[index].append(arguments)
            elif arguments is not None:
                invalid_delta_count += 1

    argument_errors: dict[str, str] = {}
    replay_calls: list[dict[str, Any]] = []
    for index in sorted(calls_by_index):
        arguments = "".join(arguments_by_index[index])
        try:
            parsed_arguments = json.loads(arguments)
            if not isinstance(parsed_arguments, dict):
                raise ValueError("arguments JSON must be an object")
        except ValueError as error:
            argument_errors[str(index)] = str(error)
        call = calls_by_index[index]
        replay_calls.append(
            {
                "id": call["id"] or f"call_{index}",
                "type": call["type"],
                "function": {
                    "name": call["function"]["name"],
                    "arguments": arguments,
                },
            }
        )

    call_count = len(calls_by_index)
    discovered = delta_count > 0
    if not discovered and declared is not True:
        status: CheckStatus = "skip"
        reason = "the model did not declare tools and emitted no tool-call deltas"
    else:
        # The enclave breaks on a delta missing index, or on function.name
        # arriving after the first delta for an index -- not on how MANY calls
        # a model chose to make. Cerebras/gpt-oss-120b answered the forced
        # prompt with one perfectly-formed call; failing that would fail a
        # provider for a model's decision, which is not a contract defect.
        shape_ok = (
            missing_index_count == 0
            and invalid_delta_count == 0
            and not late_name_indices
            and not argument_errors
        )
        if not shape_ok:
            status = "fail"
            reason = None
        elif call_count >= 2:
            status = "pass"
            reason = None
        else:
            status = "warn"
            reason = (
                "every delta was well formed, but the model emitted "
                f"{call_count} tool call(s), so parallel-call correlation was "
                "not exercised"
            )

    result = check_result(
        id="tools.parallel-deltas",
        tier=5,
        status=status,
        assertion=assertion,
        measured={
            "capability_declared": declared,
            "capability_discovered": discovered,
            "tool_delta_count": delta_count,
            "tool_call_count": call_count,
            "indices": sorted(calls_by_index),
            "missing_index_count": missing_index_count,
            "invalid_delta_count": invalid_delta_count,
            "late_name_indices": late_name_indices,
            "argument_errors": argument_errors,
            "reason": reason,
        },
        contract_ref=(
            "enclave-go/internal/llm/stream_translate.go "
            "(toolCalls[delta.Index], name captured when the accumulator starts)"
        ),
        marketplace_bullet="Parallel tool-call deltas can be translated without losing names or arguments.",
        remediation=(
            "Emit index on every tool_calls delta, put function.name in the first "
            "delta for each index, and stream argument fragments that concatenate "
            "to one JSON object. Emit at least two calls when parallel calls are required."
        ),
        error_type="stream_error" if status == "fail" else None,
        error_message="parallel tool-call deltas violate enclave invariants"
        if status == "fail"
        else None,
    )
    replayable = (
        status == "pass"
        and len(replay_calls) >= 2
        and all(call["function"]["name"] for call in replay_calls)
    )
    return result, _ToolTurn(replay_calls) if replayable else None


async def run_tool_round_trip_check(
    client: GatewayClient,
    model: str,
    turn: _ToolTurn | None,
    *,
    unavailable_reason: str,
) -> CheckResult:
    """Replay the exact empty-string tool turn built in ``llm/byok.go``."""

    assertion = (
        'a replayed assistant tool turn uses content:"" plus role:"tool" results '
        "and receives a normal non-empty assistant completion"
    )
    if turn is None:
        return check_result(
            id="tools.round-trip",
            tier=5,
            status="skip",
            assertion=assertion,
            measured={"reason": unavailable_reason},
            contract_ref="enclave-go/internal/llm/byok.go (openAICompatibleToolMessages)",
            marketplace_bullet="Tool results can be replayed through the OpenAI-compatible endpoint.",
            remediation="Fix the parallel tool-call turn first, then rerun the round-trip.",
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Call lookup_weather once and lookup_time once for Athens, then "
                "summarize both results."
            ),
        },
        {"role": "assistant", "content": "", "tool_calls": turn.calls},
    ]
    for call in turn.calls:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": '{"result":"ok"}',
            }
        )

    try:
        response = await client.chat(
            model=model,
            prompt="tool round-trip replay",
            temperature=0,
            extra={"messages": messages},
        )
        response_status: int | None = response.status_code
        text = _chat_text(response) if response.status_code == 200 else ""
        ok = response.status_code == 200 and bool(text.strip())
        if response.status_code != 200:
            error_type, error_status, error_message = _response_error(response)
        elif not ok:
            error_type, error_status, error_message = (
                "empty_stream",
                response.status_code,
                "tool replay returned no normal assistant content",
            )
        else:
            error_type, error_status, error_message = None, None, None
    except httpx.HTTPError as error:
        ok = False
        text = ""
        response_status = None
        error_type, error_status, error_message = (
            error.__class__.__name__,
            None,
            str(error),
        )

    return check_result(
        id="tools.round-trip",
        tier=5,
        status="pass" if ok else "fail",
        assertion=assertion,
        measured={
            "http_status": response_status,
            "tool_call_count": len(turn.calls),
            "assistant_content_type": "string",
            "assistant_content_length": 0,
            "completion_nonempty": bool(text.strip()),
        },
        contract_ref=(
            "enclave-go/internal/llm/byok.go "
            "(openAICompatibleToolMessages emits Content: text.String())"
        ),
        marketplace_bullet="Tool-only assistant history with empty-string content is accepted.",
        remediation=(
            'Accept assistant messages with content:"" and tool_calls, followed by '
            "one role:tool message per tool_call_id, then return a normal completion."
        ),
        error_type=error_type,
        error_status=error_status,
        error_message=error_message,
    )


async def run_tool_checks(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> list[CheckResult]:
    """Run both Tier 5 tool checks against one selected model."""

    parallel, turn = await run_parallel_tool_delta_check(
        client, model, declared=declared
    )
    reason = str(parallel.measured.get("reason") or "parallel tool turn was invalid")
    round_trip = await run_tool_round_trip_check(
        client,
        model,
        turn,
        unavailable_reason=reason,
    )
    return [parallel, round_trip]
