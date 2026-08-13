"""Tier 5: JSON-object and JSON-Schema response-format compatibility."""

from __future__ import annotations

import json
from typing import Any

import httpx
from jsonschema import Draft202012Validator, ValidationError

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.contract import _response_error
from tr_provider_check.http import GatewayClient, probe_verdict
from tr_provider_check.report import CheckResult, CheckStatus, check_result

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["answer", "count"],
    "additionalProperties": False,
}


def _structured_answer_text(response: httpx.Response) -> str:
    """Return only the assistant's content, ignoring any reasoning channel."""

    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Content-parts form: concatenate only the text parts.
    if isinstance(content, list):
        parts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(part for part in parts if part)
    return ""


def _skip(check_id: str, assertion: str) -> CheckResult:
    return check_result(
        id=check_id,
        tier=5,
        status="skip",
        assertion=assertion,
        measured={"reason": "the selected model declares structured_output=false"},
        contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
        marketplace_bullet="Structured response formats are honored when supported.",
        remediation="Declare structured_output=true only after response_format is supported.",
    )


async def _run_response_format(
    client: GatewayClient,
    model: str,
    *,
    check_id: str,
    assertion: str,
    response_format: dict[str, Any],
    declared: bool | None,
    validate_schema: bool,
) -> CheckResult:
    try:
        response = await client.chat(
            model=model,
            prompt=(
                'Return only this JSON object: {"answer":"PONG","count":2}. '
                "Do not use Markdown fences or add properties."
            ),
            temperature=0,
            extra={"response_format": response_format},
        )
    except httpx.HTTPError as error:
        transport_status, reason = probe_verdict(None, declared=declared is True)
        return check_result(
            id=check_id,
            tier=5,
            status=transport_status,
            assertion=assertion,
            measured={
                "capability_declared": declared,
                "error": error.__class__.__name__,
                "inconclusive": True,
                "reason": reason,
            },
            contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
            marketplace_bullet="Structured response formats are accepted by the selected route.",
            remediation="Configure the selected route to accept the forwarded response_format field.",
            error_type=error.__class__.__name__,
            error_message=str(error),
        )

    if response.status_code != 200:
        error_type, error_status, error_message = _response_error(response)
        rejection_status, reason = probe_verdict(
            response.status_code, declared=declared is True
        )
        return check_result(
            id=check_id,
            tier=5,
            status=rejection_status,
            assertion=assertion,
            measured={
                "reason": reason,
                "inconclusive": rejection_status == "warn",
                "capability_declared": declared,
                "http_status": response.status_code,
            },
            contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
            marketplace_bullet="Structured response formats are accepted by the selected route.",
            remediation="Configure the selected route to accept the forwarded response_format field.",
            error_type=error_type,
            error_status=error_status,
            error_message=error_message,
        )

    # NOT _chat_text: that extractor deliberately falls back to
    # reasoning_content so a reasoning model's PONG is still found. A
    # structured answer is the assistant's content, never its chain of
    # thought. Parsing the fallback failed endpoints whose content held exact
    # JSON while reasoning_content held prose; both were
    # reported identically as malformed JSON at the same byte offset.
    text = _structured_answer_text(response)
    parse_error: str | None = None
    validation_error: str | None = None
    parsed: object = None
    try:
        parsed = json.loads(text)
    except ValueError as error:
        parse_error = str(error)
    if parse_error is None and validate_schema:
        try:
            Draft202012Validator(_OUTPUT_SCHEMA).validate(parsed)
        except ValidationError as error:
            validation_error = error.message
    ok = parse_error is None and validation_error is None
    result_status: CheckStatus = (
        "pass" if ok else "fail" if declared is True else "skip"
    )
    return check_result(
        id=check_id,
        tier=5,
        status=result_status,
        assertion=assertion,
        measured={
            "capability_declared": declared,
            "capability_discovered": ok,
            "http_status": response.status_code,
            "body_parsed_as_json": parse_error is None,
            "schema_valid": (
                None
                if not validate_schema or parse_error is not None
                else validation_error is None
            ),
            "parse_error": parse_error,
            "validation_error": validation_error,
        },
        contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
        marketplace_bullet="Accepted structured-output requests return conforming JSON.",
        remediation=(
            "Once response_format is accepted, return a JSON value that conforms "
            "to the requested mode and supplied JSON Schema."
        ),
        error_type="stream_error" if result_status == "fail" else None,
        error_message="accepted response_format returned non-conforming output"
        if result_status == "fail"
        else None,
    )


async def run_json_object_check(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> CheckResult:
    """Check ``response_format`` forwarding from ``llm/byok.go`` as JSON object."""

    assertion = assertion_for("structured.json-object")
    if declared is False:
        return _skip("structured.json-object", assertion)
    return await _run_response_format(
        client,
        model,
        check_id="structured.json-object",
        assertion=assertion,
        response_format={"type": "json_object"},
        declared=declared,
        validate_schema=False,
    )


async def run_json_schema_check(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> CheckResult:
    """Check ``response_format`` forwarding from ``llm/byok.go`` as JSON Schema."""

    assertion = assertion_for("structured.json-schema")
    if declared is False:
        return _skip("structured.json-schema", assertion)
    return await _run_response_format(
        client,
        model,
        check_id="structured.json-schema",
        assertion=assertion,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "provider_check_payload",
                "strict": True,
                "schema": _OUTPUT_SCHEMA,
            },
        },
        declared=declared,
        validate_schema=True,
    )


async def run_structured_checks(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> list[CheckResult]:
    """Run both Tier 5 structured checks against one selected model."""

    return [
        await run_json_object_check(client, model, declared=declared),
        await run_json_schema_check(client, model, declared=declared),
    ]
