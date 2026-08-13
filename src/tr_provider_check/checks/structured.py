"""Tier 5: JSON-object and JSON-Schema response-format compatibility."""

from __future__ import annotations

import json
from typing import Any

import httpx
from jsonschema import Draft202012Validator, ValidationError

from tr_provider_check.contract import _chat_text, _response_error
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult, CheckStatus, check_result

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "const": "PONG"},
        "count": {"type": "integer", "const": 2},
    },
    "required": ["answer", "count"],
    "additionalProperties": False,
}


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
        transport_status: CheckStatus = "warn" if declared is True else "skip"
        return check_result(
            id=check_id,
            tier=5,
            status=transport_status,
            assertion=assertion,
            measured={
                "reason": "response_format request failed",
                "capability_declared": declared,
                "error": error.__class__.__name__,
            },
            contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
            marketplace_bullet="Structured response formats are accepted by the selected route.",
            remediation="Configure the selected route to accept the forwarded response_format field.",
            error_type="invalid_request",
            error_message=str(error),
        )

    if response.status_code != 200:
        error_type, error_status, error_message = _response_error(response)
        rejection_status: CheckStatus = "warn" if declared is True else "skip"
        return check_result(
            id=check_id,
            tier=5,
            status=rejection_status,
            assertion=assertion,
            measured={
                "reason": "response_format was rejected",
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

    text = _chat_text(response)
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
    return check_result(
        id=check_id,
        tier=5,
        status="pass" if ok else "fail",
        assertion=assertion,
        measured={
            "capability_declared": declared,
            "capability_discovered": True,
            "http_status": response.status_code,
            "body_parsed_as_json": parse_error is None,
            "schema_valid": None if not validate_schema else validation_error is None,
            "parse_error": parse_error,
            "validation_error": validation_error,
        },
        contract_ref="enclave-go/internal/llm/byok.go (ResponseFormat passthrough)",
        marketplace_bullet="Accepted structured-output requests return conforming JSON.",
        remediation=(
            "Once response_format is accepted, return a JSON value that conforms "
            "to the requested mode and supplied JSON Schema."
        ),
        error_type="stream_error" if not ok else None,
        error_message="accepted response_format returned non-conforming output"
        if not ok
        else None,
    )


async def run_json_object_check(
    client: GatewayClient,
    model: str,
    *,
    declared: bool | None,
) -> CheckResult:
    """Check ``response_format`` forwarding from ``llm/byok.go`` as JSON object."""

    assertion = (
        "response_format type=json_object returns assistant content that parses as JSON"
    )
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

    assertion = "response_format type=json_schema returns JSON that validates against the supplied schema"
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
