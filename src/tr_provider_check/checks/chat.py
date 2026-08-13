"""Tier 3 checks for non-streaming OpenAI chat response semantics."""

from __future__ import annotations

from typing import Any

import httpx

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.contract import (
    PONG_PROMPT,
    _chat_text,
    _pong_matches,
    _response_error,
)
from tr_provider_check.http import (
    GatewayClient,
    gateway_omits_temperature,
    max_token_parameter,
    probe_verdict,
    provider_for_model,
)
from tr_provider_check.report import CheckResult, check_result

_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})
_SHOULD_TOLERATE_FIELDS: tuple[dict[str, Any], ...] = (
    {"top_k": 40},
    {"thinking": {"type": "disabled"}},
    {"reasoning": {"effort": "low"}},
    {"reasoning_effort": "low"},
)


def _first_choice(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0]


def _usage_values(payload: object) -> tuple[int | None, int | None, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None, None, None
    usage = payload["usage"]
    assert isinstance(usage, dict)

    def strict_int(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if value is not None:
                return (
                    value
                    if isinstance(value, int) and not isinstance(value, bool)
                    else None
                )
        return None

    return (
        strict_int("prompt_tokens", "input_tokens"),
        strict_int("completion_tokens", "output_tokens"),
        strict_int("total_tokens"),
    )


def _dependent_skip(check_id: str, assertion: str, reason: str) -> CheckResult:
    return check_result(
        id=check_id,
        tier=3,
        status="skip",
        assertion=assertion,
        measured={"reason": reason},
        contract_ref="enclave-go/internal/llm/byok.go; enclave-go/internal/llm/stream_translate.go",
        marketplace_bullet=assertion,
        remediation="Fix the base chat response first, then rerun this dependent assertion.",
    )


async def _request_status(
    client: GatewayClient, model: str, *, extra: dict[str, Any] | None = None
) -> tuple[int | None, tuple[str | None, int | None, str | None]]:
    try:
        response = await client.chat(
            model=model,
            prompt=PONG_PROMPT,
            temperature=None,
            extra=extra,
        )
    except httpx.HTTPError as error:
        return None, (error.__class__.__name__, None, None)
    if response.status_code == 200:
        return 200, (None, None, None)
    return response.status_code, _response_error(response)


async def run_chat_checks(client: GatewayClient, model: str) -> list[CheckResult]:
    """Run tier 3 checks grounded in ``byok.go`` and ``stream_translate.go``.

    Response text deliberately uses the vendored ``_chat_text`` extractor, so
    canonical strings, list parts, ``reasoning_content``, and ``reasoning`` are
    all accepted exactly as production's probe accepts them.
    """

    results: list[CheckResult] = []
    try:
        response = await client.chat(
            model=model,
            prompt=PONG_PROMPT,
            temperature=None,
        )
    except httpx.HTTPError as error:
        probe_status, reason = probe_verdict(None, declared=True)
        results.append(
            check_result(
                id="chat.pong",
                tier=3,
                status=probe_status,
                assertion=assertion_for("chat.pong"),
                measured={
                    "http_status": None,
                    "error_type": error.__class__.__name__,
                    "reason": reason,
                },
                contract_ref="enclave-go/internal/llm/byok.go; src/trusted_router/synthetic/probes.py",
                marketplace_bullet="The deterministic PONG probe returns recognizable assistant or reasoning text.",
                remediation="Make POST /chat/completions reachable and return an OpenAI-compatible JSON completion for the supplied model.",
                error_type=error.__class__.__name__,
            )
        )
        for check_id, assertion in (
            (
                "chat.non-empty",
                assertion_for("chat.non-empty"),
            ),
            ("chat.usage", assertion_for("chat.usage")),
            ("chat.model", assertion_for("chat.model")),
            ("chat.finish-reason", assertion_for("chat.finish-reason")),
        ):
            results.append(_dependent_skip(check_id, assertion, "base request failed"))
        payload: object = None
        text = ""
    else:
        if response.status_code != 200:
            probe_status, reason = probe_verdict(response.status_code, declared=True)
            error_type, error_status, error_message = _response_error(response)
            results.append(
                check_result(
                    id="chat.pong",
                    tier=3,
                    status=probe_status,
                    assertion=assertion_for("chat.pong"),
                    measured={
                        "http_status": response.status_code,
                        "reason": reason,
                    },
                    contract_ref="enclave-go/internal/llm/byok.go; vendored PONG_PROMPT/_chat_text/_pong_matches",
                    marketplace_bullet="The deterministic PONG probe returns recognizable assistant or reasoning text.",
                    remediation="Make POST /chat/completions serve the selected model; use 429/5xx only for genuinely transient failures.",
                    error_type=error_type,
                    error_status=error_status,
                    error_message=error_message,
                )
            )
            for check_id in (
                "chat.non-empty",
                "chat.usage",
                "chat.model",
                "chat.finish-reason",
            ):
                results.append(
                    _dependent_skip(
                        check_id,
                        assertion_for(check_id),
                        "base request did not return HTTP 200",
                    )
                )
            payload = None
            text = ""
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            text = _chat_text(response)
            base_ok = payload is not None
            non_empty = base_ok and bool(text.strip())
            results.append(
                check_result(
                    id="chat.non-empty",
                    tier=3,
                    status="pass" if non_empty else "fail",
                    assertion=assertion_for("chat.non-empty"),
                    measured={
                        "http_status": response.status_code,
                        "extracted_characters": len(text),
                    },
                    contract_ref="enclave-go/internal/llm/stream_translate.go (content and reasoning_content); vendored _chat_text",
                    marketplace_bullet="Chat completions contain usable assistant or reasoning output.",
                    remediation="Populate choices[0].message.content, reasoning_content, or reasoning with non-empty text; do not finish a successful PONG request with only a role or finish marker.",
                    error_type=None if non_empty else "empty_stream",
                )
            )
            pong_ok = non_empty and _pong_matches(text)
            results.append(
                check_result(
                    id="chat.pong",
                    tier=3,
                    status="pass" if pong_ok else "skip" if not non_empty else "fail",
                    assertion=assertion_for("chat.pong"),
                    measured={"matched": pong_ok, "extracted_characters": len(text)},
                    contract_ref="enclave-go/internal/llm/byok.go; vendored PONG_PROMPT/_chat_text/_pong_matches",
                    marketplace_bullet="The deterministic PONG probe returns recognizable assistant or reasoning text.",
                    remediation="For the exact prompt 'reply exactly PONG', return text containing PONG (case-insensitive); wrappers and punctuation are accepted.",
                    error_type="pong_mismatch" if non_empty and not pong_ok else None,
                )
            )

            prompt_tokens, completion_tokens, total_tokens = _usage_values(payload)
            usage_ok = (
                prompt_tokens is not None
                and completion_tokens is not None
                and total_tokens is not None
                and prompt_tokens >= 0
                and completion_tokens >= 0
                and total_tokens == prompt_tokens + completion_tokens
            )
            results.append(
                check_result(
                    id="chat.usage",
                    tier=3,
                    status="pass" if usage_ok else "fail",
                    assertion=assertion_for("chat.usage"),
                    measured={
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    contract_ref="enclave-go/internal/llm/stream_translate.go (openAIStreamUsage and settlement translation)",
                    marketplace_bullet="Usage is integer-valued and internally self-consistent.",
                    remediation="Emit integer prompt_tokens, completion_tokens, and total_tokens (or the tolerated input_tokens/output_tokens aliases in diagnostic JSON), with total_tokens exactly equal to input plus output.",
                    error_type=None if usage_ok else "provider_error",
                )
            )

            response_model = payload.get("model") if isinstance(payload, dict) else None
            model_ok = isinstance(response_model, str) and response_model == model
            results.append(
                check_result(
                    id="chat.model",
                    tier=3,
                    status="pass" if model_ok else "warn",
                    assertion=assertion_for("chat.model"),
                    measured={
                        "requested_model": model,
                        "response_model": response_model,
                    },
                    contract_ref="enclave-go/internal/llm/stream_translate.go (model is tolerated but ignored)",
                    marketplace_bullet="Completion metadata identifies the model that served the run.",
                    remediation="Set the top-level model field to the exact requested native model id, or document and consistently return the resolved serving id if aliases are intentional.",
                    error_type=None if model_ok else "provider_error",
                )
            )

            choice = _first_choice(payload)
            finish_reason = choice.get("finish_reason") if choice else None
            finish_ok = finish_reason in _FINISH_REASONS
            results.append(
                check_result(
                    id="chat.finish-reason",
                    tier=3,
                    status="pass" if finish_ok else "warn",
                    assertion=assertion_for("chat.finish-reason"),
                    measured={"finish_reason": finish_reason},
                    contract_ref="enclave-go/internal/llm/stream_translate.go::mapOpenAIFinishReason",
                    marketplace_bullet="Finish reasons map without being silently normalized to end_turn.",
                    remediation="Return stop for a natural stop, length for a token cap, tool_calls for tool invocation, or content_filter for policy filtering; do not invent provider-specific terminal values.",
                    error_type=None if finish_ok else "provider_error",
                )
            )

    provider = client.provider or provider_for_model(model)
    if gateway_omits_temperature(provider, model):
        results.append(
            check_result(
                id="chat.temperature-zero",
                tier=3,
                status="skip",
                assertion=assertion_for("chat.temperature-zero"),
                measured={
                    "reason": "the gateway omits temperature for this provider/model (byok.go openAICompatibleTemperature)"
                },
                contract_ref="enclave-go/internal/llm/byok.go::openAICompatibleTemperature; vendored _rotation_omits_temperature",
                marketplace_bullet="Gateway-supported deterministic sampling requests are accepted.",
                remediation="No change required: TrustedRouter intentionally omits this field for the selected model family.",
            )
        )
    else:
        try:
            temperature_response = await client.chat(
                model=model,
                prompt=PONG_PROMPT,
                temperature=0,
            )
            temperature_status = temperature_response.status_code
            temperature_verdict, temperature_reason = probe_verdict(
                temperature_status, declared=True
            )
            temperature_ok = temperature_response.status_code == 200
            temperature_error = (
                (None, None, None)
                if temperature_ok
                else _response_error(temperature_response)
            )
        except httpx.HTTPError as error:
            temperature_ok = False
            temperature_status = None
            temperature_verdict, temperature_reason = probe_verdict(None, declared=True)
            temperature_error = (error.__class__.__name__, None, None)
        results.append(
            check_result(
                id="chat.temperature-zero",
                tier=3,
                status=temperature_verdict,
                assertion=assertion_for("chat.temperature-zero"),
                measured={
                    "http_status": temperature_status,
                    "inconclusive": temperature_verdict == "warn",
                    "reason": temperature_reason,
                },
                contract_ref="enclave-go/internal/llm/byok.go::openAICompatibleTemperature",
                marketplace_bullet="Gateway-supported deterministic sampling requests are accepted.",
                remediation="Permit numeric temperature=0 in the chat request schema and pass it to the engine; if the model itself forbids temperature, ensure its declared provider/model identity matches a TrustedRouter omission gate.",
                error_type=temperature_error[0],
                error_status=temperature_error[1],
                error_message=temperature_error[2],
            )
        )

    field_observations: list[dict[str, Any]] = []
    tolerated_error: tuple[str | None, int | None, str | None] = (None, None, None)
    for extra in _SHOULD_TOLERATE_FIELDS:
        status_code, request_error = await _request_status(client, model, extra=extra)
        field_name = next(iter(extra))
        field_verdict, field_reason = probe_verdict(status_code, declared=False)
        field_observations.append(
            {
                "field": field_name,
                "http_status": status_code,
                "probe_status": field_verdict,
                "reason": field_reason,
            }
        )
        if status_code != 200 and tolerated_error == (None, None, None):
            tolerated_error = request_error
    fields_ok = bool(field_observations) and all(
        row["http_status"] == 200 for row in field_observations
    )
    results.append(
        check_result(
            id="chat.forwarded-fields",
            tier=3,
            # SHOULD-tolerate, not MUST: the enclave forwards these only for
            # models that declare the capability, and OpenAI itself -- the
            # reference implementation of this wire format -- answers 400 for
            # top_k, thinking and reasoning. Failing a provider for matching
            # OpenAI's own behaviour would make the report worthless, so this
            # is advisory.
            status="pass" if fields_ok else "warn",
            assertion=assertion_for("chat.forwarded-fields"),
            measured={"fields": field_observations},
            contract_ref="enclave-go/internal/llm/byok.go::buildOpenAICompatibleRequest",
            marketplace_bullet="The chat schema tolerates optional fields the gateway forwards for capable models.",
            remediation="Allow these optional keys in the request schema. Ignore a field when the selected engine does not implement it instead of rejecting the entire request; preserve implemented values when it does.",
            error_type=tolerated_error[0],
            error_status=tolerated_error[1],
            error_message=tolerated_error[2],
        )
    )

    key = max_token_parameter(provider, model)
    try:
        cap_response = await client.chat(
            model=model,
            prompt=PONG_PROMPT,
            temperature=None,
            max_tokens=16,
            max_tokens_key=key,
        )
        cap_status = cap_response.status_code
        cap_verdict, cap_reason = probe_verdict(cap_status, declared=True)
        cap_ok = cap_status == 200
        cap_error = (None, None, None) if cap_ok else _response_error(cap_response)
    except httpx.HTTPError as error:
        cap_ok = False
        cap_status = None
        cap_verdict, cap_reason = probe_verdict(None, declared=True)
        cap_error = (error.__class__.__name__, None, None)
    results.append(
        check_result(
            id="chat.max-token-spelling",
            tier=3,
            status=cap_verdict,
            assertion=assertion_for("chat.max-token-spelling"),
            measured={
                "field": key,
                "http_status": cap_status,
                "reason": cap_reason,
            },
            contract_ref="enclave-go/internal/llm/byok.go::requiresMaxCompletionTokens/buildOpenAICompatibleRequest",
            marketplace_bullet="Explicit output caps use the provider/model-compatible max-token spelling.",
            remediation=f"Accept {key} as a positive integer for this model. TrustedRouter omits all max-token fields when the caller did not explicitly set a cap and sends exactly this one spelling when it did.",
            error_type=cap_error[0],
            error_status=cap_error[1],
            error_message=cap_error[2],
        )
    )
    return results
