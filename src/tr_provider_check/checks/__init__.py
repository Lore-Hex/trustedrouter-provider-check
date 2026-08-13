"""Endpoint compatibility checks grouped by contract tier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tr_provider_check.checks.callability import (
    DEFAULT_MAX_SWEEP_MODELS,
    run_callability_checks,
)
from tr_provider_check.checks.catalog import CatalogEvidence, run_catalog_checks
from tr_provider_check.checks.chat import run_chat_checks
from tr_provider_check.checks.perf import DEFAULT_PERF_SAMPLES, run_performance_checks
from tr_provider_check.checks.streaming import run_streaming_checks
from tr_provider_check.checks.structured import run_structured_checks
from tr_provider_check.checks.tools import run_tool_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult, check_result


@dataclass(frozen=True)
class CheckRun:
    """Checks plus report metadata that does not belong in a check row."""

    checks: list[CheckResult]
    performance: dict[str, Any]
    selected_model: str | None
    provider_id: str | None


async def run_checks(
    *,
    base_url: str,
    api_key: str | None,
    model: str | None = None,
    catalog_url: str | None = None,
    tier: int = 4,
    max_sweep_models: int = DEFAULT_MAX_SWEEP_MODELS,
    perf_samples: int = DEFAULT_PERF_SAMPLES,
) -> CheckRun:
    """Run every contract tier through ``tier`` against one provider endpoint."""

    if tier not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("tier must be between 1 and 6")
    if perf_samples < 1:
        raise ValueError("perf_samples must be at least 1")

    results: list[CheckResult] = []
    evidence = CatalogEvidence()
    performance: dict[str, Any] = {
        "advisory": True,
        "status": "not_run",
        "reason": "Tier 6 was not requested",
    }
    async with GatewayClient(base_url, api_key) as client:
        catalog_results, native_models = await run_catalog_checks(
            client,
            catalog_url=catalog_url,
            evidence=evidence,
        )
        results.extend(catalog_results)

        if tier >= 2:
            results.extend(
                await run_callability_checks(
                    client, native_models, max_models=max_sweep_models
                )
            )

        selected_model = model or (native_models[0] if native_models else None)
        if tier >= 3:
            if selected_model is None:
                results.append(
                    check_result(
                        id="chat.unavailable",
                        tier=3,
                        status="skip",
                        assertion="Tier 3 has a selected native model to inspect",
                        measured={
                            "reason": "no --model and native discovery returned no ids"
                        },
                        contract_ref="enclave-go/internal/llm/byok.go",
                        marketplace_bullet="A native model is available for chat checks.",
                        remediation="Fix /models discovery or pass --model explicitly.",
                    )
                )
            else:
                results.extend(await run_chat_checks(client, selected_model))

        if tier >= 4:
            if selected_model is None:
                results.append(
                    check_result(
                        id="stream.unavailable",
                        tier=4,
                        status="skip",
                        assertion="Tier 4 has a selected native model to inspect",
                        measured={
                            "reason": "no --model and native discovery returned no ids"
                        },
                        contract_ref="enclave-go/internal/llm/byok.go",
                        marketplace_bullet="A native model is available for streaming checks.",
                        remediation="Fix /models discovery or pass --model explicitly.",
                    )
                )
            else:
                results.extend(await run_streaming_checks(client, selected_model))

        if tier >= 5:
            if selected_model is None:
                for check_id, assertion, contract_ref in (
                    (
                        "tools.parallel-deltas",
                        "a selected model is available for parallel tool-delta checks",
                        "enclave-go/internal/llm/stream_translate.go",
                    ),
                    (
                        "tools.round-trip",
                        "a selected model is available for the empty-string tool round-trip",
                        "enclave-go/internal/llm/byok.go",
                    ),
                    (
                        "structured.json-object",
                        "a selected model is available for json_object response_format",
                        "enclave-go/internal/llm/byok.go",
                    ),
                    (
                        "structured.json-schema",
                        "a selected model is available for json_schema response_format",
                        "enclave-go/internal/llm/byok.go",
                    ),
                ):
                    results.append(
                        check_result(
                            id=check_id,
                            tier=5,
                            status="skip",
                            assertion=assertion,
                            measured={"reason": "no selected native model"},
                            contract_ref=contract_ref,
                            marketplace_bullet="Tier 5 capability checks have a selected model.",
                            remediation="Fix /models discovery or pass --model explicitly.",
                        )
                    )
            else:
                results.extend(
                    await run_tool_checks(
                        client,
                        selected_model,
                        declared=evidence.capability(selected_model, "tools"),
                    )
                )
                results.extend(
                    await run_structured_checks(
                        client,
                        selected_model,
                        declared=evidence.capability(
                            selected_model, "structured_output"
                        ),
                    )
                )

        if tier >= 6:
            if selected_model is None:
                results.append(
                    check_result(
                        id="perf.production-benchmark",
                        tier=6,
                        status="skip",
                        assertion="a selected model is available for advisory performance samples",
                        measured={"reason": "no selected native model"},
                        contract_ref="enclave-go/internal/llm/byok.go",
                        marketplace_bullet="Performance samples have a selected native model.",
                        remediation="Fix /models discovery or pass --model explicitly.",
                    )
                )
                performance = {
                    "advisory": True,
                    "status": "not_run",
                    "reason": "no selected native model",
                }
            else:
                perf_results, performance = await run_performance_checks(
                    client,
                    selected_model,
                    sample_count=perf_samples,
                    evidence=evidence,
                    secrets=(api_key,) if api_key else (),
                )
                results.extend(perf_results)

    return CheckRun(
        checks=results,
        performance=performance,
        selected_model=selected_model,
        provider_id=evidence.provider_id,
    )


__all__ = ["CheckRun", "run_checks"]
