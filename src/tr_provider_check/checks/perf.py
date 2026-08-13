"""Tier 6: advisory production-methodology latency and throughput samples."""

from __future__ import annotations

import dataclasses
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from tr_provider_check.checks.catalog import CatalogEvidence
from tr_provider_check.contract import (
    ProviderBenchmarkSample,
    _elapsed_ms_with_clock,
    _observe_provider_stream,
    _response_error,
    aggregate_leaderboard,
    model_deadlines_declared,
)
from tr_provider_check.http import GatewayClient, provider_for_model
from tr_provider_check.report import CheckResult, CheckStatus, check_result
from tr_provider_check.sample import error_sample, sample_from_stream

DEFAULT_PERF_SAMPLES = 3
PERF_MAX_TOKENS = 512
PERF_MINIMUM_OUTPUT_TOKENS = 128
_ESTIMATED_INPUT_TOKENS_PER_SAMPLE = 64
_THROUGHPUT_PROMPT = (
    "Continue writing the lowercase word benchmark separated by single spaces "
    "until the response token limit stops you. Do not count, explain, use "
    "punctuation, or stop early."
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _spend_estimate(
    pricing: dict[str, Any] | None,
    *,
    sample_count: int,
    observed_input_tokens: int,
    observed_output_tokens: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "currency": "USD",
        "unit": "per_1m_tokens",
        "requested_billed_completions": sample_count,
        "assumed_input_tokens_per_completion": _ESTIMATED_INPUT_TOKENS_PER_SAMPLE,
        "maximum_output_tokens_per_completion": PERF_MAX_TOKENS,
        "observed_input_tokens": observed_input_tokens,
        "observed_output_tokens": observed_output_tokens,
    }
    if pricing is None:
        return {
            **base,
            "available": False,
            "reason": "no validated Catalog v2 pricing matched the selected model",
            "estimated_cap_cost_usd": None,
            "observed_cost_usd": None,
        }
    try:
        input_price = Decimal(str(pricing["input"]))
        output_price = Decimal(str(pricing["output"]))
        minimum = Decimal(str(pricing.get("minimum_request") or "0"))
    except (KeyError, InvalidOperation, ValueError):
        return {
            **base,
            "available": False,
            "reason": "validated pricing could not be converted to decimal values",
            "estimated_cap_cost_usd": None,
            "observed_cost_usd": None,
        }

    per_completion = (
        input_price * _ESTIMATED_INPUT_TOKENS_PER_SAMPLE
        + output_price * PERF_MAX_TOKENS
    ) / Decimal(1_000_000)
    per_completion = max(per_completion, minimum)
    observed = (
        input_price * observed_input_tokens + output_price * observed_output_tokens
    ) / Decimal(1_000_000)
    return {
        **base,
        "available": True,
        "input_usd_per_1m_tokens": str(input_price),
        "output_usd_per_1m_tokens": str(output_price),
        "minimum_request_usd": str(minimum),
        "estimated_cap_cost_usd": str(per_completion * sample_count),
        "observed_cost_usd": str(observed),
        "note": "The cap estimate assumes 64 input tokens for the fixed prompt; provider tokenization can vary.",
    }


def _insufficient_sample(
    sample: ProviderBenchmarkSample,
    *,
    provider: str,
    model: str,
    output_tokens: int,
    first_token_milliseconds: int | None,
    ttfb_milliseconds: int | None,
    secrets: tuple[str, ...],
) -> ProviderBenchmarkSample:
    insufficient = error_sample(
        provider=provider,
        model=model,
        source="synthetic_throughput",
        error_type="insufficient_throughput_sample",
        error_status=None,
        error_message=(
            f"output_tokens={output_tokens}; minimum={PERF_MINIMUM_OUTPUT_TOKENS}"
        ),
        elapsed_milliseconds=sample.elapsed_milliseconds,
        ttfb_milliseconds=ttfb_milliseconds,
        secrets=secrets,
    )
    insufficient.output_tokens = output_tokens
    insufficient.first_token_milliseconds = first_token_milliseconds
    return insufficient


async def run_performance_checks(
    client: GatewayClient,
    model: str,
    *,
    sample_count: int,
    evidence: CatalogEvidence,
    secrets: tuple[str, ...] = (),
) -> tuple[list[CheckResult], dict[str, Any]]:
    """Measure the stream consumed by ``llm/stream_translate.go`` and score it."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    provider = client.provider or evidence.provider_id or provider_for_model(model)
    provider = provider or "custom"
    throughput_samples: list[ProviderBenchmarkSample] = []
    latency_samples: list[ProviderBenchmarkSample] = []
    observed_input_tokens = 0
    observed_output_tokens = 0

    for _ in range(sample_count):
        started = time.perf_counter()
        try:
            async with client.stream_chat(
                model=model,
                prompt=_THROUGHPUT_PROMPT,
                temperature=0,
                max_tokens=PERF_MAX_TOKENS,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    error_type, error_status, error_message = _response_error(response)
                    throughput_samples.append(
                        error_sample(
                            provider=provider,
                            model=model,
                            source="synthetic_throughput",
                            error_type=error_type,
                            error_status=error_status,
                            error_message=error_message,
                            elapsed_milliseconds=_elapsed_ms_with_clock(
                                started, time.perf_counter
                            ),
                            secrets=secrets,
                        )
                    )
                    continue
                observation = await _observe_provider_stream(
                    response,
                    started=started,
                )
        except (httpx.HTTPError, ValueError) as error:
            throughput_samples.append(
                error_sample(
                    provider=provider,
                    model=model,
                    source="synthetic_throughput",
                    error_type=error.__class__.__name__,
                    error_status=None,
                    error_message=str(error),
                    elapsed_milliseconds=_elapsed_ms_with_clock(
                        started, time.perf_counter
                    ),
                    secrets=secrets,
                )
            )
            continue

        observed_input_tokens += observation.usage.input_tokens
        observed_output_tokens += observation.usage.output_tokens
        sample = sample_from_stream(
            provider=provider,
            model=model,
            observation=observation,
            source="synthetic_throughput",
            secrets=secrets,
        )
        if (
            sample.status == "success"
            and observation.usage.output_tokens < PERF_MINIMUM_OUTPUT_TOKENS
        ):
            sample = _insufficient_sample(
                sample,
                provider=provider,
                model=model,
                output_tokens=observation.usage.output_tokens,
                first_token_milliseconds=observation.first_token_milliseconds,
                ttfb_milliseconds=observation.ttfb_milliseconds,
                secrets=secrets,
            )
        throughput_samples.append(sample)
        if sample.status == "success":
            latency_samples.append(
                dataclasses.replace(sample, source="provider_check_performance_latency")
            )

    leaderboard_rows = [*throughput_samples, *latency_samples]
    leaderboard = aggregate_leaderboard(leaderboard_rows)
    reliability = evidence.reliability(model) or {}
    declared_first = _number(reliability.get("first_token_timeout_seconds"))
    declared_completion = _number(reliability.get("completion_timeout_seconds"))
    deadlines = model_deadlines_declared(
        model,
        declared_first_token_seconds=declared_first,
        declared_completion_seconds=declared_completion,
    )
    deadline_observations = [
        sample.first_token_milliseconds
        for sample in latency_samples
        if sample.first_token_milliseconds is not None
    ]
    deadline_ms = int(deadlines.first_token_seconds * 1000)
    within_deadline_count = sum(
        milliseconds <= deadline_ms for milliseconds in deadline_observations
    )

    models = leaderboard.get("models")
    model_stats = models[0] if isinstance(models, list) and models else None
    eligible = bool(
        isinstance(model_stats, dict) and model_stats.get("rank_eligible") is True
    )
    successful = sum(sample.status == "success" for sample in throughput_samples)
    insufficient = sum(
        sample.error_type == "insufficient_throughput_sample"
        for sample in throughput_samples
    )
    performance = {
        "advisory": True,
        "methodology": {
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": PERF_MAX_TOKENS,
            "minimum_output_tokens": PERF_MINIMUM_OUTPUT_TOKENS,
            "throughput_formula": "output_tokens * 1000 / elapsed_milliseconds_from_request_start",
            "latency_rows_reuse_same_billed_completions": True,
        },
        "requested_samples": sample_count,
        "completed_samples": len(throughput_samples),
        "successful_samples": successful,
        "insufficient_samples": insufficient,
        "leaderboard_row_count": len(leaderboard_rows),
        "leaderboard_eligible": eligible,
        "deadlines": {
            "source": "declared" if declared_first is not None else "fallback",
            "first_token_milliseconds": deadline_ms,
            "completion_milliseconds": int(deadlines.completion_seconds * 1000),
            "observed_sample_count": len(deadline_observations),
            "within_first_token_deadline_count": within_deadline_count,
        },
        "spend_estimate": _spend_estimate(
            evidence.pricing(model),
            sample_count=sample_count,
            observed_input_tokens=observed_input_tokens,
            observed_output_tokens=observed_output_tokens,
        ),
        "samples": [dataclasses.asdict(sample) for sample in throughput_samples],
        "leaderboard": leaderboard,
    }
    status: CheckStatus = "pass" if successful else "warn"
    result = check_result(
        id="perf.production-benchmark",
        tier=6,
        status=status,
        assertion=(
            "advisory samples expose TTFB, TTFT, and effective throughput using "
            "production's request-start denominator and vendored leaderboard score"
        ),
        measured={
            "requested_samples": sample_count,
            "successful_samples": successful,
            "insufficient_samples": insufficient,
            "leaderboard_eligible": eligible,
            "first_token_deadline_ms": deadline_ms,
        },
        contract_ref=(
            "enclave-go/internal/llm/byok.go (stream/include_usage); "
            "src/trusted_router/synthetic/probes.py::provider_throughput_probe"
        ),
        marketplace_bullet="Advisory latency and effective-throughput measurements use production scoring.",
        remediation=(
            "For an eligible throughput sample, stream at least 128 provider-reported "
            "output tokens and include final usage. Performance never changes conformance."
        ),
        error_type="insufficient_throughput_sample" if not successful else None,
        error_message="no eligible performance sample" if not successful else None,
    )
    return [result], performance
