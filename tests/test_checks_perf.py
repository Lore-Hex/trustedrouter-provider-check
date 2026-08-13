"""Tier 6 uses billed long streams and the vendored production score."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.catalog import CatalogEvidence
from tr_provider_check.checks.perf import run_performance_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult


def _evidence(*, with_pricing: bool = True) -> CatalogEvidence:
    row: dict[str, Any] = {
        "id": "mock/model",
        "reliability": {
            "first_token_timeout_seconds": 7,
            "completion_timeout_seconds": 31,
        },
    }
    if with_pricing:
        row["pricing"] = {
            "currency": "USD",
            "unit": "per_1m_tokens",
            "input": "1.0",
            "output": "2.0",
            "minimum_request": "0",
        }
    return CatalogEvidence(provider_id="mock", declared_models=[row])


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    sample_count: int = 2,
    evidence: CatalogEvidence | None = None,
) -> tuple[CheckResult, dict[str, Any]]:
    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results, performance = await run_performance_checks(
                client,
                "mock/model",
                sample_count=sample_count,
                evidence=evidence or _evidence(),
            )
    assert len(results) == 1
    assert results[0].id == "perf.production-benchmark"
    return results[0], performance


@pytest.mark.asyncio
async def test_perf_recipe_sample_count_score_deadlines_and_spend(
    mock_server: MockOpenAIServer,
) -> None:
    result, performance = await _run(mock_server, "conforming", sample_count=2)

    assert result.status == "pass"
    assert performance["advisory"] is True
    assert performance["requested_samples"] == 2
    assert performance["completed_samples"] == 2
    assert performance["successful_samples"] == 2
    assert performance["leaderboard_row_count"] == 4
    assert performance["leaderboard_eligible"] is True
    assert performance["deadlines"] == {
        "source": "declared",
        "first_token_milliseconds": 7000,
        "completion_milliseconds": 31000,
        "observed_sample_count": 2,
        "within_first_token_deadline_count": 2,
    }
    spend = performance["spend_estimate"]
    assert spend["available"] is True
    assert spend["estimated_cap_cost_usd"] == "0.002176"
    assert spend["observed_input_tokens"] == 36
    assert spend["observed_output_tokens"] == 512

    requests = [
        record
        for record in mock_server.request_log
        if record.path == "/v1/chat/completions"
    ]
    assert len(requests) == 2
    for request in requests:
        assert request.body is not None
        assert request.body["stream"] is True
        assert request.body["stream_options"] == {"include_usage": True}
        assert request.body["max_tokens"] == 512

    samples = performance["samples"]
    assert isinstance(samples, list) and len(samples) == 2
    for sample in samples:
        assert sample["source"] == "synthetic_throughput"
        assert sample["status"] == "success"
        assert sample["output_tokens"] == 256
        elapsed = sample["elapsed_milliseconds"]
        assert isinstance(elapsed, int) and elapsed > 0
        assert sample["speed_tokens_per_second"] == round(256 * 1000 / elapsed, 3)
        assert sample["first_token_milliseconds"] is not None
        assert sample["ttfb_milliseconds"] is not None

    models = performance["leaderboard"]["models"]
    assert isinstance(models, list) and len(models) == 1
    model = models[0]
    assert model["ttft_sample_count"] == 2
    assert model["throughput_sample_count"] == 2
    assert model["p50_ttft_ms"] is not None
    assert model["p50_ttfb_ms"] is not None
    assert model["p50_tokens_per_second"] is not None


@pytest.mark.asyncio
async def test_buffered_perf_still_uses_request_start_elapsed_denominator(
    mock_server: MockOpenAIServer,
) -> None:
    _, buffered = await _run(mock_server, "buffered_stream", sample_count=1)
    sample = buffered["samples"][0]

    assert sample["output_tokens"] == 256
    assert sample["elapsed_milliseconds"] >= (
        mock_server.knobs.buffered_stream_delay_seconds * 1000 * 0.8
    )
    assert sample["speed_tokens_per_second"] == round(
        sample["output_tokens"] * 1000 / sample["elapsed_milliseconds"], 3
    )


@pytest.mark.asyncio
async def test_insufficient_output_is_advisory_warning_with_negative_control(
    mock_server: MockOpenAIServer,
) -> None:
    warned, insufficient = await _run(
        mock_server,
        "perf_insufficient_output",
        sample_count=1,
    )
    mock_server.clear_requests()
    passed, control = await _run(mock_server, "conforming", sample_count=1)

    assert warned.status == "warn"
    assert warned.status != "fail"
    assert insufficient["successful_samples"] == 0
    assert insufficient["insufficient_samples"] == 1
    samples = insufficient["samples"]
    assert isinstance(samples, list) and len(samples) == 1
    assert samples[0]["status"] == "unsupported"
    assert samples[0]["error_type"] == "insufficient_throughput_sample"
    assert samples[0]["output_tokens"] == 127
    assert insufficient["leaderboard_eligible"] is False
    assert passed.status == "pass"
    assert control["successful_samples"] == 1


@pytest.mark.asyncio
async def test_perf_reports_unavailable_spend_estimate_without_catalog_pricing(
    mock_server: MockOpenAIServer,
) -> None:
    result, performance = await _run(
        mock_server,
        "conforming",
        sample_count=1,
        evidence=_evidence(with_pricing=False),
    )

    assert result.status == "pass"
    spend = performance["spend_estimate"]
    assert spend["available"] is False
    assert spend["estimated_cap_cost_usd"] is None
    assert "no validated Catalog v2 pricing" in spend["reason"]


def test_perf_rejects_zero_sample_count_with_positive_control() -> None:
    async def exercise(sample_count: int) -> None:
        server = MockOpenAIServer()
        server.start()
        try:
            async with GatewayClient(f"{server.base_url}/v1", None) as client:
                async with asyncio.timeout(HANG_GUARD_SECONDS):
                    await run_performance_checks(
                        client,
                        "mock/model",
                        sample_count=sample_count,
                        evidence=_evidence(),
                    )
        finally:
            server.close()

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(exercise(0))
    asyncio.run(exercise(1))
