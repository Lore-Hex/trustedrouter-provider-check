"""End-to-end regressions for live false positives and the empty-stream defect."""

from __future__ import annotations

import pytest

from tests.mockserver.app import MockOpenAIServer
from tr_provider_check.checks import run_checks
from tr_provider_check.report import report_document


@pytest.mark.asyncio
async def test_embeddings_route_first_still_grades_callable_chat_model(
    mock_server: MockOpenAIServer,
) -> None:
    mock_server.set_default_mode("models_embedding_first")
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model=None,
        tier=4,
        max_sweep_models=0,
    )

    assert run.selected_model == "mock/chat"
    assert not [row for row in run.checks if row.status == "fail"]
    assert report_document(run.checks)["summary"]["conformance_gate"] is True


@pytest.mark.asyncio
async def test_plain_429_on_every_tier_three_probe_warns_and_gate_passes(
    mock_server: MockOpenAIServer,
) -> None:
    mock_server.set_default_mode("queue_then_429")
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model="mock/model",
        tier=3,
    )
    tier_three = [row for row in run.checks if row.tier == 3]

    assert tier_three
    assert not [row for row in tier_three if row.status == "fail"]
    assert any(row.status == "warn" for row in tier_three)
    assert report_document(run.checks)["summary"]["conformance_gate"] is True


@pytest.mark.asyncio
async def test_validated_declared_dead_chat_model_fails_tier_two(
    mock_server: MockOpenAIServer,
) -> None:
    mock_server.set_default_mode("catalog_lists_404_model")
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model=None,
        catalog_url=(
            f"{mock_server.base_url}/catalog.v2.json?mode=catalog_declares_dead_chat"
        ),
        tier=2,
    )
    callability = next(
        row for row in run.checks if row.id == "callability.advertised-models"
    )

    assert callability.status == "fail"
    assert callability.measured["declared_dead_count"] == 1


@pytest.mark.asyncio
async def test_http_200_empty_body_fails_end_to_end_stream_gate(
    mock_server: MockOpenAIServer,
) -> None:
    mock_server.set_default_mode("empty_stream_200")
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model="mock/model",
        tier=4,
    )
    framing = next(row for row in run.checks if row.id == "stream.sse-framing")

    assert framing.status == "fail"
    assert framing.measured["genuinely_empty"] is True
    assert report_document(run.checks)["summary"]["conformance_gate"] is False


@pytest.mark.asyncio
async def test_gzipped_stream_preserves_tool_timing_and_incremental_delivery(
    mock_server: MockOpenAIServer,
) -> None:
    mock_server.set_default_mode("gzipped_sse")
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model="mock/model",
        tier=4,
    )
    rows = {row.id: row for row in run.checks}

    timing = rows["stream.first-delta"].measured
    assert rows["stream.sse-framing"].status == "pass"
    assert timing["first_tool_delta_milliseconds"] is not None
    assert timing["first_delta_milliseconds"] == timing["first_tool_delta_milliseconds"]
    assert rows["stream.incremental-delivery"].status == "pass"
