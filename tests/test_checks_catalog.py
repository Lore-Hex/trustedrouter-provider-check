"""Tier 1 checks fail only for their isolated mock contract defects."""

from __future__ import annotations

import asyncio

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.catalog import (
    CatalogEvidence,
    _vendored_schema,
    run_catalog_checks,
)
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult


async def _run(
    server: MockOpenAIServer,
    mode: str,
    *,
    catalog_mode: str = "conforming",
) -> tuple[list[CheckResult], list[str], CatalogEvidence]:
    evidence = CatalogEvidence()
    async with GatewayClient(
        f"{server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": mode},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            results, models = await run_catalog_checks(
                client,
                catalog_url=(f"{server.base_url}/catalog.v2.json?mode={catalog_mode}"),
                schema=_vendored_schema(),
                evidence=evidence,
            )
            return results, models, evidence


def _statuses(results: list[CheckResult]) -> dict[str, str]:
    assert len(results) == 2
    rows: dict[str, str] = {result.id: result.status for result in results}
    assert len(rows) == len(results)
    assert set(rows) == {"catalog.native-model-discovery", "catalog.declared-v2"}
    return rows


@pytest.mark.asyncio
async def test_catalog_conforming_control_is_green(
    mock_server: MockOpenAIServer,
) -> None:
    results, models, evidence = await _run(mock_server, "conforming")
    assert _statuses(results) == {
        "catalog.native-model-discovery": "pass",
        "catalog.declared-v2": "pass",
    }
    assert models == ["mock/model"]
    assert evidence.provider_id == "mock"
    assert evidence.capability("mock/model", "tools") is True
    assert evidence.capability("mock/model", "structured_output") is True
    assert evidence.reliability("mock/model") is not None
    assert evidence.pricing("mock/model") is not None


@pytest.mark.asyncio
async def test_catalog_native_discovery_rejects_empty_but_accepts_noncanonical_ids(
    mock_server: MockOpenAIServer,
) -> None:
    rejected, rejected_models, _ = await _run(mock_server, "native_models_empty")
    accepted, accepted_models, _ = await _run(mock_server, "native_noncanonical_ids")

    assert _statuses(rejected) == {
        "catalog.native-model-discovery": "fail",
        "catalog.declared-v2": "pass",
    }
    assert rejected_models == []
    assert _statuses(accepted) == {
        "catalog.native-model-discovery": "pass",
        "catalog.declared-v2": "pass",
    }
    assert accepted_models == ["deepseek-chat", "qwen3-4b:latest"]


@pytest.mark.asyncio
async def test_declared_catalog_rejects_native_id_syntax_only_in_declaration(
    mock_server: MockOpenAIServer,
) -> None:
    rejected, _, rejected_evidence = await _run(
        mock_server,
        "conforming",
        catalog_mode="invalid_declared_catalog",
    )
    accepted, _, accepted_evidence = await _run(mock_server, "conforming")

    assert _statuses(rejected) == {
        "catalog.native-model-discovery": "pass",
        "catalog.declared-v2": "fail",
    }
    assert _statuses(accepted) == {
        "catalog.native-model-discovery": "pass",
        "catalog.declared-v2": "pass",
    }
    assert rejected_evidence.declared_models == []
    assert len(accepted_evidence.declared_models) == 1


@pytest.mark.asyncio
async def test_models_without_an_object_envelope_still_yield_ids(
    mock_server: MockOpenAIServer,
) -> None:
    # pearlresearch.ai returns {"data": [...], "pricing_source": ...} with no
    # top-level "object": "list". The ids are present and the gateway never
    # reads that envelope field, so discovery must succeed rather than
    # reporting a working endpoint as having a broken catalog.
    results, models, _ = await _run(mock_server, "models_without_object_envelope")

    assert models == ["mock/model"]
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")
    assert discovery.status == "pass"


@pytest.mark.asyncio
async def test_models_without_a_data_array_still_fails(
    mock_server: MockOpenAIServer,
) -> None:
    # Negative control: dropping the envelope check must not make discovery
    # unfalsifiable.
    results, models, _ = await _run(mock_server, "models_without_data_array")

    assert models == []
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")
    assert discovery.status == "fail"
