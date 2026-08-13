"""Tier 1 checks fail only for their isolated mock contract defects."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.checks.catalog import (
    CatalogEvidence,
    _vendored_schema,
    load_catalog_schema,
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
    # This real-world shape has readable data[] ids but no top-level
    # object:list. The gateway never reads that envelope field.
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


@pytest.mark.asyncio
async def test_bare_array_models_response_still_yields_ids(
    mock_server: MockOpenAIServer,
) -> None:
    # Together returns a top-level JSON array of 280 models rather than
    # {"data": [...]}. The ids are readable and the gateway routes from a
    # curated catalog, so discovery must succeed.
    results, models, _ = await _run(mock_server, "models_bare_array")

    assert models == ["mock/model"]
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")
    assert discovery.status == "pass"


@pytest.mark.asyncio
async def test_transient_models_failure_is_inconclusive_not_unsupported(
    mock_server: MockOpenAIServer,
) -> None:
    # A 429/5xx on /models is a capacity blip. Failing it also empties the id
    # list, so every later tier skips and the report says nothing about an
    # otherwise healthy endpoint.
    results, models, _ = await _run(mock_server, "models_transient_503")
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")

    assert discovery.status == "warn"
    assert models == []

    # Negative control: a 200 with no readable ids is still a hard failure.
    results2, _models2, _ = await _run(mock_server, "models_without_data_array")
    assert (
        next(r for r in results2 if r.id == "catalog.native-model-discovery").status
        == "fail"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("models_non_200", "models endpoint was not HTTP 200"),
        ("models_non_json", "models endpoint was not JSON"),
        ("models_duplicate_rows", "native model ids must be unique"),
    ],
)
async def test_models_permanent_http_and_malformed_payloads_fail(
    mock_server: MockOpenAIServer,
    mode: str,
    expected_error: str,
) -> None:
    results, models, _ = await _run(mock_server, mode)
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")

    assert discovery.status == "fail"
    assert discovery.measured["error"] == expected_error
    assert models == []


@pytest.mark.asyncio
async def test_one_malformed_model_row_does_not_discard_readable_ids(
    mock_server: MockOpenAIServer,
) -> None:
    results, models, _ = await _run(mock_server, "models_malformed_rows")
    discovery = next(r for r in results if r.id == "catalog.native-model-discovery")

    assert discovery.status == "pass"
    assert models == ["mock/model"]
    assert discovery.measured["unreadable_row_count"] == 2


@pytest.mark.asyncio
async def test_declared_catalog_fetch_failure_warns_validity_unknown(
    mock_server: MockOpenAIServer,
) -> None:
    results, _, evidence = await _run(
        mock_server, "conforming", catalog_mode="catalog_fetch_503"
    )
    declared = next(r for r in results if r.id == "catalog.declared-v2")

    assert declared.status == "warn"
    assert declared.measured["reason"] == "catalog URL unreachable; validity unknown"
    assert evidence.declared_models == []


@pytest.mark.asyncio
async def test_malformed_published_schema_falls_back_to_vendored_copy() -> None:
    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"type": 42})

    schema, source = await load_catalog_schema(transport=httpx.MockTransport(malformed))

    assert source == "vendored-offline-fallback"
    assert schema == _vendored_schema()


def test_catalog_evidence_identifies_only_declared_chat_serving_ids() -> None:
    evidence = CatalogEvidence(
        declared_models=[
            {
                "id": "owner/chat",
                "type": "chat",
                "endpoints": ["chat/completions"],
            },
            {"id": "owner/embed", "type": "embedding", "endpoints": []},
        ]
    )

    assert evidence.declared_chat_model_ids(["chat", "embed", "other"]) == {"chat"}
