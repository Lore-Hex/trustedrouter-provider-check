"""Tier 1: native model discovery and declared marketplace catalog checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from collections.abc import Iterable
from typing import Any

import httpx
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.contract import (
    _CAPABILITY_FIELDS,
    _ERROR_CONTRACT_FIELDS,
    _LIFECYCLE_FIELDS,
    _MODEL_ID_RE,
    _MODEL_V2_FIELDS,
    _OWNER_RE,
    _PRICING_FIELDS,
    _PROVIDER_V2_FIELDS,
    _RELIABILITY_FIELDS,
    _TOP_LEVEL_V2_FIELDS,
    _decimal,
)
from tr_provider_check.http import GatewayClient, probe_inconclusive, probe_verdict
from tr_provider_check.report import CheckResult, check_result

SCHEMA_URL = "https://trustedrouter.com/providers/marketplace/catalog.v2.schema.json"


@dataclass(frozen=True)
class NativeModels:
    ids: list[str]
    response_status: int | None
    error: str | None = None
    capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)
    unreadable_rows: int = 0


@dataclass
class CatalogEvidence:
    """Validated capability, reliability, and pricing evidence for later tiers."""

    provider_id: str | None = None
    native_capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)
    declared_models: list[dict[str, Any]] = field(default_factory=list)

    def _declared_matches(self, model: str) -> list[dict[str, Any]]:
        exact = [row for row in self.declared_models if row.get("id") == model]
        if exact:
            return exact
        native = model.split("/", 1)[-1]
        suffix = [
            row
            for row in self.declared_models
            if str(row.get("id") or "").split("/", 1)[-1] == native
        ]
        return suffix if len(suffix) == 1 else []

    def capability(self, model: str, name: str) -> bool | None:
        """Return true/false only when a native or validated declaration says so."""

        claims: list[bool] = []
        native = self.native_capabilities.get(model, {}).get(name)
        if isinstance(native, bool):
            claims.append(native)
        for row in self._declared_matches(model):
            capabilities = row.get("capabilities")
            value = capabilities.get(name) if isinstance(capabilities, dict) else None
            if isinstance(value, bool):
                claims.append(value)
        if any(claims):
            return True
        return False if claims else None

    def reliability(self, model: str) -> dict[str, Any] | None:
        matches = self._declared_matches(model)
        if len(matches) != 1:
            return None
        value = matches[0].get("reliability")
        return value if isinstance(value, dict) else None

    def pricing(self, model: str) -> dict[str, Any] | None:
        matches = self._declared_matches(model)
        if len(matches) != 1:
            return None
        value = matches[0].get("pricing")
        return value if isinstance(value, dict) else None

    def declared_chat_model_ids(self, models: Iterable[str]) -> set[str]:
        """Return native ids backed by a validated chat/completions declaration."""

        declared: set[str] = set()
        for model in models:
            for row in self._declared_matches(model):
                endpoints = row.get("endpoints")
                if (
                    row.get("type") == "chat"
                    and isinstance(endpoints, list)
                    and "chat/completions" in endpoints
                ):
                    declared.add(model)
                    break
        return declared


def _exact_fields(value: object, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise ValueError(
            f"{label} fields invalid: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_vendored_rules(payload: object) -> None:
    """Apply id, decimal, and exact-field rules exported in ``contract.py``."""

    _exact_fields(payload, _TOP_LEVEL_V2_FIELDS, "catalog")
    assert isinstance(payload, dict)
    provider = payload["provider"]
    _exact_fields(provider, _PROVIDER_V2_FIELDS, "catalog.provider")
    assert isinstance(provider, dict)
    provider_id = provider.get("id")
    if not isinstance(provider_id, str) or _OWNER_RE.fullmatch(provider_id) is None:
        raise ValueError("catalog.provider.id is invalid")
    _exact_fields(
        provider.get("error_contract"),
        _ERROR_CONTRACT_FIELDS,
        "catalog.provider.error_contract",
    )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("catalog.data must be an array")
    for index, row in enumerate(rows):
        label = f"catalog.data[{index}]"
        _exact_fields(row, _MODEL_V2_FIELDS, label)
        assert isinstance(row, dict)
        model_id = row.get("id")
        if not isinstance(model_id, str) or _MODEL_ID_RE.fullmatch(model_id) is None:
            raise ValueError(f"{label}.id is not a declared owner/model id")
        owner = row.get("owned_by")
        if not isinstance(owner, str) or _OWNER_RE.fullmatch(owner) is None:
            raise ValueError(f"{label}.owned_by is invalid")
        _exact_fields(
            row.get("capabilities"), _CAPABILITY_FIELDS, f"{label}.capabilities"
        )
        _exact_fields(row.get("pricing"), _PRICING_FIELDS, f"{label}.pricing")
        _exact_fields(row.get("lifecycle"), _LIFECYCLE_FIELDS, f"{label}.lifecycle")
        _exact_fields(
            row.get("reliability"), _RELIABILITY_FIELDS, f"{label}.reliability"
        )
        pricing = row["pricing"]
        assert isinstance(pricing, dict)
        _decimal(pricing.get("input"), label=f"{label}.pricing.input")
        _decimal(pricing.get("output"), label=f"{label}.pricing.output")
        _decimal(
            pricing.get("cached_input"),
            label=f"{label}.pricing.cached_input",
            nullable=True,
        )
        _decimal(
            pricing.get("cache_write"),
            label=f"{label}.pricing.cache_write",
            nullable=True,
        )
        _decimal(
            pricing.get("minimum_request"), label=f"{label}.pricing.minimum_request"
        )


def _vendored_schema() -> dict[str, Any]:
    schema_file = resources.files("tr_provider_check").joinpath(
        "data/catalog.v2.schema.json"
    )
    payload = json.loads(schema_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("vendored catalog schema root must be an object")
    return payload


async def load_catalog_schema(
    *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[dict[str, Any], str]:
    """Fetch the published v2 schema, falling back to the packaged copy."""

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0), transport=transport, follow_redirects=True
        ) as client:
            response = await client.get(SCHEMA_URL)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("schema root is not an object")
            Draft202012Validator.check_schema(payload)
            return payload, "published"
    except (httpx.HTTPError, SchemaError, ValueError, TypeError):
        return _vendored_schema(), "vendored-offline-fallback"


async def discover_native_models(client: GatewayClient) -> NativeModels:
    """Read endpoint ids without applying the marketplace owner/model regex.

    ``enclave-go/internal/llm/byok.go`` sends the selected upstream model id
    unchanged. Native ids such as ``deepseek-chat`` and ``qwen3-4b:latest``
    are therefore valid even though declared marketplace ids use owner/model.
    """

    try:
        response = await client.models()
    except httpx.HTTPError as error:
        return NativeModels([], None, error.__class__.__name__)
    if response.status_code != 200:
        # A 429/5xx on /models is a capacity blip, not an unsupported route.
        # Failing here also empties native.ids, so every later tier skips and
        # the report says nothing about an otherwise healthy endpoint.
        return NativeModels(
            [],
            response.status_code,
            "models endpoint was unavailable during discovery"
            if probe_inconclusive(response.status_code)
            else "models endpoint was not HTTP 200",
        )
    try:
        payload = response.json()
    except ValueError:
        return NativeModels([], response.status_code, "models endpoint was not JSON")
    # Read ids from data[] rather than gating on the envelope's object field.
    # One live endpoint returns {"data": [...]} with valid ids and no top-level
    # "object": "list"; requiring it found zero models and blocked downstream
    # tiers. Another returns a bare top-level array of hundreds of models.
    # Both shapes have perfectly readable ids, and
    # the gateway routes from a curated catalog rather than this envelope.
    # Accept any shape whose model ids can be read, and let the declared-v2
    # check police the marketplace declaration.
    if isinstance(payload, list):
        rows: object = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    else:
        return NativeModels(
            [], response.status_code, "models response has no readable model list"
        )
    if not isinstance(rows, list) or not rows:
        return NativeModels([], response.status_code, "models data must be non-empty")
    ids: list[str] = []
    unreadable_rows = 0
    capabilities: dict[str, dict[str, bool]] = {}
    for row in rows:
        model_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(model_id, str) or not model_id.strip():
            # One malformed row must not discard the readable ids beside it:
            # Together advertises 280 models, and a single bad entry used to
            # throw away all of them.
            unreadable_rows += 1
            continue
        ids.append(model_id)
        assert isinstance(row, dict)
        raw_capabilities = row.get("capabilities")
        if isinstance(raw_capabilities, dict):
            discovered = {
                str(key): value
                for key, value in raw_capabilities.items()
                if isinstance(value, bool)
            }
            if discovered:
                capabilities[model_id] = discovered
    if len(ids) != len(set(ids)):
        return NativeModels([], response.status_code, "native model ids must be unique")
    return NativeModels(
        ids,
        response.status_code,
        capabilities=capabilities,
        unreadable_rows=unreadable_rows,
    )


async def run_catalog_checks(
    client: GatewayClient,
    *,
    catalog_url: str | None,
    schema: dict[str, Any] | None = None,
    schema_transport: httpx.AsyncBaseTransport | None = None,
    evidence: CatalogEvidence | None = None,
) -> tuple[list[CheckResult], list[str]]:
    """Run tier 1 checks justified by ``llm/byok.go`` plus Catalog v2."""

    native = await discover_native_models(client)
    if evidence is not None:
        evidence.native_capabilities.update(native.capabilities)
    native_ok = bool(native.ids) and native.error is None
    native_probe_status, native_probe_reason = probe_verdict(
        native.response_status, declared=True
    )
    native_status = (
        "pass"
        if native_ok
        else native_probe_status
        if native.response_status != 200
        else "fail"
    )
    results = [
        check_result(
            id="catalog.native-model-discovery",
            tier=1,
            status=native_status,
            assertion=assertion_for("catalog.native-model-discovery"),
            measured={
                "http_status": native.response_status,
                "model_count": len(native.ids),
                "model_ids": native.ids,
                "unreadable_row_count": native.unreadable_rows,
                "error": native.error,
                "reason": native_probe_reason
                if native.response_status != 200
                else None,
            },
            contract_ref="enclave-go/internal/llm/byok.go (upstreamID is passed through)",
            marketplace_bullet="Native /v1/models ids are discoverable and are not forced into owner/model syntax.",
            remediation=(
                'Return HTTP 200 JSON shaped as {"object":"list","data":[{"id":...}]}; '
                "include every chat-served native id exactly once. Do not rename engine-native ids merely to satisfy the marketplace catalog regex."
            ),
            error_type="unsupported_route" if native_status == "fail" else None,
            error_status=native.response_status,
            error_message=native.error,
        )
    ]

    if catalog_url is None:
        results.append(
            check_result(
                id="catalog.declared-v2",
                tier=1,
                status="skip",
                assertion=assertion_for("catalog.declared-v2"),
                measured={"reason": "--catalog-url was not supplied"},
                contract_ref=f"{SCHEMA_URL}; enclave-go/internal/llm/byok.go",
                marketplace_bullet="Provider Reliability Contract v2 catalog is machine-valid.",
                remediation="Publish the marketplace declaration and pass its public URL with --catalog-url.",
            )
        )
        return results, native.ids

    active_schema: dict[str, Any]
    schema_source: str
    if schema is None:
        active_schema, schema_source = await load_catalog_schema(
            transport=schema_transport
        )
    else:
        active_schema, schema_source = schema, "test-supplied"
    error_text: str | None = None
    fetch_error = False
    catalog_error_type: str | None = None
    declared_rows = 0
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0), follow_redirects=True
        ) as public_client:
            response = await public_client.get(catalog_url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        fetch_error = True
        catalog_error_type = error.__class__.__name__
        error_text = f"{error.__class__.__name__}: {error}"
    else:
        try:
            payload = response.json()
            Draft202012Validator(
                active_schema, format_checker=FormatChecker()
            ).validate(payload)
            _validate_vendored_rules(payload)
            assert isinstance(payload, dict)
            data = payload.get("data")
            declared_rows = len(data) if isinstance(data, list) else 0
            if evidence is not None:
                provider = payload.get("provider")
                if isinstance(provider, dict) and isinstance(provider.get("id"), str):
                    evidence.provider_id = provider["id"]
                if isinstance(data, list):
                    evidence.declared_models.extend(
                        row for row in data if isinstance(row, dict)
                    )
        except (
            AssertionError,
            KeyError,
            TypeError,
            ValidationError,
            ValueError,
            # contract._decimal raises RuntimeError for a malformed decimal
            # string (contract.py:1519-1525). It was absent from this list, so
            # a provider catalog with a bad price field crashed the entire run
            # instead of being reported as a catalog problem — the one input
            # this checker exists to be robust against.
            RuntimeError,
        ) as error:
            catalog_error_type = error.__class__.__name__
            error_text = f"{error.__class__.__name__}: {error}"
    catalog_ok = error_text is None
    results.append(
        check_result(
            id="catalog.declared-v2",
            tier=1,
            status="pass" if catalog_ok else "warn" if fetch_error else "fail",
            assertion=assertion_for("catalog.declared-v2"),
            measured={
                "schema_source": schema_source,
                "declared_model_count": declared_rows,
                "error": error_text,
                "reason": (
                    "catalog URL unreachable; validity unknown" if fetch_error else None
                ),
            },
            contract_ref=f"{SCHEMA_URL}; scripts/pricing/provider_contract_catalog.py; enclave-go/internal/llm/byok.go",
            marketplace_bullet="Provider Reliability Contract v2 catalog is machine-valid.",
            remediation=(
                "Validate the public declaration against catalog.v2.schema.json, emit every required field and no extras, use canonical owner/model ids only in this declaration, and encode every price as a non-negative decimal string. Keep /v1/models native ids unchanged."
            ),
            error_type=(
                catalog_error_type
                if fetch_error
                else "probe_config_error"
                if not catalog_ok
                else None
            ),
            error_message=error_text,
        )
    )
    return results, native.ids
