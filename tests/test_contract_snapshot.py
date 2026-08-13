"""Replay production exports and prove every provenance-claimed symbol is guarded."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import inspect
import json
import textwrap
from functools import partial
from typing import Any

import httpx
import pytest

from tr_provider_check import contract, snapshot

ADAPTED: dict[str, str] = {
    "model_deadlines": (
        "TrustedRouter's private trusted_router.catalog module is unavailable; "
        "the fallback policy and Provider Contract v2 declared-budget clamps are "
        "vendored as separate public-package functions."
    ),
}

# Widening this allowlist requires a deliberate edit and a written reason.
assert set(ADAPTED) == {"model_deadlines"}

# Removing the ignored provider argument makes the public fallback API honest,
# while production still passes it inside the vendored aggregator. Pin both
# sides of that one call-site adaptation so neither can drift silently.
_CALLSITE_SOURCE_ADAPTATIONS = {
    "leaderboard.aggregate_leaderboard": {
        "production": "1ae336252e85afbe0181b55ed8c37288c8f144d01ad8c6db4888eb739bf20601",
        "public": "2aa537856649489690f70d9843d9d39353ac667c96783b73883231b3b805c5cf",
    }
}

_UPSTREAM_PATHS = {
    "catalog": "scripts/pricing/provider_contract_catalog.py",
    "components": "src/trusted_router/synthetic/components.py",
    "leaderboard": "src/trusted_router/synthetic/leaderboard.py",
    "probes": "src/trusted_router/synthetic/probes.py",
    "provider_reliability": "src/trusted_router/provider_reliability.py",
    "routes": "scripts/classify_provider_routes.py",
}

# These provenance-claimed symbols are deliberately not source-hashed by the
# production exporter. Keeping every name and its behavioral/value guard here
# makes a new unpinned symbol a test failure instead of an invisible omission.
_UNHASHED_PROVENANCE_ALLOWLIST: dict[str, str] = {
    "FailureAttribution": "Its fields and enum values are exercised by failure-classification replay.",
    "FailureClass": "Every exported classification result pins its serialized enum values.",
    "FailureOwner": "Every exported classification result pins its serialized enum values.",
    "ModelDeadlines": "Deadline replay pins both fields for every exported policy row.",
    "ProviderModelStats": "The complete serialized leaderboard replay pins its data shape and methods.",
    "ProviderStats": "The complete serialized leaderboard replay pins its data shape and methods.",
    "_StreamObservation": "Live mock-stream observation tests pin every observation field.",
    "_StreamUsage": "Extractor and live stream tests pin its three token fields.",
    "_median_float": "Scalar branch tests and leaderboard replay pin its behavior.",
    "_rotation_error_excluded_from_uptime": "The exported rotation exclusion table is replayed row by row.",
    "model_deadlines_declared": "This public-package adaptation is pinned by exported deadline replay.",
    "model_deadlines_fallback": "This public-package adaptation is pinned by exported deadline replay.",
    "MONITOR_CONFIGURATION_ERROR_TYPES": "The snapshot marker table pins this constant's complete value.",
    "_ACCOUNT_QUOTA_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_CAPABILITY_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_CAPACITY_SCOPES": "The catalog frozenset replay pins this constant's complete value.",
    "_CONFIG_TYPES": "The snapshot marker table pins this constant's complete value.",
    "_CUSTOMER_QUOTA_TYPES": "The snapshot marker table pins this constant's complete value.",
    "_DEAD_MARKERS": "The snapshot marker table and classifier table pin this constant.",
    "_DEAD_STATUSES": "The snapshot marker table and classifier table pin this constant.",
    "_ENDPOINTS": "The catalog frozenset replay pins this constant's complete value.",
    "_ERROR_CONTRACT_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_FAST_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_INPUT_MODALITIES": "The catalog frozenset replay pins this constant's complete value.",
    "_LIFECYCLE_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_LIFECYCLE_STATUSES": "The catalog frozenset replay pins this constant's complete value.",
    "_MODEL_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_MODEL_ID_RE": "The catalog regex replay pins this pattern.",
    "_MODEL_V2_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_OUTPUT_MODALITIES": "The catalog frozenset replay pins this constant's complete value.",
    "_OWNER_RE": "The catalog regex replay pins this pattern.",
    "_PRICING_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_PROBE_CONFIG_ERROR_TYPES": "The snapshot marker table pins this constant's complete value.",
    "_PROBE_CONFIG_MESSAGE_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_PROVIDER_V2_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_RELIABILITY_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_SLOW_REASONING_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_STREAM_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_THROUGHPUT_PROMPT": "The exported prompt replay pins this constant's exact value.",
    "_TIMEOUT_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "_TOP_LEVEL_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_TOP_LEVEL_V2_FIELDS": "The catalog frozenset replay pins this constant's complete value.",
    "_UNSUPPORTED_ROUTE_ERROR_TYPES": "The snapshot marker table pins this constant's complete value.",
    "_UNSUPPORTED_ROUTE_MESSAGE_MARKERS": "The snapshot marker table pins this constant's complete value.",
    "PONG_PROMPT": "The exported prompt replay pins this constant's exact value.",
}


def _normalized_source(fn: Any) -> str:
    """Return stable source text while ignoring docstrings and comment-only churn."""
    source = textwrap.dedent(inspect.getsource(fn))
    parsed = ast.parse(source)
    definition = parsed.body[0]
    ignored_lines: set[int] = set()
    if (
        isinstance(definition, (ast.AsyncFunctionDef, ast.FunctionDef))
        and definition.body
    ):
        first_statement = definition.body[0]
        if (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            ignored_lines.update(
                range(
                    first_statement.lineno,
                    (first_statement.end_lineno or first_statement.lineno) + 1,
                )
            )
    lines = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if line_number in ignored_lines or not stripped or stripped.startswith("#"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def _provenance_claimed_symbols() -> set[str]:
    """Collect top-level definitions/assignments carrying an adjacent provenance tag."""
    source = inspect.getsource(contract)
    source_lines = source.splitlines()
    parsed = ast.parse(source)
    claimed: set[str] = set()
    for node in parsed.body:
        decorators = getattr(node, "decorator_list", [])
        start_line = min([node.lineno, *(item.lineno for item in decorators)])
        if start_line < 2:
            continue
        provenance = source_lines[start_line - 2]
        if not provenance.startswith(("# Source:", "# Adapted from:")):
            continue
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef)):
            claimed.add(node.name)
        elif isinstance(node, ast.Assign):
            claimed.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            claimed.add(node.target.id)
    return claimed


def _stable_result(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _stable_result(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _stable_result(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_result(item) for item in value]
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    return repr(value)


@pytest.fixture(scope="module")
def contract_data() -> dict[str, Any]:
    return snapshot.load_contract()


def test_snapshot_is_internally_consistent(contract_data: dict[str, Any]) -> None:
    assert snapshot.contract_version == contract_data["contract_version"]
    unhashed = dict(contract_data)
    expected = unhashed.pop("contract_version")
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected


def test_copy_fidelity(contract_data: dict[str, Any]) -> None:
    source_hashes = contract_data["source_hashes"]
    assert source_hashes
    assert set(ADAPTED) <= {name.rsplit(".", 1)[1] for name in source_hashes}

    verified = 0
    for qualified_name, expected_hash in source_hashes.items():
        symbol_name = qualified_name.rsplit(".", 1)[1]
        assert hasattr(contract, symbol_name), (
            f"missing vendored symbol {qualified_name}"
        )
        if symbol_name in ADAPTED:
            continue
        local_source = _normalized_source(getattr(contract, symbol_name))
        local_hash = hashlib.sha256(local_source.encode()).hexdigest()
        upstream_area = qualified_name.split(".", 1)[0]
        upstream_path = _UPSTREAM_PATHS[upstream_area]
        if qualified_name in _CALLSITE_SOURCE_ADAPTATIONS:
            adaptation = _CALLSITE_SOURCE_ADAPTATIONS[qualified_name]
            assert expected_hash == adaptation["production"], (
                f"snapshot production hash mismatch for {qualified_name}\n"
                "Authoritative side: the live quill-router exporter.\n"
                "Remediation: re-export the snapshot with "
                "`uv run python scripts/export_provider_check_contract.py`.\n"
                f"snapshot hash: {expected_hash}\n"
                f"pinned production hash: {adaptation['production']}\n"
                f"normalized local source:\n{local_source}"
            )
            assert local_hash == adaptation["public"], (
                f"public adaptation hash mismatch for {qualified_name}\n"
                "Authoritative side: the documented public call-site adaptation.\n"
                f"Remediation: re-sync this function from quill-router {upstream_path}, "
                "then reapply the documented provider-argument adaptation.\n"
                f"expected public hash: {adaptation['public']}\n"
                f"local hash: {local_hash}\n"
                f"normalized local source:\n{local_source}"
            )
        else:
            assert local_hash == expected_hash, (
                f"vendored source mismatch for {qualified_name}\n"
                "Authoritative side: the production hash in the exported snapshot.\n"
                f"Remediation: re-sync this function from quill-router {upstream_path}.\n"
                f"snapshot hash: {expected_hash}\n"
                f"local hash: {local_hash}\n"
                f"normalized local source:\n{local_source}"
            )
        verified += 1

    assert verified == len(source_hashes) - len(ADAPTED)


def test_every_provenance_claimed_symbol_has_a_guard(
    contract_data: dict[str, Any],
) -> None:
    claimed = _provenance_claimed_symbols()
    hashed = {
        qualified_name.rsplit(".", 1)[1]
        for qualified_name in contract_data["source_hashes"]
    }
    allowlisted = set(_UNHASHED_PROVENANCE_ALLOWLIST)

    assert claimed
    assert len(hashed) == len(contract_data["source_hashes"]), (
        "source_hashes contains duplicate symbol suffixes; completeness would be ambiguous"
    )
    assert allowlisted <= claimed, (
        f"remove stale provenance allowlist entries: {sorted(allowlisted - claimed)}"
    )
    assert claimed == hashed | allowlisted, (
        "Every provenance-claimed symbol must be source-hashed or explicitly "
        "allowlisted with the behavioral/value guard that pins it. "
        f"unguarded={sorted(claimed - hashed - allowlisted)}, "
        f"unexpected_hashes={sorted(hashed - claimed)}"
    )
    assert all(reason.strip() for reason in _UNHASHED_PROVENANCE_ALLOWLIST.values())


def _response(input_value: dict[str, Any]) -> httpx.Response:
    if "json" in input_value:
        return httpx.Response(input_value["status_code"], json=input_value["json"])
    return httpx.Response(input_value["status_code"], text=input_value.get("text", ""))


def _run_extractor(name: str, input_value: dict[str, Any]) -> Any:
    function_name = name.split(":", 1)[0]
    if function_name == "chat_text":
        return contract._chat_text(_response(input_value))
    if function_name == "responses_text":
        return contract._responses_text(_response(input_value))
    if function_name == "pong_matches":
        return contract._pong_matches(input_value["text"])
    if function_name in {
        "sse_line_payload",
        "sse_line_has_content",
        "sse_line_error",
        "sse_line_finish_reason",
        "sse_line_usage",
    }:
        return getattr(contract, f"_{function_name}")(input_value["line"])
    if function_name == "first_int":
        return contract._first_int(input_value["values"], *input_value["keys"])
    if function_name == "response_error":
        return contract._response_error(_response(input_value))
    raise AssertionError(f"unhandled extractor {function_name}")


def test_extractor_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["extractors"]
    assert rows
    for row in rows:
        expected = row["result"]
        exception_type = (
            getattr(builtins, expected, None) if isinstance(expected, str) else None
        )
        if isinstance(exception_type, type) and issubclass(exception_type, Exception):
            with pytest.raises(exception_type):
                _run_extractor(row["name"], row["input"])
        else:
            assert _stable_result(_run_extractor(row["name"], row["input"])) == expected


def test_decision_table_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["decision_tables"]
    assert rows
    for row in rows:
        assert (
            contract._rotation_max_tokens(row["provider"], row["model"])
            == row["max_tokens"]
        )
        assert (
            contract._rotation_omits_temperature(row["provider"], row["model"])
            == row["omits_temperature"]
        )


def _attribution_dict(attribution: contract.FailureAttribution) -> dict[str, Any]:
    return {
        "owner": attribution.owner.value,
        "failure_class": attribution.failure_class.value,
        "counts_toward_provider_availability": attribution.counts_toward_provider_availability,
        "capacity_rejected": attribution.capacity_rejected,
    }


def test_failure_classification_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["failure_classification"]
    assert rows
    for row in rows:
        actual = contract.classify_provider_failure(**row["input"])
        assert _attribution_dict(actual) == row["attribution"]


_DECLARED_DEADLINES: dict[str, tuple[float | None, float | None]] = {
    "catalog/first-floor": (1.0, 40.0),
    "catalog/first-ceiling": (400.0, 400.0),
    "catalog/completion-floor": (5.0, 10.0),
    "catalog/completion-ceiling": (100.0, 1_000.0),
    "catalog/completion-fallback": (20.0, None),
    # The exporter's BYOK catalog row is filtered out of the Credits policy.
    "catalog/usage-filter": (None, None),
}


def test_model_deadline_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["model_deadlines"]
    assert rows
    for row in rows:
        if row["model"] in _DECLARED_DEADLINES:
            first, completion = _DECLARED_DEADLINES[row["model"]]
            call = partial(
                contract.model_deadlines_declared,
                row["model"],
                declared_first_token_seconds=first,
                declared_completion_seconds=completion,
            )
        else:
            call = partial(
                contract.model_deadlines,
                row["model"],
                default_first_token_seconds=row["default_first_token_seconds"],
            )
        if "exception" in row:
            exception_type = getattr(builtins, row["exception"])
            with pytest.raises(exception_type):
                call()
        else:
            actual = call()
            assert actual.first_token_seconds == row["first_token_seconds"]
            assert actual.completion_seconds == row["completion_seconds"]


def _leaderboard_samples(
    sample_fields: list[str],
) -> list[contract.ProviderBenchmarkSample]:
    alpha = "alpha"
    beta = "beta"

    def sample(
        provider: str,
        model: str,
        created_at: str,
        **overrides: Any,
    ) -> contract.ProviderBenchmarkSample:
        values: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "source": "synthetic",
            "status": "success",
            "error_type": None,
            "error_status": None,
            "error_message": None,
            "created_at": created_at,
            "output_tokens": 0,
            "elapsed_milliseconds": None,
            "speed_tokens_per_second": None,
            "first_token_milliseconds": None,
            "ttfb_milliseconds": None,
        }
        values.update(overrides)
        if sorted(values) != sample_fields:
            raise RuntimeError("leaderboard sample fields drifted")
        return contract.ProviderBenchmarkSample(**values)

    # The aggregator does not bucket by wall clock; these fixed values only
    # exercise deterministic last-seen selection.
    return [
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T00:00:00Z",
            first_token_milliseconds=100,
            ttfb_milliseconds=80,
        ),
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T01:00:00Z",
            first_token_milliseconds=140,
            ttfb_milliseconds=100,
        ),
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T02:00:00Z",
            first_token_milliseconds=18_000,
            ttfb_milliseconds=120,
        ),
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T03:00:00Z",
            status="error",
            error_type="ReadTimeout",
        ),
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T04:00:00Z",
            source="synthetic_throughput",
            output_tokens=200,
            elapsed_milliseconds=10_000,
            speed_tokens_per_second=999.0,
        ),
        sample(
            alpha,
            "alpha/steady",
            "2026-01-01T05:00:00Z",
            source="synthetic_throughput",
            speed_tokens_per_second=30.0,
        ),
        sample(
            alpha,
            "alpha/thin",
            "2026-01-01T06:00:00Z",
            status="unsupported",
            error_type="unsupported_route",
            error_status=404,
        ),
        sample(alpha, "alpha/thin", "2026-01-01T07:00:00Z"),
        sample(
            alpha,
            "alpha/challenger",
            "2026-01-01T07:10:00Z",
            first_token_milliseconds=500,
            ttfb_milliseconds=400,
        ),
        sample(
            alpha,
            "alpha/challenger",
            "2026-01-01T07:20:00Z",
            first_token_milliseconds=600,
            ttfb_milliseconds=450,
        ),
        sample(
            alpha,
            "alpha/challenger",
            "2026-01-01T07:30:00Z",
            first_token_milliseconds=700,
            ttfb_milliseconds=500,
        ),
        sample(
            alpha,
            "alpha/challenger",
            "2026-01-01T07:40:00Z",
            first_token_milliseconds=800,
            ttfb_milliseconds=550,
        ),
        sample(
            alpha,
            "alpha/throughput-only",
            "2026-01-01T07:50:00Z",
            source="synthetic_throughput",
            output_tokens=300,
            elapsed_milliseconds=10_000,
        ),
        sample(
            beta,
            "beta/stable",
            "2026-01-01T08:00:00Z",
            first_token_milliseconds=200,
            ttfb_milliseconds=150,
        ),
        sample(
            beta,
            "beta/stable",
            "2026-01-01T09:00:00Z",
            first_token_milliseconds=250,
            ttfb_milliseconds=175,
        ),
        sample(
            beta,
            "beta/stable",
            "2026-01-01T10:00:00Z",
            status="error",
            error_type="rate_limit_error",
            error_status=429,
        ),
        sample(
            beta,
            "beta/stable",
            "2026-01-01T10:30:00Z",
            first_token_milliseconds=225,
            ttfb_milliseconds=160,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T11:00:00Z",
            first_token_milliseconds=30_000,
            ttfb_milliseconds=200,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T12:00:00Z",
            status="error",
            error_type="router_error",
            error_status=503,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T13:00:00Z",
            status="error",
            error_type="rate_limit_exceeded",
            error_status=429,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T14:00:00Z",
            status="error",
            error_type="provider_auth_config",
            error_status=401,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T15:00:00Z",
            status="error",
            error_type=None,
            error_status=500,
        ),
        sample(
            beta,
            "beta/thin",
            "2026-01-01T16:00:00Z",
            status="error",
            error_type=None,
            error_status=None,
        ),
    ]


def test_leaderboard_replay(contract_data: dict[str, Any]) -> None:
    expected = contract_data["leaderboard"]
    assert expected
    sample_fields = contract_data["sample_fields"]
    assert (
        sorted(
            field.name for field in dataclasses.fields(contract.ProviderBenchmarkSample)
        )
        == sample_fields
    )
    actual = contract.aggregate_leaderboard(
        _leaderboard_samples(sample_fields),
        min_samples=1,
        model_rank_min_samples=4,
        provider_rank_min_samples=5,
        rank_min_ttft_samples=3,
    )
    actual["models_order"] = [
        f"{row['provider']}/{row['model']}" for row in actual["models"]
    ]
    actual["providers_order"] = [row["provider"] for row in actual["providers"]]
    actual["models"] = sorted(
        actual["models"], key=lambda row: (row["provider"], row["model"])
    )
    actual["providers"] = sorted(actual["providers"], key=lambda row: row["provider"])
    assert json.loads(json.dumps(actual, sort_keys=True)) == expected


def test_catalog_decimal_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["catalog_contract"]["decimal_behavior"]
    assert rows
    for row in rows:
        if "exception" in row:
            exception_type = getattr(builtins, row["exception"])
            with pytest.raises(exception_type):
                contract._decimal(
                    row["input"], label="provider_check", nullable=row["nullable"]
                )
        else:
            actual = contract._decimal(
                row["input"], label="provider_check", nullable=row["nullable"]
            )
            assert (str(actual) if actual is not None else None) == row["result"]


def test_prompt_constants_replay(contract_data: dict[str, Any]) -> None:
    assert contract_data["prompts"] == {
        "pong": contract.PONG_PROMPT,
        "throughput": contract._THROUGHPUT_PROMPT,
    }


def test_catalog_invariants_replay(contract_data: dict[str, Any]) -> None:
    catalog = contract_data["catalog_contract"]
    assert catalog["invariants"] == {
        "error_contract": {
            "rate_limit_status": 429,
            "overload_status": 503,
            "retry_after_header": "Retry-After",
        },
        "pricing": {
            "currency": "USD",
            "unit": "per_1m_tokens",
            "minimum_request": 0,
            "cache_write_allowed_values": [None, 0],
            "prompt_caching_matches_cached_input_presence": True,
        },
    }


def test_catalog_regex_patterns_replay(contract_data: dict[str, Any]) -> None:
    catalog = contract_data["catalog_contract"]
    assert contract._MODEL_ID_RE.pattern == catalog["model_id_pattern"]
    assert contract._OWNER_RE.pattern == catalog["owner_pattern"]


def test_catalog_frozensets_replay(contract_data: dict[str, Any]) -> None:
    frozensets = contract_data["catalog_contract"]["frozensets"]
    assert frozensets
    for name, expected in frozensets.items():
        assert sorted(getattr(contract, name)) == expected


def test_marker_constants_replay(contract_data: dict[str, Any]) -> None:
    markers = {
        name: expected
        for name, expected in contract_data["markers"].items()
        if name != "router_origin_behavior"
    }
    assert markers
    for snapshot_name, expected in markers.items():
        assert expected
        base_name = snapshot_name.upper()
        symbol_name = next(
            (
                candidate
                for candidate in (f"_{base_name}", base_name)
                if hasattr(contract, candidate)
            ),
            None,
        )
        assert symbol_name is not None, (
            f"no local counterpart for marker group {snapshot_name}"
        )
        assert sorted(getattr(contract, symbol_name)) == expected


def test_router_origin_behavior_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["markers"]["router_origin_behavior"]
    assert rows
    for row in rows:
        assert contract.is_router_origin_error(row["input"]) == row["result"]


def test_rotation_error_classification_replay(contract_data: dict[str, Any]) -> None:
    rows = contract_data["rotation_errors"]["classification"]
    assert rows
    for row in rows:
        assert contract._rotation_error_type(**row["input"]) == row["result"]


def test_rotation_error_uptime_exclusion_replay(
    contract_data: dict[str, Any],
) -> None:
    rows = contract_data["rotation_errors"]["excluded_from_uptime"]
    assert rows
    for row in rows:
        assert (
            contract._rotation_error_excluded_from_uptime(row["input"]) == row["result"]
        )


def test_scalar_helper_branches() -> None:
    assert contract._elapsed_ms_with_clock(10.0, lambda: 10.125) == 125
    assert contract._median_float([]) is None
    assert contract._median_float([4.0]) == 4.0
    assert contract._median_float([9.0, 1.0, 5.0]) == 5.0
    assert contract._median_float([1.0, 3.0]) == 2.0


def test_fallback_deadline_api_cannot_silently_accept_provider() -> None:
    assert "provider" not in inspect.signature(contract.model_deadlines).parameters


def test_behavioral_replay_assertion_floor(contract_data: dict[str, Any]) -> None:
    counts = {
        "extractors": len(contract_data["extractors"]),
        "decision_tables": len(contract_data["decision_tables"]),
        "failure_classification": len(contract_data["failure_classification"]),
        "model_deadlines": len(contract_data["model_deadlines"]),
        "leaderboard_fields": len(contract_data["leaderboard"]),
        "catalog_decimal": len(contract_data["catalog_contract"]["decimal_behavior"]),
        "rotation_classification": len(
            contract_data["rotation_errors"]["classification"]
        ),
        "rotation_exclusions": len(
            contract_data["rotation_errors"]["excluded_from_uptime"]
        ),
        "router_origin": len(contract_data["markers"]["router_origin_behavior"]),
    }
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) > 300
    print("behavioral replay rows:", json.dumps(counts, sort_keys=True))
