"""The versioned report envelope stays valid and keeps advisory work non-gating."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from tr_provider_check.checks.assertions import CHECK_ASSERTIONS
from tr_provider_check.report import CheckStatus, check_result, report_document


def _target() -> dict[str, Any]:
    return {
        "base_url": "https://inference.example.test/v1",
        "model": "owner/model",
        "provider": "owner",
        "requested_tier": 6,
        "catalog_url": None,
        "authentication": {
            "mode": "none",
            "authorization_header_sent": False,
            "note": "none; no Authorization header was sent",
        },
        "max_sweep_models": 25,
        "perf_samples": 3,
    }


def _result(*, tier: int, status: CheckStatus):
    check_id = {
        1: "catalog.declared-v2",
        4: "stream.sse-framing",
        5: "structured.json-object",
        6: "perf.production-benchmark",
    }[tier]
    return check_result(
        id=check_id,
        tier=tier,
        status=status,
        assertion="the named test assertion holds",
        measured={"value": 1},
        contract_ref="enclave-go/internal/llm/byok.go",
        marketplace_bullet="Test report row.",
        remediation="Fix the named test behavior.",
        error_type="stream_error" if status == "fail" else None,
    )


def _schema() -> dict[str, Any]:
    path = resources.files("tr_provider_check").joinpath(
        "data/provider-report.schema.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_full_report_validates_against_checked_in_json_schema() -> None:
    document = report_document(
        [_result(tier=1, status="pass"), _result(tier=6, status="warn")],
        target=_target(),
        performance={"advisory": True, "successful_samples": 0},
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    )
    validator = Draft202012Validator(_schema(), format_checker=FormatChecker())

    validator.validate(document)
    assert document["generated_at"] == "2026-08-13T12:00:00+00:00"
    assert document["target"]["authentication"]["mode"] == "none"
    assert document["performance"]["advisory"] is True

    broken = dict(document)
    broken.pop("submission")
    with pytest.raises(ValidationError, match="submission"):
        validator.validate(broken)


def test_conformance_gate_uses_only_tiers_one_through_four() -> None:
    advisory_failure = report_document(
        [_result(tier=1, status="pass"), _result(tier=5, status="fail")],
        target=_target(),
    )
    conformance_failure = report_document(
        [_result(tier=4, status="fail"), _result(tier=6, status="warn")],
        target=_target(),
    )

    assert advisory_failure["summary"]["failed"] == 1
    assert advisory_failure["summary"]["conformance_gate"] is True
    assert advisory_failure["summary"]["provider_owned_failures"] == 1
    assert conformance_failure["summary"]["conformance_gate"] is False


def test_report_schema_enumerates_every_public_check_id() -> None:
    schema = _schema()
    ids = schema["properties"]["checks"]["items"]["properties"]["id"]["enum"]

    assert set(ids) == set(CHECK_ASSERTIONS)
    assert len(ids) == len(set(ids))
    assert schema["$id"].startswith(
        "https://raw.githubusercontent.com/Lore-Hex/trustedrouter-provider-check/"
    )
