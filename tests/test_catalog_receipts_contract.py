"""Catalog v2 accepts only the closed optional receipt capability."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tests.mockserver.app import _catalog
from tr_provider_check.checks.catalog import (
    CatalogEvidence,
    _validate_vendored_rules,
    _vendored_schema,
)


def _receipt_catalog() -> dict[str, Any]:
    payload = copy.deepcopy(_catalog())
    payload["data"][0]["capabilities"]["receipts"] = {
        "spec": "inference-receipt/1",
        "algorithms": ["EdDSA"],
        "delivery": ["header", "stream-chunk"],
    }
    return payload


def test_optional_receipt_capability_validates_and_is_exposed_as_evidence() -> None:
    payload = _receipt_catalog()

    Draft202012Validator(_vendored_schema()).validate(payload)
    _validate_vendored_rules(payload)
    evidence = CatalogEvidence(declared_models=payload["data"])

    assert evidence.receipt_capability("mock/model") == {
        "spec": "inference-receipt/1",
        "algorithms": ["EdDSA"],
        "delivery": ["header", "stream-chunk"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spec", "inference-receipt/2"),
        ("algorithms", ["ES256"]),
        ("delivery", ["body"]),
    ],
)
def test_receipt_capability_rejects_unknown_vocabulary(
    field: str, value: object
) -> None:
    payload = _receipt_catalog()
    payload["data"][0]["capabilities"]["receipts"][field] = value

    with pytest.raises(ValidationError):
        Draft202012Validator(_vendored_schema()).validate(payload)
    with pytest.raises(ValueError, match="unsupported"):
        _validate_vendored_rules(payload)
