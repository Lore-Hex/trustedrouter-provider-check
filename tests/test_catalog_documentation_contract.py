"""Catalog v2 validates optional per-model usage documentation."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tests.mockserver.app import _catalog
from tr_provider_check.checks.catalog import (
    _validate_vendored_rules,
    _vendored_schema,
)


def _documentation_catalog() -> dict[str, Any]:
    payload = copy.deepcopy(_catalog())
    payload["data"][0]["documentation"] = {
        "description": "Extract structured fields from a document.",
        "input_format": "Send the document text in the user message.",
        "output_format": "A JSON object containing the extracted fields.",
        "example_input": "Invoice 123 is due on 2026-09-30.",
        "example_output": '{"invoice_id":"123","due_date":"2026-09-30"}',
    }
    return payload


def test_optional_model_documentation_validates() -> None:
    payload = _documentation_catalog()

    Draft202012Validator(_vendored_schema()).validate(payload)
    _validate_vendored_rules(payload)


@pytest.mark.parametrize(
    ("mutation",),
    [
        (lambda docs: docs.pop("example_output"),),
        (lambda docs: docs.__setitem__("unknown", "not allowed"),),
        (lambda docs: docs.__setitem__("description", "x" * 1001),),
        (lambda docs: docs.__setitem__("input_format", "x" * 2001),),
        (lambda docs: docs.__setitem__("example_input", "x" * 8193),),
    ],
)
def test_model_documentation_rejects_invalid_shapes(mutation: Any) -> None:
    payload = _documentation_catalog()
    mutation(payload["data"][0]["documentation"])

    with pytest.raises(ValidationError):
        Draft202012Validator(_vendored_schema()).validate(payload)
