import pytest
from jsonschema import Draft202012Validator, ValidationError

from tests.mockserver.app import _catalog
from tr_provider_check.checks.catalog import _vendored_schema


@pytest.mark.parametrize(
    "version,seconds,valid",
    [
        ("2.0", 900, True),
        ("2.0", 901, False),
        ("2.0", 3540, False),
        ("2.1", 3540, True),
        ("2.1", 3600, True),
        ("2.1", 3601, False),
        ("2.1", 0, False),
        ("2.1", True, False),
        ("2.1", "3540", False),
        ("2.2", 120, False),
    ],
)
def test_versioned_completion_schema(
    version: str, seconds: object, valid: bool
) -> None:
    schema = _vendored_schema()
    Draft202012Validator.check_schema(schema)
    payload = _catalog()
    payload["contract_version"] = version
    payload["data"][0]["reliability"]["completion_timeout_seconds"] = seconds
    validator = Draft202012Validator(schema)
    if valid:
        validator.validate(payload)
    else:
        with pytest.raises(ValidationError):
            validator.validate(payload)
