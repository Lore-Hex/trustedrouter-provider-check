import pytest

from tr_provider_check.contract import _rotation_error_type


@pytest.mark.parametrize(
    "field", ["max_tokens", "max_completion_tokens", "temperature", "top_p"]
)
def test_unsupported_sampling_parameter_is_not_a_missing_route(field: str) -> None:
    assert (
        _rotation_error_type(
            "invalid_request_error",
            400,
            f"Unsupported parameter: '{field}' is not supported with this model.",
        )
        == "probe_config_error"
    )


def test_explicit_model_not_found_remains_an_unsupported_route() -> None:
    assert (
        _rotation_error_type("model_not_found", 404, "model does not exist")
        == "unsupported_route"
    )


def test_overload_with_parameter_word_is_still_an_outage() -> None:
    assert (
        _rotation_error_type(
            "service_unavailable_error",
            503,
            "max_tokens processing temporarily unavailable",
        )
        == "service_unavailable_error"
    )
