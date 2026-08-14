"""A provider's catalog is untrusted input, and this tool's whole job is to
survive a bad one.

Both defects here turn a malformed declaration into a crash of the entire run,
which is the worst possible outcome for a conformance checker: the provider
being checked cannot tell a broken field from a broken checker, and an
otherwise healthy endpoint reports nothing at all.

JSON permits the bare literals NaN and Infinity and Python's json module
accepts them by default, so a non-finite number in a catalog is not exotic —
it is what a naive float serializer emits for a missing value.
"""

from __future__ import annotations

import math

import pytest

from tr_provider_check.checks.perf import _number


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_declared_numbers_are_not_usable(value: float) -> None:
    """_number fed a non-finite value used to return it, and it propagated
    silently into spend and latency estimates before crashing the run."""
    assert _number(value) is None


@pytest.mark.parametrize("value", [0, 1, -1, 0.5, 1e9, -1e9])
def test_ordinary_declared_numbers_still_pass_through(value: float) -> None:
    """The guard must not disturb real declarations, including negatives — the
    caller decides what a negative means, this function only decides usable."""
    assert _number(value) == float(value)


@pytest.mark.parametrize("value", [True, False, "1.0", None, [], {}])
def test_non_numeric_declarations_are_rejected(value: object) -> None:
    assert _number(value) is None


def test_number_is_total_over_floats() -> None:
    """Whatever comes back is finite, so downstream arithmetic cannot inherit a
    NaN that only surfaces as a crash three functions later."""
    for value in [0.0, -0.0, 1e-300, 1e300, math.nan, math.inf, -math.inf]:
        result = _number(value)
        assert result is None or math.isfinite(result)


def test_catalog_validation_catches_the_decimal_runtime_error() -> None:
    """contract._decimal raises RuntimeError for a malformed decimal string.

    It was missing from the handler that guards catalog validation, so a
    catalog with a bad price field escaped as an uncaught exception rather than
    being reported as a catalog problem.

    Checked structurally with ast rather than by substring, so it cannot pass
    because "RuntimeError" appears somewhere unrelated in the module.
    """
    import ast
    import inspect

    from tr_provider_check.checks import catalog

    tree = ast.parse(inspect.getsource(catalog))
    guards_catalog_validation = [
        {
            name.id
            for name in ast.walk(handler.type)
            if isinstance(name, ast.Name)
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if handler.type is not None
    ]
    validation_handlers = [
        names for names in guards_catalog_validation if "ValidationError" in names
    ]
    assert validation_handlers, "no handler guards catalog validation any more"
    assert any("RuntimeError" in names for names in validation_handlers), (
        "the handler guarding catalog validation must catch RuntimeError; "
        "contract._decimal raises it for a malformed decimal string"
    )


def test_the_decimal_helper_really_raises_runtime_error() -> None:
    """Pins the premise of the test above: if contract._decimal ever switches
    to ValueError, the handler requirement can be relaxed."""
    from tr_provider_check import contract

    with pytest.raises(RuntimeError):
        contract._decimal("not-a-decimal", label="price")
