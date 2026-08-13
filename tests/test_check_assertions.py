"""Keep skip and graded branches on one assertion per public check id."""

from __future__ import annotations

import pytest

from tests.mockserver.app import MockOpenAIServer
from tr_provider_check.checks import run_checks
from tr_provider_check.checks.assertions import CHECK_ASSERTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "conforming",
        "native_models_empty",
        "queue_then_429",
        "empty_stream_200",
        "rejects_tools",
        "rejects_response_format",
    ],
)
async def test_every_emitted_assertion_is_keyed_by_check_id(
    mock_server: MockOpenAIServer,
    mode: str,
) -> None:
    mock_server.set_default_mode(mode)
    run = await run_checks(
        base_url=f"{mock_server.base_url}/v1",
        api_key="test-key",
        model="mock/model" if mode != "native_models_empty" else None,
        tier=5,
        perf_samples=1,
    )

    for row in run.checks:
        assert row.assertion == CHECK_ASSERTIONS[row.id]
