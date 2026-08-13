"""Exercise request construction, retries, reporting, and sample projection."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from datetime import UTC, datetime

import pytest

from tests.mockserver.app import HANG_GUARD_SECONDS, MockOpenAIServer
from tr_provider_check.contract import _StreamObservation, _StreamUsage
from tr_provider_check.http import (
    provider_for_base_url,
    GatewayClient,
    max_token_parameter,
    provider_for_model,
)
from tr_provider_check.report import (
    CheckResult,
    check_result,
    report_json,
    report_rows,
    report_table,
)
from tr_provider_check.sample import error_sample, sample_from_stream


def test_gateway_request_recipe_matches_enclave_field_rules() -> None:
    client = GatewayClient("https://example.test/v1/", "secret")
    try:
        ordinary = client.chat_body(
            model="deepseek-chat",
            prompt="hello",
            stream=True,
            temperature=0,
        )
        capped = client.chat_body(
            model="openai/gpt-5.1",
            prompt="hello",
            stream=False,
            max_tokens=17,
            extra={"top_k": 4},
        )
    finally:
        asyncio.run(client.aclose())

    assert ordinary == {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
    }
    assert capped == {
        "model": "openai/gpt-5.1",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "max_completion_tokens": 17,
        "top_k": 4,
    }
    assert provider_for_model("deepseek-chat") == ""
    assert provider_for_model("openai/gpt-5.1") == "openai"
    assert max_token_parameter("openai", "gpt-5.1") == "max_completion_tokens"
    assert max_token_parameter("mock", "gpt-5.1") == "max_tokens"


@pytest.mark.asyncio
async def test_gateway_retries_transient_status_only_before_body_exposure(
    mock_server: MockOpenAIServer,
) -> None:
    delays: list[float] = []

    async def record_delay(value: float) -> None:
        delays.append(value)

    async with GatewayClient(
        f"{mock_server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": "queue_then_429"},
        sleep=record_delay,
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            async with client.stream_chat(
                model="mock/model",
                prompt="hello",
            ) as response:
                assert response.status_code == 429
                await response.aread()

    requests = [
        record
        for record in mock_server.request_log
        if record.path == "/v1/chat/completions"
    ]
    assert len(requests) == 3
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_gateway_does_not_retry_after_successful_headers_expose_stream(
    mock_server: MockOpenAIServer,
) -> None:
    async with GatewayClient(
        f"{mock_server.base_url}/v1",
        "test-key",
        headers={"X-Mock-Mode": "midstream_error"},
    ) as client:
        async with asyncio.timeout(HANG_GUARD_SECONDS):
            async with client.stream_chat(
                model="mock/model",
                prompt="hello",
            ) as response:
                assert response.status_code == 200
                body = await response.aread()

    assert b'"error"' in body
    requests = [
        record
        for record in mock_server.request_log
        if record.path == "/v1/chat/completions"
    ]
    assert len(requests) == 1


def test_reports_redact_nested_secrets_and_preserve_attribution() -> None:
    sentinel = "SENTINEL-SECRET"
    result = check_result(
        id="test.failure",
        tier=4,
        status="fail",
        assertion=f"failure echoed {sentinel}",
        measured={"nested": [f"Bearer {sentinel}"]},
        contract_ref="enclave-go/internal/llm/byok.go",
        marketplace_bullet="test",
        remediation=f"remove {sentinel}",
        error_type="stream_error",
    )

    rows = report_rows([result], secrets=(sentinel,))
    rendered_json = report_json([result], secrets=(sentinel,))
    rendered_table = report_table([result], secrets=(sentinel,))
    serialized_rows = json.dumps(rows)

    assert len(rows) == 1
    assert rows[0]["owner"] == "provider"
    assert rows[0]["failure_class"] == "provider_stream"
    assert sentinel not in serialized_rows
    assert sentinel not in rendered_json
    assert sentinel not in rendered_table
    assert serialized_rows.count("[REDACTED]") == 3
    assert "Summary: 0 pass, 1 fail, 0 warn, 0 skip" in rendered_table


def test_report_table_handles_an_explicitly_empty_result_set() -> None:
    rendered = report_table([])
    assert "TIER" in rendered
    assert "Summary: 0 pass, 0 fail, 0 warn, 0 skip" in rendered


def test_stream_samples_cover_success_empty_length_and_redacted_error() -> None:
    created_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    success = sample_from_stream(
        provider="mock",
        model="mock/model",
        observation=_StreamObservation(
            ttfb_milliseconds=10,
            first_token_milliseconds=20,
            last_token_milliseconds=80,
            elapsed_milliseconds=100,
            finish_reason="stop",
            usage=_StreamUsage(input_tokens=4, output_tokens=5),
        ),
        created_at=created_at,
    )
    empty = sample_from_stream(
        provider="mock",
        model="mock/model",
        observation=_StreamObservation(elapsed_milliseconds=30),
        created_at=created_at,
    )
    capped = sample_from_stream(
        provider="mock",
        model="mock/model",
        observation=_StreamObservation(
            elapsed_milliseconds=30,
            finish_reason="length",
        ),
        created_at=created_at,
    )
    sentinel = "SENTINEL-KEY"
    errored = error_sample(
        provider="mock",
        model="mock/model",
        error_type="unsupported_route",
        error_status=404,
        error_message=f"key={sentinel}" + "x" * 400,
        elapsed_milliseconds=40,
        secrets=(sentinel,),
        created_at=created_at,
    )

    assert success.status == "success"
    assert success.output_tokens == 5
    assert success.speed_tokens_per_second == 50.0
    assert success.created_at == "2026-01-02T03:04:00+00:00"
    assert empty.status == "error" and empty.error_type == "empty_stream"
    assert capped.status == "unsupported"
    assert capped.error_type == "probe_config_error"
    assert errored.status == "unsupported"
    assert errored.error_message is not None
    assert sentinel not in errored.error_message
    assert len(errored.error_message) == 300


def _result(assertion: str, measured: dict[str, Any] | None = None) -> CheckResult:
    return check_result(
        id="chat.max-token-spelling",
        tier=3,
        status="fail",
        assertion=assertion,
        measured=measured or {},
        contract_ref="byok.go",
        marketplace_bullet="max_token_spelling",
        remediation="Accept the max-token spelling the enclave sends.",
        error_type="provider_error",
        error_status=500,
    )


def test_short_placeholder_keys_do_not_corrupt_the_report() -> None:
    # Redaction is a literal replace, so a one-character key once turned every
    # "x" in the report into "[REDACTED]" ("ma[REDACTED]-token-spelling"),
    # observed on the first live run. Below the credential floor a value is a
    # placeholder, not a secret.
    rows = report_rows(
        [_result("the provider accepts the max-token spelling")], secrets=("x",)
    )

    assert rows[0]["id"] == "chat.max-token-spelling"
    assert "[REDACTED]" not in rows[0]["assertion"]
    assert "max-token" in rows[0]["assertion"]


def test_real_length_keys_are_still_redacted() -> None:
    secret = "sk-live-abcdef0123456789"
    rows = report_rows(
        [_result("request failed", {"url": f"https://api.example/v1?key={secret}"})],
        secrets=(secret,),
    )

    assert secret not in json.dumps(rows)
    assert "[REDACTED]" in rows[0]["measured"]["url"]


def test_json_endpoints_are_not_asked_for_event_stream(
    mock_server: MockOpenAIServer,
) -> None:
    # A live run against z.ai returned 406 Not Acceptable for GET /models
    # because Accept: text/event-stream was a client-wide default. The tool
    # then reported it as the provider's broken catalog. Only the streaming
    # completion may ask for SSE.
    async def _exercise() -> None:
        async with GatewayClient(f"{mock_server.base_url}/v1", "k" * 24) as client:
            await client.models()
            async with client.stream_chat(
                model="mock/model", prompt="reply exactly PONG"
            ) as response:
                await response.aread()

    asyncio.run(_exercise())

    by_path: dict[str, str] = {}
    for record in mock_server.request_log:
        by_path[record.path] = record.headers.get("accept", "")

    assert by_path["/v1/models"] == "application/json"
    assert by_path["/v1/chat/completions"] == "text/event-stream"


@pytest.mark.parametrize(
    ("base_url", "model", "expected"),
    [
        # A bare native id has no "owner/" prefix, so the provider must come
        # from the host or byok.go's gpt-5.x rename never fires and the tool
        # blames OpenAI for rejecting max_tokens.
        ("https://api.openai.com/v1", "gpt-5.4-nano", "max_completion_tokens"),
        ("https://api.openai.com/v1", "gpt-4o-mini", "max_tokens"),
        ("https://api.z.ai/api/paas/v4", "glm-4.6", "max_tokens"),
        ("https://inference.example.com/v1", "custom-model", "max_tokens"),
    ],
)
def test_max_token_spelling_follows_the_endpoint_host(
    base_url: str, model: str, expected: str
) -> None:
    provider = provider_for_base_url(base_url)
    assert max_token_parameter(provider, model) == expected
