"""Convert live compatibility observations into vendored leaderboard rows."""

from __future__ import annotations

from datetime import UTC, datetime

from tr_provider_check.contract import (
    ProviderBenchmarkSample,
    _StreamObservation,
    _rotation_error_excluded_from_uptime,
)
from tr_provider_check.report import redact_text


def _created_at(value: datetime | None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def _scrub(message: str | None, secrets: tuple[str, ...]) -> str | None:
    if message is None:
        return None
    return redact_text(message, secrets)[:300]


def error_sample(
    *,
    provider: str,
    model: str,
    error_type: str,
    error_status: int | None,
    error_message: str | None,
    elapsed_milliseconds: int | None,
    ttfb_milliseconds: int | None = None,
    source: str = "provider_check",
    created_at: datetime | None = None,
    secrets: tuple[str, ...] = (),
) -> ProviderBenchmarkSample:
    """Build a production-compatible failure row from a live HTTP observation."""

    status = (
        "unsupported" if _rotation_error_excluded_from_uptime(error_type) else "error"
    )
    return ProviderBenchmarkSample(
        provider=provider,
        model=model,
        source=source,
        status=status,
        error_type=error_type,
        error_status=error_status,
        error_message=_scrub(error_message, secrets),
        created_at=_created_at(created_at),
        output_tokens=0,
        elapsed_milliseconds=elapsed_milliseconds,
        speed_tokens_per_second=None,
        first_token_milliseconds=None,
        ttfb_milliseconds=ttfb_milliseconds,
    )


def sample_from_stream(
    *,
    provider: str,
    model: str,
    observation: _StreamObservation,
    source: str = "provider_check",
    created_at: datetime | None = None,
    secrets: tuple[str, ...] = (),
) -> ProviderBenchmarkSample:
    """Project the vendored stream observer into ``ProviderBenchmarkSample``."""

    if observation.stream_error is not None:
        error_type, error_status, error_message = observation.stream_error
        return error_sample(
            provider=provider,
            model=model,
            source=source,
            error_type=error_type,
            error_status=error_status,
            error_message=error_message,
            elapsed_milliseconds=observation.elapsed_milliseconds,
            ttfb_milliseconds=observation.ttfb_milliseconds,
            created_at=created_at,
            secrets=secrets,
        )
    if observation.first_token_milliseconds is None:
        error_type = (
            "probe_config_error"
            if observation.finish_reason == "length"
            else "empty_stream"
        )
        return error_sample(
            provider=provider,
            model=model,
            source=source,
            error_type=error_type,
            error_status=None,
            error_message=None,
            elapsed_milliseconds=observation.elapsed_milliseconds,
            ttfb_milliseconds=observation.ttfb_milliseconds,
            created_at=created_at,
            secrets=secrets,
        )

    output_tokens = observation.usage.output_tokens
    elapsed = observation.elapsed_milliseconds
    speed = (
        output_tokens * 1000 / elapsed if output_tokens > 0 and elapsed > 0 else None
    )
    return ProviderBenchmarkSample(
        provider=provider,
        model=model,
        source=source,
        status="success",
        error_type=None,
        error_status=None,
        error_message=None,
        created_at=_created_at(created_at),
        output_tokens=output_tokens,
        elapsed_milliseconds=elapsed,
        speed_tokens_per_second=round(speed, 3) if speed is not None else None,
        first_token_milliseconds=observation.first_token_milliseconds,
        ttfb_milliseconds=observation.ttfb_milliseconds,
    )
