"""Structured, secret-safe results for provider compatibility checks."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from tr_provider_check import __version__
from tr_provider_check.contract import classify_provider_failure
from tr_provider_check.snapshot import contract_version

CheckStatus = Literal["pass", "fail", "warn", "skip"]
REPORT_VERSION = "1.0"


@dataclass(frozen=True)
class CheckResult:
    """One independently actionable provider-contract assertion."""

    id: str
    tier: int
    status: CheckStatus
    assertion: str
    owner: str
    failure_class: str
    measured: dict[str, Any] = field(default_factory=dict)
    contract_ref: str = ""
    marketplace_bullet: str = ""
    remediation: str = ""


def check_result(
    *,
    id: str,
    tier: int,
    status: CheckStatus,
    assertion: str,
    measured: dict[str, Any],
    contract_ref: str,
    marketplace_bullet: str,
    remediation: str,
    error_type: str | None = None,
    error_status: int | None = None,
    error_message: str | None = None,
) -> CheckResult:
    """Build a result and use production attribution for every mapped failure."""

    attribution = classify_provider_failure(
        status="success" if status in {"pass", "skip"} else "error",
        error_type=error_type,
        error_status=error_status,
        error_message=error_message,
    )
    return CheckResult(
        id=id,
        tier=tier,
        status=status,
        assertion=assertion,
        owner=attribution.owner.value,
        failure_class=attribution.failure_class.value,
        measured=measured,
        contract_ref=contract_ref,
        marketplace_bullet=marketplace_bullet,
        remediation=remediation,
    )


#: Substrings shorter than this are not treated as credentials. Redaction is a
#: literal find-and-replace, so a one-character key turns every occurrence of
#: that letter in the report into "[REDACTED]" -- observed live as
#: "ma[REDACTED]-token-spelling" for a key of "x". Real provider keys are far
#: longer than this floor, so nothing worth hiding is skipped; short values are
#: placeholders whose exposure costs nothing next to an unreadable report.
_MIN_REDACTABLE_SECRET = 8


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if len(secret) >= _MIN_REDACTABLE_SECRET:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    if isinstance(value, dict):
        return {str(key): _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


def report_rows(
    results: list[CheckResult], *, secrets: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Return report dictionaries with credentials removed recursively."""

    rows = [dataclasses.asdict(result) for result in results]
    redacted = _redact(rows, secrets)
    assert isinstance(redacted, list)
    return redacted


def report_document(
    results: list[CheckResult],
    *,
    target: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    secrets: tuple[str, ...] = (),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the complete versioned report envelope and redact it recursively."""

    passed = sum(result.status == "pass" for result in results)
    failed = sum(result.status == "fail" for result in results)
    warned = sum(result.status == "warn" for result in results)
    skipped = sum(result.status == "skip" for result in results)
    conformance_gate = not any(
        result.status == "fail" and result.tier <= 4 for result in results
    )
    provider_owned_failures = sum(
        result.status == "fail" and result.owner == "provider" for result in results
    )
    active_target = target or {
        "base_url": "unknown",
        "model": None,
        "provider": None,
        "requested_tier": max((result.tier for result in results), default=1),
        "catalog_url": None,
        "authentication": {
            "mode": "unspecified",
            "authorization_header_sent": False,
            "note": "authentication metadata was not supplied to the renderer",
        },
        "max_sweep_models": None,
        "perf_samples": None,
    }
    document = {
        "report_version": REPORT_VERSION,
        "suite_version": __version__,
        "contract_version": contract_version,
        "generated_at": (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "target": active_target,
        "summary": {
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "skipped": skipped,
            "conformance_gate": conformance_gate,
            "provider_owned_failures": provider_owned_failures,
        },
        "checks": [dataclasses.asdict(result) for result in results],
        "performance": performance
        or {
            "advisory": True,
            "status": "not_run",
            "reason": "no performance data supplied",
        },
        "submission": {
            "signature": None,
            "signing_key_id": None,
            "canonicalization": "JCS/RFC8785",
            "nonce": None,
        },
    }
    redacted = _redact(document, secrets)
    assert isinstance(redacted, dict)
    return redacted


def report_json(
    results: list[CheckResult],
    *,
    target: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    secrets: tuple[str, ...] = (),
    generated_at: datetime | None = None,
) -> str:
    """Serialize the full stable report without exposing credentials."""

    return json.dumps(
        report_document(
            results,
            target=target,
            performance=performance,
            secrets=secrets,
            generated_at=generated_at,
        ),
        indent=2,
        sort_keys=True,
    )


def report_table(
    results: list[CheckResult],
    *,
    target: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    secrets: tuple[str, ...] = (),
) -> str:
    """Render a compact human table with the same recursive secret redaction."""

    rows = report_rows(results, secrets=secrets)
    headers = ("TIER", "STATUS", "CHECK", "ASSERTION")
    values = [
        (
            str(row["tier"]),
            str(row["status"]).upper(),
            str(row["id"]),
            str(row["assertion"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        if values
        else len(headers[index])
        for index in range(len(headers))
    ]

    def render(row: tuple[str, str, str, str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    divider = (
        "-" * widths[0],
        "-" * widths[1],
        "-" * widths[2],
        "-" * widths[3],
    )
    lines = [render(headers), render(divider)]
    lines.extend(render(row) for row in values)
    pass_count = sum(row[1] == "PASS" for row in values)
    fail_count = sum(row[1] == "FAIL" for row in values)
    warn_count = sum(row[1] == "WARN" for row in values)
    skip_count = sum(row[1] == "SKIP" for row in values)
    lines.append(
        f"Summary: {pass_count} pass, {fail_count} fail, "
        f"{warn_count} warn, {skip_count} skip"
    )
    gate = not any(result.status == "fail" and result.tier <= 4 for result in results)
    lines.append(f"Conformance gate (Tiers 1-4): {'PASS' if gate else 'FAIL'}")
    if target is not None:
        authentication = target.get("authentication")
        if isinstance(authentication, dict):
            note = authentication.get("note")
            if isinstance(note, str) and note:
                lines.append(f"Authentication: {note}")
    if performance is not None and performance.get("status") != "not_run":
        lines.append(
            "Performance (advisory): "
            f"{performance.get('successful_samples', 0)}/"
            f"{performance.get('requested_samples', 0)} eligible samples; "
            f"leaderboard eligible={performance.get('leaderboard_eligible', False)}"
        )
    return "\n".join(lines)
