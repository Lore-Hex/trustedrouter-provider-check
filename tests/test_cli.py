"""Protect the installed command's live, secret-safe endpoint interface."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from tests.mockserver.app import MockOpenAIServer
from tr_provider_check import __version__
from tr_provider_check.cli import API_KEY_ENV, main
from tr_provider_check.snapshot import contract_version


def test_no_arguments_prints_help_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    output = capsys.readouterr().out
    assert "usage: trustedrouter-provider-check" in output
    assert "--base-url" in output
    assert "--api-key" in output
    assert "--model" in output
    assert "--catalog-url" in output
    assert "--tier" in output
    assert "--json OUT" in output


def test_version_identifies_package_and_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    output = capsys.readouterr().out.strip()
    assert output == (
        f"trustedrouter-provider-check {__version__} (contract {contract_version[:12]})"
    )
    assert re.fullmatch(
        r"trustedrouter-provider-check \d+\.\d+\.\d+ \(contract [0-9a-f]{12}\)",
        output,
    )


def test_print_contract_version_emits_bare_full_hash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--print-contract-version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == contract_version


def test_help_describes_live_tiers(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "catalog, callability, chat" in output
    assert "tools, structured-output, and performance" in output
    assert "highest tier to run" in output
    assert "--perf-samples N" in output
    assert API_KEY_ENV in output


def test_unknown_flag_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--definitely-unknown"])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "unrecognized arguments: --definitely-unknown" in error


def test_live_run_requires_base_url_but_allows_keyless_endpoint(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit) as missing_url:
        main(["--tier", "1", "--api-key", "key"])
    assert missing_url.value.code == 2
    assert "--base-url is required" in capsys.readouterr().err

    exit_code = main(["--tier", "1", "--base-url", f"{mock_server.base_url}/v1"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Authentication: none; no Authorization header was sent" in output
    assert len(mock_server.request_log) == 1
    assert "authorization" not in mock_server.request_log[0].headers


def test_human_table_is_default_and_tier_is_maximum(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--base-url",
            f"{mock_server.base_url}/v1",
            "--api-key",
            "test-key",
            "--tier",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "TIER" in output and "STATUS" in output and "CHECK" in output
    assert "catalog.native-model-discovery" in output
    assert "catalog.declared-v2" in output
    assert "callability.advertised-models" not in output
    assert "Summary: 1 pass, 0 fail, 0 warn, 1 skip" in output


def test_env_key_never_appears_in_report_stdout_stderr_or_logs(
    mock_server: MockOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SENTINEL-SECRET-API-KEY-DO-NOT-PRINT"
    output_path = tmp_path / "provider-report.json"
    monkeypatch.setenv(API_KEY_ENV, sentinel)
    caplog.set_level(logging.DEBUG)

    exit_code = main(
        [
            "--base-url",
            f"{mock_server.base_url}/v1",
            "--model",
            "mock/model",
            "--tier",
            "4",
            "--json",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    rendered_json = output_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert output_path.is_file()
    document = json.loads(rendered_json)
    assert len(document["checks"]) == 18
    assert document["target"]["authentication"]["mode"] == "bearer"
    assert document["target"]["authentication"]["authorization_header_sent"] is True
    assert sentinel not in rendered_json
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in caplog.text
    authenticated = [
        record
        for record in mock_server.request_log
        if record.headers.get("authorization") == f"Bearer {sentinel}"
    ]
    assert len(authenticated) >= 1


def test_json_dash_writes_machine_report_to_stdout(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--base-url",
            f"{mock_server.base_url}/v1",
            "--api-key",
            "test-key",
            "--tier",
            "1",
            "--json",
            "-",
        ]
    )
    output = capsys.readouterr().out
    document = json.loads(output)

    assert exit_code == 0
    assert set(document) == {
        "report_version",
        "suite_version",
        "contract_version",
        "generated_at",
        "target",
        "summary",
        "checks",
        "performance",
        "submission",
    }
    assert len(document["checks"]) == 2
    assert {row["id"] for row in document["checks"]} == {
        "catalog.native-model-discovery",
        "catalog.declared-v2",
    }
    assert document["summary"] == {
        "passed": 1,
        "failed": 0,
        "warned": 0,
        "skipped": 1,
        "conformance_gate": True,
        "provider_owned_failures": 0,
    }
    assert document["submission"] == {
        "signature": None,
        "signing_key_id": None,
        "canonicalization": "JCS/RFC8785",
        "nonce": None,
    }


def test_perf_samples_must_be_positive_with_default_as_control(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as invalid:
        main(
            [
                "--base-url",
                f"{mock_server.base_url}/v1",
                "--tier",
                "1",
                "--perf-samples",
                "0",
            ]
        )
    assert invalid.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err

    assert (
        main(
            [
                "--base-url",
                f"{mock_server.base_url}/v1",
                "--tier",
                "1",
            ]
        )
        == 0
    )


def test_max_sweep_models_rejects_negative_and_accepts_zero(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as invalid:
        main(
            [
                "--base-url",
                f"{mock_server.base_url}/v1",
                "--tier",
                "1",
                "--max-sweep-models",
                "-1",
            ]
        )
    assert invalid.value.code == 2
    assert "must be non-negative" in capsys.readouterr().err

    assert (
        main(
            [
                "--base-url",
                f"{mock_server.base_url}/v1",
                "--tier",
                "1",
                "--max-sweep-models",
                "0",
            ]
        )
        == 0
    )


def test_tier_six_wires_tier_five_and_configured_perf_samples(
    mock_server: MockOpenAIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--base-url",
            f"{mock_server.base_url}/v1",
            "--model",
            "mock/model",
            "--tier",
            "6",
            "--perf-samples",
            "1",
            "--json",
            "-",
        ]
    )
    document = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    checks = {row["id"]: row for row in document["checks"]}
    assert len(checks) == len(document["checks"])
    assert checks["tools.parallel-deltas"]["status"] == "pass"
    assert checks["tools.round-trip"]["status"] == "pass"
    assert checks["structured.json-object"]["status"] == "pass"
    assert checks["structured.json-schema"]["status"] == "pass"
    assert checks["perf.production-benchmark"]["status"] == "pass"
    assert document["performance"]["requested_samples"] == 1
    assert document["performance"]["successful_samples"] == 1
    assert document["summary"]["conformance_gate"] is True

    perf_requests = [
        record
        for record in mock_server.request_log
        if isinstance(record.body, dict)
        and (
            record.body.get("max_tokens") == 512
            or record.body.get("max_completion_tokens") == 512
        )
    ]
    assert len(perf_requests) == 1
