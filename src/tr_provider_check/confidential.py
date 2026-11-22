"""Synthetic chat probe over Tinfoil's attested, TLS-key-pinned transport."""

import argparse
import importlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

_PIN = re.compile(r"[\w.-]+/[\w.-]+(?:@[\w.-]+)?@sha256:[0-9a-f]{64}", re.ASCII)
_LABEL = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?")


class ProbeFailure(Exception):
    """Only fixed reason codes, never remote messages or content."""


def _client(host: str, repo: str | None, project: str | None) -> Any:
    try:
        sdk = importlib.import_module("tinfoil")
    except ImportError:
        raise ProbeFailure("install_cc_extra") from None
    pin = sdk.Pin(repo=repo) if repo else sdk.Pin(project=project)
    return sdk.EnclaveHTTPClient(host, pin, transport_mode="tls", timeout=30.0)


def _canary(client: Any, host: str, model: str, key: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return only the word PONG."}],
        "max_tokens": 16,
        "stream": False,
    }
    with client.http_client.stream(
        "POST",
        f"https://{host}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Accept-Encoding": "identity"},
        json=payload,
        follow_redirects=False,
    ) as response:
        if response.status_code != 200:
            raise ProbeFailure(f"http_{response.status_code}")
        data = bytearray()
        for block in response.iter_bytes(chunk_size=8192):
            if len(data) + len(block) > 65536:
                raise ProbeFailure("oversized_response")
            data.extend(block)
        try:
            result = json.loads(data)
            choice = result["choices"][0]
            matches = (
                result["model"] == model
                and choice["finish_reason"] == "stop"
                and choice["message"]["content"].strip() == "PONG"
            )
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            matches = False
        if not matches:
            raise ProbeFailure("unexpected_canary_response")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", required=True, help="Direct model enclave DNS name; no URL or proxy"
    )
    parser.add_argument("--model", required=True, help="Exact serving model ID")
    pins = parser.add_mutually_exclusive_group(required=True)
    pins.add_argument("--repo-pin", help="v2 signed owner/repo@sha256:<release digest>")
    pins.add_argument(
        "--project-pin", help="v3 signed org/project@sha256:<config digest>"
    )
    parser.add_argument(
        "--send-canary",
        required=True,
        action="store_true",
        help="Consent to a billable synthetic request",
    )
    args = parser.parse_args(argv)
    if (
        len(args.host) > 253
        or "." not in args.host
        or not all(_LABEL.fullmatch(x) for x in args.host.split("."))
    ):
        parser.error("host must be a DNS name, not a URL")
    if not _PIN.fullmatch(args.repo_pin or args.project_pin):
        parser.error(
            "supply an independent immutable sha256 pin, not a floating reference"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,191}", args.model):
        parser.error("invalid model ID")
    key = os.environ.get("TR_PROVIDER_API_KEY")
    if not key:
        parser.error("set TR_PROVIDER_API_KEY in your environment")
    stage = "attestation"
    report = dict(
        profile="tinfoil-direct",
        host=args.host,
        model=args.model,
        pin=args.repo_pin or args.project_pin,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    try:
        with _client(args.host, args.repo_pin, args.project_pin) as client:
            stage = "attested_chat"
            _canary(client, args.host, args.model, key)
    except Exception as exc:
        reason = (
            str(exc) if isinstance(exc, ProbeFailure) else "verifier_or_transport_error"
        )
        print(
            json.dumps({**report, "result": "failed", "stage": stage, "reason": reason})
        )
        return 1
    print(
        json.dumps(
            {
                **report,
                "result": "attested_chat_passed",
                "marketplace_review": "required",
            }
        )
    )
    return 0
