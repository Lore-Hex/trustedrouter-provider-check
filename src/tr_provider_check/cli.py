"""Command-line entry point for the provider checker."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from tr_provider_check import __version__
from tr_provider_check.checks import run_checks
from tr_provider_check.checks.callability import DEFAULT_MAX_SWEEP_MODELS
from tr_provider_check.checks.perf import DEFAULT_PERF_SAMPLES
from tr_provider_check.report import report_document, report_json, report_table
from tr_provider_check.snapshot import contract_version

_PROG = "trustedrouter-provider-check"
API_KEY_ENV = "TR_PROVIDER_API_KEY"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Run TrustedRouter's provider-side catalog, callability, chat, "
            "streaming, tools, structured-output, and performance checks."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (contract {contract_version[:12]})",
    )
    parser.add_argument(
        "--print-contract-version",
        action="version",
        version=contract_version,
        help="print the full production contract hash and exit",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API root, including /v1 when the endpoint uses it",
    )
    parser.add_argument(
        "--api-key",
        help=(
            f"optional provider API key (prefer {API_KEY_ENV}); when omitted, "
            "no Authorization header is sent"
        ),
    )
    parser.add_argument(
        "--model",
        help="native model id for Tiers 3-6 (defaults to the first /models id)",
    )
    parser.add_argument(
        "--catalog-url",
        help="public Provider Contract v2 catalog URL; omitted means that check skips",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=(1, 2, 3, 4, 5, 6),
        default=4,
        help="highest tier to run, including every lower tier (default: 4)",
    )
    parser.add_argument(
        "--perf-samples",
        type=_positive_int,
        default=DEFAULT_PERF_SAMPLES,
        metavar="N",
        help=("number of billed Tier 6 throughput completions (default: %(default)s)"),
    )
    parser.add_argument(
        "--max-sweep-models",
        type=int,
        default=DEFAULT_MAX_SWEEP_MODELS,
        metavar="N",
        help=(
            "cap Tier 2 at N advertised models (default: %(default)s). Each swept "
            "model is a billed completion, and /models often advertises embedding, "
            "speech, and image ids that cannot answer chat. Skipped ids are always "
            "listed in the report; 0 sweeps everything."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="OUT",
        help="write a redacted JSON report to OUT, or use - for JSON on stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 2
    options = parser.parse_args(arguments)
    if not options.base_url:
        parser.error("--base-url is required when running checks")
    api_key = options.api_key or os.environ.get(API_KEY_ENV)

    run = asyncio.run(
        run_checks(
            base_url=options.base_url,
            api_key=api_key,
            model=options.model,
            catalog_url=options.catalog_url,
            tier=options.tier,
            max_sweep_models=options.max_sweep_models,
            perf_samples=options.perf_samples,
        )
    )
    secrets = (api_key,) if api_key else ()
    target = {
        "base_url": options.base_url,
        "model": run.selected_model,
        "provider": run.provider_id,
        "requested_tier": options.tier,
        "catalog_url": options.catalog_url,
        "authentication": {
            "mode": "bearer" if api_key else "none",
            "authorization_header_sent": bool(api_key),
            "note": (
                "Bearer credential supplied; report values are recursively redacted"
                if api_key
                else "none; no Authorization header was sent"
            ),
        },
        "max_sweep_models": options.max_sweep_models,
        "perf_samples": options.perf_samples,
    }
    if options.json == "-":
        print(
            report_json(
                run.checks,
                target=target,
                performance=run.performance,
                secrets=secrets,
            )
        )
    else:
        print(
            report_table(
                run.checks,
                target=target,
                performance=run.performance,
                secrets=secrets,
            )
        )
        if options.json:
            output_path = Path(options.json)
            output_path.write_text(
                report_json(
                    run.checks,
                    target=target,
                    performance=run.performance,
                    secrets=secrets,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"JSON report: {output_path}")
    document = report_document(
        run.checks,
        target=target,
        performance=run.performance,
        secrets=secrets,
    )
    summary = document["summary"]
    assert isinstance(summary, dict)
    return 0 if summary["conformance_gate"] is True else 1
