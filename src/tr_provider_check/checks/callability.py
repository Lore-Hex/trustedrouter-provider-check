"""Tier 2: classify every advertised native model as OK, DEAD, or FLAKY."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Any

import httpx

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.contract import PONG_PROMPT, _classify, _response_error
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult, CheckStatus, check_result

Sleep = Callable[[float], Awaitable[None]]

#: Upper bound on models swept in one run. Every sweep entry is a real, billed
#: completion, and /models is not a list of chat models: OpenAI advertises 133
#: ids of which 44 are embeddings, speech, or image endpoints that can never
#: answer /chat/completions. Sweeping the lot bills the caller for a hundred
#: requests and reports non-chat ids as DEAD chat routes. The bound is reported
#: in every result, never applied silently -- a truncated sweep that looks
#: complete is worse than no sweep.
DEFAULT_MAX_SWEEP_MODELS = 25


async def run_callability_checks(
    client: GatewayClient,
    models: Sequence[str],
    *,
    delay_seconds: float = 0.1,
    sleep: Sleep = asyncio.sleep,
    max_models: int = DEFAULT_MAX_SWEEP_MODELS,
    declared_chat_models: Collection[str] = (),
) -> tuple[list[CheckResult], list[str]]:
    """Sweep models serially using ``llm/byok.go``'s chat route and fields.

    The DEAD/FLAKY decision itself is the byte-vendored production
    ``scripts/classify_provider_routes.py::_classify`` function. Serial delay
    mirrors that script and avoids turning discovery into a rate-limit burst.
    """

    if max_models < 0:
        raise ValueError("max_models must be non-negative")
    if not models:
        return [
            check_result(
                id="callability.advertised-models",
                tier=2,
                status="skip",
                assertion=assertion_for("callability.advertised-models"),
                measured={"reason": "native model discovery did not produce models"},
                contract_ref="enclave-go/internal/llm/byok.go; scripts/classify_provider_routes.py",
                marketplace_bullet="Every advertised model has a live chat/completions route.",
                remediation="Fix /v1/models discovery first, then rerun the callability sweep.",
            )
        ], []

    advertised = list(models)
    swept = advertised[:max_models] if max_models > 0 else advertised
    not_swept = advertised[len(swept) :]

    observations: list[dict[str, Any]] = []
    attribution_error: tuple[str | None, int | None, str | None] = (None, None, None)
    for index, model in enumerate(swept):
        status_code: int | None = None
        verdict: str
        error_kind: str | None = None
        try:
            response = await client.chat(
                model=model,
                prompt=PONG_PROMPT,
                temperature=None,
                max_tokens=4,
            )
            status_code = response.status_code
            verdict = _classify(status_code, response.text[:4096])
            if verdict != "ok":
                error_kind, error_status, error_message = _response_error(response)
                if verdict == "dead" and error_kind.startswith("http_"):
                    error_kind = "unsupported_route"
                if attribution_error == (None, None, None):
                    attribution_error = (error_kind, error_status, error_message)
        except httpx.HTTPError as error:
            verdict = _classify(None, str(error))
            error_kind = error.__class__.__name__
            if attribution_error == (None, None, None):
                attribution_error = (error_kind, None, None)
        observations.append(
            {
                "model": model,
                "verdict": verdict,
                "http_status": status_code,
                "error_type": error_kind,
            }
        )
        if index != len(swept) - 1 and delay_seconds > 0:
            await sleep(delay_seconds)

    dead_count = sum(row["verdict"] == "dead" for row in observations)
    flaky_count = sum(row["verdict"] == "flaky" for row in observations)
    declared_dead_count = sum(
        row["verdict"] == "dead" and row["model"] in declared_chat_models
        for row in observations
    )
    status: CheckStatus
    if declared_dead_count:
        # The provider declared these ids as chat-served, so a dead route is a
        # broken promise.
        status = "fail"
    elif dead_count or flaky_count:
        # Without a declaration the tool cannot know which advertised ids are
        # chat-served: OpenAI's /models lists embeddings, speech and image
        # endpoints that answer 404 on /chat/completions by design. Calling
        # those DEAD chat routes would accuse a correct provider, so an
        # undeclared sweep is advisory.
        status = "warn"
    else:
        status = "pass"
    error_type, error_status, error_message = attribution_error
    callable_models = [row["model"] for row in observations if row["verdict"] == "ok"]

    return [
        check_result(
            id="callability.advertised-models",
            tier=2,
            status=status,
            assertion=assertion_for("callability.advertised-models"),
            measured={
                "advertised_count": len(advertised),
                "swept_count": len(swept),
                "not_swept_count": len(not_swept),
                "not_swept": not_swept,
                "dead_count": dead_count,
                "declared_dead_count": declared_dead_count,
                "flaky_count": flaky_count,
                "models": observations,
            },
            contract_ref="enclave-go/internal/llm/byok.go; scripts/classify_provider_routes.py::_classify",
            marketplace_bullet="All advertised model ids resolve on POST /v1/chat/completions.",
            remediation=(
                "For DEAD rows, remove the id from /v1/models or provision that exact id for this API key and chat route. For FLAKY rows, keep the route advertised but fix capacity/network health and return Retry-After on overload before rerunning."
                + (
                    f" {len(not_swept)} advertised ids were NOT swept because the run is bounded at {max_models}; raise --max-sweep-models to cover them. Ids that serve embeddings, speech, or images cannot answer /chat/completions and should not be read as chat routes."
                    if not_swept
                    else ""
                )
            ),
            error_type=error_type,
            error_status=error_status,
            error_message=error_message,
        )
    ], callable_models
