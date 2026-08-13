"""Gateway-equivalent OpenAI-compatible HTTP client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx

from tr_provider_check.report import CheckStatus

from urllib.parse import urlsplit


TOTAL_TIMEOUT_SECONDS = 10 * 60.0
CONNECT_TIMEOUT_SECONDS = 10.0
KEEPALIVE_EXPIRY_SECONDS = 90.0
MAX_IDLE_CONNECTIONS = 1024
MAX_IDLE_CONNECTIONS_PER_HOST = 128
MAX_TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

MaxTokenParameter = Literal["max_tokens", "max_completion_tokens"]
Sleep = Callable[[float], Awaitable[None]]


def provider_for_model(model: str) -> str:
    """Infer the marketplace owner only when a canonical id supplies one."""

    owner, separator, _ = model.partition("/")
    return owner if separator else ""


#: Host -> TrustedRouter provider name. The enclave chooses provider-specific
#: request shapes (notably the max_tokens vs max_completion_tokens rename in
#: byok.go's requiresMaxCompletionTokens), and it learns the provider from
#: routing context this tool does not have. Inferring it from the endpoint host
#: keeps those choices correct for a bare native model id: passing
#: "gpt-5.4-nano" with no "openai/" prefix previously left the provider empty,
#: so the tool sent max_tokens, OpenAI answered 400, and the report blamed the
#: provider for rejecting a spelling TrustedRouter would never have sent it.
_PROVIDER_BY_HOST_SUFFIX: tuple[tuple[str, str], ...] = (
    ("api.openai.com", "openai"),
    ("api.deepseek.com", "deepseek"),
    ("api.z.ai", "zai"),
    ("api.cerebras.ai", "cerebras"),
    ("api.mistral.ai", "mistral"),
    ("api.together.xyz", "together"),
    ("api.novita.ai", "novita"),
    ("api.fireworks.ai", "fireworks"),
    ("api.minimax.io", "minimax"),
)


def provider_for_base_url(base_url: str) -> str:
    """Infer the TrustedRouter provider name from an endpoint host."""

    host = urlsplit(base_url).hostname or ""
    host = host.casefold()
    for suffix, provider in _PROVIDER_BY_HOST_SUFFIX:
        if host == suffix or host.endswith("." + suffix):
            return provider
    return ""


def max_token_parameter(provider: str, model: str) -> MaxTokenParameter:
    """Match byok.go's per-provider spelling for an explicitly supplied cap."""

    normalized = model.casefold().split("/", 1)[-1]
    if provider.casefold() == "openai" and normalized.startswith(
        ("gpt-5", "o1", "o3", "o4")
    ):
        return "max_completion_tokens"
    return "max_tokens"


def gateway_omits_temperature(provider: str, model: str) -> bool:
    """Mirror byok.go's openAICompatibleTemperature omission rule.

    The vendored ``_rotation_omits_temperature`` is the SYNTHETIC PROBE's rule
    and is narrower than what production actually sends. The gateway drops
    temperature whenever ``kimiUsesFixedSampling`` or
    ``requiresMaxCompletionTokens`` holds, and the first of those is
    provider-independent: a k2.5 served by Novita or Together also gets it
    omitted, because those hosted routes reject temperature=0 with
    invalid_request_error too. Asserting the probe's narrower rule would send
    temperature to a route production never sends it to, and fail a provider
    for a request TrustedRouter would never make.
    """

    model_l = model.casefold()
    provider_l = provider.casefold()
    kimi_fixed_sampling = (
        "kimi-k2.5" in model_l
        or "kimi-k2.6" in model_l
        or "kimi-k3" in model_l
        or (provider_l == "kimi" and "kimi-k2." in model_l)
    )
    return (
        kimi_fixed_sampling
        or max_token_parameter(provider, model) == "max_completion_tokens"
    )


def probe_inconclusive(status_code: int | None) -> bool:
    """Whether a capability probe learned nothing because the backend was down.

    A capability probe infers support from the response to a request carrying
    the field. That inference only holds for a 4xx: the server read the request
    and refused it. A 5xx or 429 means it never got that far, so reporting
    "this provider rejects X" states a cause the evidence does not support.
    Observed against a live endpoint whose backend returns intermittent 502s:
    a 502 on the temperature probe was reported as a hard temperature-rejection
    failure, and the same request succeeded minutes later.
    """

    if status_code is None:
        return True
    return status_code == 429 or 500 <= status_code <= 599


def probe_verdict(
    status_code: int | None, *, declared: bool
) -> tuple[CheckStatus, str | None]:
    """Apply the shared capability/route probe evidence ladder.

    HTTP 200 proves the request was accepted. A missing response, 429, or 5xx
    proves neither support nor rejection and is therefore advisory. A permanent
    rejection fails only when the capability or route was declared; otherwise
    the probe did not discover the capability and skips.
    """

    if status_code == 200:
        return "pass", None
    if status_code is None:
        return "warn", "probe did not receive an HTTP response; support is unknown"
    if probe_inconclusive(status_code):
        return "warn", "backend unavailable during the probe; support is unknown"
    if declared:
        return "fail", "the declared capability or route rejected the probe"
    return "skip", "the undeclared capability was not discovered by the probe"


def _is_transient_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _is_transient_transport_error(error: httpx.HTTPError) -> bool:
    return isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.WriteError,
        ),
    )


#: Per-request Accept headers. The gateway asks for SSE only when it is
#: actually streaming; JSON endpoints get the JSON type.
SSE_ACCEPT = "text/event-stream"
JSON_ACCEPT = "application/json"


class GatewayClient:
    """Reusable client matching enclave keep-alive, headers, and request fields.

    Contract references:
    ``enclave-go/internal/llm/byok.go`` and
    ``enclave-go/internal/llm/http_client.go``. Same-provider retry behavior is
    in the available checkout's ``enclave-go/cmd/enclave/provider_stream.go``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        provider: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.provider = provider or provider_for_base_url(self.base_url)
        self._sleep = sleep
        # Accept is deliberately NOT a client-wide default. The gateway sends
        # text/event-stream on a streaming completion, but sending it on the
        # JSON endpoints makes a correct server refuse the request: z.ai
        # answers GET /models with 406 Not Acceptable, which this tool then
        # reported as the PROVIDER's broken catalog. Accusing a conformant
        # provider of a fault we caused is the worst failure this tool has.
        # Go's default Transport automatically advertises gzip and transparently
        # decodes it. Pinning the header keeps httpx from additionally offering
        # deflate or Brotli, which the enclave does not negotiate.
        request_headers = {
            "User-Agent": "TrustedRouter/1.0",
            "Accept-Encoding": "gzip",
        }
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            request_headers.update(headers)
        self._client = httpx.AsyncClient(
            headers=request_headers,
            timeout=httpx.Timeout(
                TOTAL_TIMEOUT_SECONDS,
                connect=CONNECT_TIMEOUT_SECONDS,
                read=TOTAL_TIMEOUT_SECONDS,
                write=TOTAL_TIMEOUT_SECONDS,
                pool=CONNECT_TIMEOUT_SECONDS,
            ),
            limits=httpx.Limits(
                max_connections=MAX_IDLE_CONNECTIONS,
                max_keepalive_connections=MAX_IDLE_CONNECTIONS_PER_HOST,
                keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
            ),
            http2=transport is None,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> GatewayClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def models(self) -> httpx.Response:
        """Fetch native endpoint model ids without applying marketplace regexes."""

        return await self._client.get(
            f"{self.base_url}/models", headers={"Accept": JSON_ACCEPT}
        )

    def chat_body(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool = True,
        temperature: float | None = 0,
        max_tokens: int | None = None,
        max_tokens_key: MaxTokenParameter | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the exact common request recipe from ``llm/byok.go``.

        ``max_tokens`` is absent unless the caller explicitly provides a cap.
        The vendored production temperature gate is used unchanged.
        """

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        provider = self.provider or provider_for_model(model)
        if temperature is not None and not gateway_omits_temperature(provider, model):
            body["temperature"] = temperature
        if max_tokens is not None:
            key = max_tokens_key or max_token_parameter(provider, model)
            body[key] = max_tokens
        if extra:
            body.update(extra)
        return body

    async def chat(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float | None = 0,
        max_tokens: int | None = None,
        max_tokens_key: MaxTokenParameter | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue a non-streaming diagnostic using the gateway's common fields."""

        body = self.chat_body(
            model=model,
            prompt=prompt,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_key=max_tokens_key,
            extra=extra,
        )
        return await self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Accept": JSON_ACCEPT},
        )

    @asynccontextmanager
    async def stream_chat(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float | None = 0,
        max_tokens: int | None = None,
        max_tokens_key: MaxTokenParameter | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Open a stream, retrying transient failures before body bytes escape.

        A yielded response is never retried: once its body is exposed, retrying
        could duplicate output or billing. This matches the byte gate in the
        available enclave ``cmd/enclave/provider_stream.go``.
        """

        body = self.chat_body(
            model=model,
            prompt=prompt,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_key=max_tokens_key,
            extra=extra,
        )
        response: httpx.Response | None = None
        for attempt in range(MAX_TRANSIENT_RETRIES + 1):
            request = self._client.build_request(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Accept": SSE_ACCEPT},
            )
            try:
                response = await self._client.send(request, stream=True)
            except httpx.HTTPError as error:
                if (
                    attempt >= MAX_TRANSIENT_RETRIES
                    or not _is_transient_transport_error(error)
                ):
                    raise
                await self._sleep(TRANSIENT_BACKOFF_SECONDS[attempt])
                continue
            if (
                _is_transient_status(response.status_code)
                and attempt < MAX_TRANSIENT_RETRIES
            ):
                await response.aread()
                await response.aclose()
                await self._sleep(TRANSIENT_BACKOFF_SECONDS[attempt])
                response = None
                continue
            break
        if response is None:  # pragma: no cover - loop invariants make this unreachable
            raise RuntimeError("stream request ended without a response")
        try:
            yield response
        finally:
            await response.aclose()
