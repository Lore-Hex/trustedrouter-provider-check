"""Tier 4 receipt checks bind exact fake-provider wire bytes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from tr_provider_check.checks import receipts as receipt_module
from tr_provider_check.checks.receipts import run_receipt_checks
from tr_provider_check.http import GatewayClient
from tr_provider_check.report import CheckResult

_CAPABILITY = {
    "spec": "inference-receipt/1",
    "algorithms": ["EdDSA"],
    "delivery": ["header", "stream-chunk"],
}
_NONCE = "provider-check-fresh-nonce_2026"
_ORIGIN = "https://provider.example"
_SEED = bytes(range(32))
_RECEIPT_IDS = {
    "receipt.header",
    "receipt.stream-position",
    "receipt.signature",
    "receipt.bindings",
    "receipt.issuer",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _b64url(value: bytes) -> str:
    return receipt_module._b64url_encode(value)


def _encode_point(point: receipt_module.Point) -> bytes:
    x, y, z, _ = point
    inverse = pow(z, receipt_module._FIELD_PRIME - 2, receipt_module._FIELD_PRIME)
    affine_x = x * inverse % receipt_module._FIELD_PRIME
    affine_y = y * inverse % receipt_module._FIELD_PRIME
    encoded = affine_y | ((affine_x & 1) << 255)
    return encoded.to_bytes(32, "little")


def _keypair(seed: bytes) -> tuple[int, bytes, bytes]:
    expanded = hashlib.sha512(seed).digest()
    scalar_bytes = bytearray(expanded[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    public_key = _encode_point(
        receipt_module._scalar_mult(receipt_module._BASE_POINT, scalar)
    )
    return scalar, expanded[32:], public_key


def _sign(message: bytes) -> tuple[bytes, bytes]:
    scalar, prefix, public_key = _keypair(_SEED)
    nonce = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little")
    nonce %= receipt_module._GROUP_ORDER
    encoded_nonce = _encode_point(
        receipt_module._scalar_mult(receipt_module._BASE_POINT, nonce)
    )
    challenge = (
        int.from_bytes(
            hashlib.sha512(encoded_nonce + public_key + message).digest(), "little"
        )
        % receipt_module._GROUP_ORDER
    )
    signature_scalar = (nonce + challenge * scalar) % receipt_module._GROUP_ORDER
    return public_key, encoded_nonce + signature_scalar.to_bytes(32, "little")


def _signed_jws(claims: dict[str, Any], *, flattened: bool) -> dict[str, str] | str:
    _, _, public_key = _keypair(_SEED)
    protected: dict[str, Any] = {
        "alg": "EdDSA",
        "typ": "inference-receipt+jws",
        "kid": _b64url(hashlib.sha256(public_key).digest()),
        "jwk": {"kty": "OKP", "crv": "Ed25519", "x": _b64url(public_key)},
    }
    if flattened:
        protected.update({"att": "test-attestation", "att_kind": "gcp-cs-jwt"})
    protected_segment = _b64url(_json_bytes(protected))
    payload_segment = _b64url(_json_bytes(claims))
    _, signature = _sign(f"{protected_segment}.{payload_segment}".encode())
    segments = {
        "protected": protected_segment,
        "payload": payload_segment,
        "signature": _b64url(signature),
    }
    if flattened:
        return segments
    return ".".join(segments[name] for name in ("protected", "payload", "signature"))


def _frame(payload: bytes) -> bytes:
    return b"data: " + payload + b"\n\n"


class _ReceiptProvider:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["x-inference-receipt"] == _NONCE
        body = json.loads(request.content)
        if body["stream"] is False:
            response_body = _json_bytes(
                {
                    "id": "chatcmpl-header",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "PONG"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            compact = _signed_jws({"rv": 1, "nonce": _NONCE}, flattened=False)
            headers = (
                {}
                if self.mode == "missing_header"
                else {"x-inference-receipt": str(compact)}
            )
            return httpx.Response(
                200,
                headers=headers,
                content=response_body,
                request=request,
            )

        first = _json_bytes(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": "PONG"}, "finish_reason": None}
                ],
            }
        )
        trailing = _json_bytes(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            }
        )
        hashed_events = [first]
        if self.mode == "receipt_not_last":
            hashed_events.append(trailing)
        request_hash = _b64url(hashlib.sha256(request.content).digest())
        if self.mode == "wrong_req_hash":
            request_hash = _b64url(b"\x00" * 32)
        response_hash = _b64url(
            hashlib.sha256(b"".join(event + b"\n" for event in hashed_events)).digest()
        )
        claims = {
            "rv": 1,
            "iss": "https://attacker.example"
            if self.mode == "wrong_issuer"
            else _ORIGIN,
            "iat": 1_700_000_000,
            "jti": "chatcmpl-stream",
            "nonce": _NONCE,
            "route": "chat.completions",
            "req": {"alg": "sha256", "hash": request_hash, "of": "body"},
            "resp": {
                "alg": "sha256",
                "hash": response_hash,
                "of": "sse-data-v1",
                "events": len(hashed_events),
            },
            "model": {
                "requested": "mock/model",
                "selected": "mock/model",
                "provider": "mock",
                "endpoint": "chat/completions",
            },
            "upstream": {"tier": "tls-webpki"},
        }
        flattened = _signed_jws(claims, flattened=True)
        assert isinstance(flattened, dict)
        if self.mode == "bad_signature":
            signature = bytearray(
                receipt_module._b64url_decode(flattened["signature"], label="test")
            )
            signature[0] ^= 1
            flattened["signature"] = _b64url(bytes(signature))
        receipt_payload = _json_bytes(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [],
                "inference_receipt": flattened,
            }
        )
        payloads = [first, receipt_payload]
        if self.mode == "receipt_not_last":
            payloads.append(trailing)
        payloads.append(b"[DONE]")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(_frame(payload) for payload in payloads),
            request=request,
        )


async def _run(
    mode: str, capability: dict[str, Any] | None = _CAPABILITY
) -> tuple[dict[str, CheckResult], _ReceiptProvider]:
    provider = _ReceiptProvider(mode)
    async with GatewayClient(
        f"{_ORIGIN}/v1",
        "test-key",
        transport=httpx.MockTransport(provider),
    ) as client:
        results = await run_receipt_checks(
            client,
            "mock/model",
            capability=capability,
            expected_origin=_ORIGIN,
            nonce_factory=lambda: _NONCE,
        )
    rows = {result.id: result for result in results}
    assert set(rows) == _RECEIPT_IDS
    assert len(results) == len(rows)
    return rows, provider


@pytest.mark.asyncio
async def test_receipts_capability_absent_skips_without_requests() -> None:
    rows, provider = await _run("conforming", capability=None)

    assert {row.status for row in rows.values()} == {"skip"}
    assert provider.requests == []


@pytest.mark.asyncio
async def test_conforming_header_and_stream_receipts_pass() -> None:
    rows, provider = await _run("conforming")

    assert {row.status for row in rows.values()} == {"pass"}
    assert len(provider.requests) == 2
    assert rows["receipt.bindings"].measured["request_hash_matches"] is True
    assert rows["receipt.bindings"].measured["response_event_count_matches"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "failed_id"),
    [
        ("missing_header", "receipt.header"),
        ("receipt_not_last", "receipt.stream-position"),
        ("bad_signature", "receipt.signature"),
        ("wrong_req_hash", "receipt.bindings"),
        ("wrong_issuer", "receipt.issuer"),
    ],
)
async def test_one_tampered_receipt_dimension_fails_one_check(
    mode: str, failed_id: str
) -> None:
    rows, _ = await _run(mode)

    assert rows[failed_id].status == "fail"
    assert {
        check_id: row.status for check_id, row in rows.items() if check_id != failed_id
    } == {check_id: "pass" for check_id in _RECEIPT_IDS - {failed_id}}


def test_strict_ed25519_verifier_matches_rfc8032_vector() -> None:
    public_key = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    signature = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )

    assert receipt_module._verify_ed25519(public_key, b"", signature) is True
    assert receipt_module._verify_ed25519(public_key, b"tampered", signature) is False
