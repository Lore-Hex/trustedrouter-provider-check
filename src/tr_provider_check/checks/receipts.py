"""Tier 4: signed inference receipt wire and binding checks."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from tr_provider_check.checks.assertions import assertion_for
from tr_provider_check.http import GatewayClient, probe_inconclusive
from tr_provider_check.report import CheckResult, CheckStatus, check_result

_RECEIPT_HEADER = "x-inference-receipt"
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{1,88}$")
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CHECK_IDS = (
    "receipt.header",
    "receipt.stream-position",
    "receipt.signature",
    "receipt.bindings",
    "receipt.issuer",
)

# RFC 8032's Edwards25519 parameters. This strict verifier accepts canonical
# encodings and prime-order points only; there is no algorithm dispatch or
# dependency-controlled fallback that could turn EdDSA into another scheme.
_FIELD_PRIME = 2**255 - 19
_GROUP_ORDER = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _FIELD_PRIME - 2, _FIELD_PRIME)) % _FIELD_PRIME
_SQRT_M1 = pow(2, (_FIELD_PRIME - 1) // 4, _FIELD_PRIME)

Point = tuple[int, int, int, int]
_IDENTITY: Point = (0, 1, 1, 0)


def _recover_x(y: int, sign: int) -> int:
    numerator = (y * y - 1) % _FIELD_PRIME
    denominator = (_D * y * y + 1) % _FIELD_PRIME
    x_squared = numerator * pow(denominator, _FIELD_PRIME - 2, _FIELD_PRIME)
    x_squared %= _FIELD_PRIME
    x = pow(x_squared, (_FIELD_PRIME + 3) // 8, _FIELD_PRIME)
    if (x * x - x_squared) % _FIELD_PRIME:
        x = x * _SQRT_M1 % _FIELD_PRIME
    if (x * x - x_squared) % _FIELD_PRIME:
        raise ValueError("Ed25519 point is not on the curve")
    if x == 0 and sign:
        raise ValueError("Ed25519 point has a non-canonical sign bit")
    return _FIELD_PRIME - x if (x & 1) != sign else x


def _decode_point(encoded: bytes) -> Point:
    if len(encoded) != 32:
        raise ValueError("Ed25519 point must be 32 bytes")
    value = int.from_bytes(encoded, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    if y >= _FIELD_PRIME:
        raise ValueError("Ed25519 point encoding is not canonical")
    x = _recover_x(y, sign)
    return (x, y, 1, x * y % _FIELD_PRIME)


def _point_add(left: Point, right: Point) -> Point:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _FIELD_PRIME
    b = (y1 + x1) * (y2 + x2) % _FIELD_PRIME
    c = 2 * _D * t1 * t2 % _FIELD_PRIME
    d = 2 * z1 * z2 % _FIELD_PRIME
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return (
        e * f % _FIELD_PRIME,
        g * h % _FIELD_PRIME,
        f * g % _FIELD_PRIME,
        e * h % _FIELD_PRIME,
    )


def _scalar_mult(point: Point, scalar: int) -> Point:
    result = _IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


def _points_equal(left: Point, right: Point) -> bool:
    return (left[0] * right[2] - right[0] * left[2]) % _FIELD_PRIME == 0 and (
        left[1] * right[2] - right[1] * left[2]
    ) % _FIELD_PRIME == 0


_BASE_Y = 4 * pow(5, _FIELD_PRIME - 2, _FIELD_PRIME) % _FIELD_PRIME
_BASE_X = _recover_x(_BASE_Y, 0)
_BASE_POINT: Point = (_BASE_X, _BASE_Y, 1, _BASE_X * _BASE_Y % _FIELD_PRIME)


def _verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify one strict, canonical Ed25519 signature."""

    if len(public_key) != 32 or len(signature) != 64:
        return False
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _GROUP_ORDER:
        return False
    try:
        public_point = _decode_point(public_key)
        nonce_point = _decode_point(signature[:32])
    except ValueError:
        return False
    if _points_equal(public_point, _IDENTITY):
        return False
    if not _points_equal(_scalar_mult(public_point, _GROUP_ORDER), _IDENTITY):
        return False
    if not _points_equal(_scalar_mult(nonce_point, _GROUP_ORDER), _IDENTITY):
        return False
    challenge = (
        int.from_bytes(
            hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
        )
        % _GROUP_ORDER
    )
    expected = _point_add(nonce_point, _scalar_mult(public_point, challenge))
    return _points_equal(_scalar_mult(_BASE_POINT, scalar), expected)


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: object, *, label: str) -> bytes:
    if not isinstance(value, str) or _B64URL_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not unpadded base64url")
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{label} is not base64url") from error
    if _b64url_encode(decoded) != value:
        raise ValueError(f"{label} is not canonical base64url")
    return decoded


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _json_object(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


@dataclass(frozen=True)
class _JWS:
    protected_segment: str
    payload_segment: str
    signature_segment: str
    protected: dict[str, Any]
    claims: dict[str, Any]
    signature: bytes


def _parse_jws_segments(protected: object, payload: object, signature: object) -> _JWS:
    protected_bytes = _b64url_decode(protected, label="JWS protected header")
    payload_bytes = _b64url_decode(payload, label="JWS payload")
    signature_bytes = _b64url_decode(signature, label="JWS signature")
    if len(signature_bytes) != 64:
        raise ValueError("Ed25519 signature must be 64 bytes")
    protected_object = _json_object(protected_bytes, label="JWS protected header")
    claims = _json_object(payload_bytes, label="JWS payload")
    if protected_object.get("typ") != "inference-receipt+jws":
        raise ValueError("receipt JWS typ is invalid")
    if protected_object.get("alg") != "EdDSA":
        raise ValueError("receipt JWS alg is invalid")
    assert isinstance(protected, str)
    assert isinstance(payload, str)
    assert isinstance(signature, str)
    return _JWS(
        protected,
        payload,
        signature,
        protected_object,
        claims,
        signature_bytes,
    )


def _parse_compact(value: str) -> _JWS:
    segments = value.split(".")
    if len(segments) != 3:
        raise ValueError("compact JWS must have exactly three segments")
    return _parse_jws_segments(*segments)


def _parse_flattened(value: object) -> _JWS:
    if not isinstance(value, dict):
        raise ValueError("inference_receipt must be an object")
    if set(value) != {"protected", "payload", "signature"}:
        raise ValueError("flattened JWS fields are invalid")
    return _parse_jws_segments(
        value.get("protected"), value.get("payload"), value.get("signature")
    )


@dataclass(frozen=True)
class _DataEvent:
    payload: bytes
    receipt: object | None
    receipt_chunk_valid: bool
    done: bool


def _sse_data_events(body: bytes) -> list[_DataEvent]:
    """Parse SSE data fields while preserving their effective payload bytes."""

    normalized = body.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    events: list[_DataEvent] = []
    data_lines: list[bytes] = []
    for line in normalized.split(b"\n"):
        if line:
            if line.startswith(b":"):
                continue
            field, separator, value = line.partition(b":")
            if field == b"data":
                if not separator:
                    value = b""
                elif value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
            continue
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        data_lines = []
        receipt: object | None = None
        receipt_chunk_valid = False
        if payload != b"[DONE]":
            try:
                parsed = _json_object(payload, label="SSE data payload")
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and "inference_receipt" in parsed:
                receipt = parsed["inference_receipt"]
                receipt_chunk_valid = (
                    parsed.get("object") == "chat.completion.chunk"
                    and parsed.get("choices") == []
                )
        events.append(
            _DataEvent(payload, receipt, receipt_chunk_valid, payload == b"[DONE]")
        )
    return events


def _canonical_https_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("expected receipt issuer must be a canonical HTTPS origin")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("expected receipt issuer has an invalid port") from error
    host = parsed.hostname.casefold()
    if ":" in host:
        host = f"[{host}]"
    authority = host if port in {None, 443} else f"{host}:{port}"
    return f"https://{authority}"


def _receipt_skip(check_id: str, reason: str) -> CheckResult:
    return check_result(
        id=check_id,
        tier=4,
        status="skip",
        assertion=assertion_for(check_id),
        measured={"reason": reason},
        contract_ref="signed-receipts-wire-format.md §§2-4, 6, 8.1, 10",
        marketplace_bullet="Declared inference receipts are bound and independently verifiable.",
        remediation="Declare only the receipt delivery modes this endpoint implements.",
    )


def _receipt_result(
    check_id: str,
    status: CheckStatus,
    measured: dict[str, Any],
    remediation: str,
) -> CheckResult:
    return check_result(
        id=check_id,
        tier=4,
        status=status,
        assertion=assertion_for(check_id),
        measured=measured,
        contract_ref="signed-receipts-wire-format.md §§2-4, 6, 8.1, 10",
        marketplace_bullet="Declared inference receipts are bound and independently verifiable.",
        remediation=remediation,
        error_type="stream_error" if status == "fail" else None,
        error_message="declared receipt evidence was invalid"
        if status == "fail"
        else None,
    )


def _stream_failure_results(status: CheckStatus, reason: str) -> list[CheckResult]:
    remediation = "Return one valid receipt chunk as the last data event before [DONE]."
    return [
        _receipt_result(
            check_id,
            status,
            {"reason": reason, "inconclusive": status == "warn"},
            remediation,
        )
        for check_id in _CHECK_IDS[1:]
    ]


def _fresh_nonce() -> str:
    return secrets.token_urlsafe(32)


async def run_receipt_checks(
    client: GatewayClient,
    model: str,
    *,
    capability: Mapping[str, Any] | None,
    expected_origin: str | None = None,
    nonce_factory: Callable[[], str] = _fresh_nonce,
) -> list[CheckResult]:
    """Check declared third-party receipt behavior without attestation chains.

    Request and response bytes are mandatory inputs to the binding result, and
    the issuer is pinned to configured endpoint state. The receipt's own
    attacker-writable ``iss`` is never used for discovery or network access.
    """

    if capability is None:
        reason = "the selected model does not declare the receipts capability"
        return [_receipt_skip(check_id, reason) for check_id in _CHECK_IDS]

    raw_delivery = capability.get("delivery")
    delivery = set(raw_delivery) if isinstance(raw_delivery, list) else set()
    nonce = nonce_factory()
    if _NONCE_RE.fullmatch(nonce) is None or nonce == "true":
        raise ValueError("nonce_factory must return a valid non-reserved receipt nonce")
    receipt_headers = {_RECEIPT_HEADER: nonce}
    results: list[CheckResult] = []

    if "header" not in delivery:
        results.append(
            _receipt_skip("receipt.header", "header delivery is not declared")
        )
    else:
        try:
            response = await client.chat(
                model=model,
                prompt="Reply exactly PONG.",
                temperature=0,
                headers=receipt_headers,
            )
        except httpx.HTTPError as error:
            results.append(
                _receipt_result(
                    "receipt.header",
                    "warn",
                    {
                        "error": error.__class__.__name__,
                        "inconclusive": True,
                    },
                    "Restore the endpoint and rerun the declared header receipt check.",
                )
            )
        else:
            compact = response.headers.get(_RECEIPT_HEADER)
            parse_error: str | None = None
            try:
                if compact is None:
                    raise ValueError("response receipt header is missing")
                _parse_compact(compact)
            except ValueError as error:
                parse_error = str(error)
            ok = response.status_code == 200 and parse_error is None
            results.append(
                _receipt_result(
                    "receipt.header",
                    "pass"
                    if ok
                    else "warn"
                    if probe_inconclusive(response.status_code)
                    else "fail",
                    {
                        "http_status": response.status_code,
                        "header_present": compact is not None,
                        "compact_jws_valid": parse_error is None,
                        "error": parse_error,
                    },
                    "Return a compact EdDSA JWS in x-inference-receipt with the required typ.",
                )
            )

    if "stream-chunk" not in delivery:
        reason = "stream-chunk delivery is not declared"
        results.extend(_receipt_skip(check_id, reason) for check_id in _CHECK_IDS[1:])
        return results

    try:
        stream_context = client.stream_chat(
            model=model,
            prompt="Reply exactly PONG.",
            temperature=0,
            headers=receipt_headers,
        )
        async with stream_context as response:
            body = await response.aread()
            request_body = response.request.content
            status_code = response.status_code
    except httpx.HTTPError as error:
        results.extend(
            _stream_failure_results(
                "warn", f"stream request failed with {error.__class__.__name__}"
            )
        )
        return results

    if status_code != 200:
        status: CheckStatus = "warn" if probe_inconclusive(status_code) else "fail"
        results.extend(_stream_failure_results(status, f"HTTP {status_code}"))
        return results

    events = _sse_data_events(body)
    receipt_indexes = [
        index for index, event in enumerate(events) if event.receipt is not None
    ]
    done_indexes = [index for index, event in enumerate(events) if event.done]
    position_ok = (
        len(receipt_indexes) == 1
        and len(done_indexes) == 1
        and events[receipt_indexes[0]].receipt_chunk_valid
        and receipt_indexes[0] + 1 == done_indexes[0]
        and done_indexes[0] == len(events) - 1
    )
    results.append(
        _receipt_result(
            "receipt.stream-position",
            "pass" if position_ok else "fail",
            {
                "data_event_count": len(events),
                "receipt_event_count": len(receipt_indexes),
                "done_event_count": len(done_indexes),
                "receipt_is_last_before_done": position_ok,
            },
            "Emit exactly one inference_receipt chunk immediately before the sole final [DONE].",
        )
    )

    flattened_value = events[receipt_indexes[0]].receipt if receipt_indexes else None
    jws: _JWS | None = None
    jws_error: str | None = None
    try:
        jws = _parse_flattened(flattened_value)
    except ValueError as error:
        jws_error = str(error)

    signature_ok = False
    kid_matches = False
    key_valid = False
    if jws is not None:
        try:
            jwk = jws.protected.get("jwk")
            if not isinstance(jwk, dict):
                raise ValueError("protected jwk is missing")
            if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
                raise ValueError("protected jwk is not Ed25519 OKP")
            public_key = _b64url_decode(jwk.get("x"), label="jwk.x")
            if len(public_key) != 32:
                raise ValueError("jwk.x must encode 32 bytes")
            key_valid = True
            expected_kid = _b64url_encode(hashlib.sha256(public_key).digest())
            kid_matches = hmac.compare_digest(
                str(jws.protected.get("kid") or ""), expected_kid
            )
            signing_input = f"{jws.protected_segment}.{jws.payload_segment}".encode(
                "ascii"
            )
            signature_ok = kid_matches and _verify_ed25519(
                public_key, signing_input, jws.signature
            )
        except (UnicodeEncodeError, ValueError) as error:
            jws_error = str(error)
    results.append(
        _receipt_result(
            "receipt.signature",
            "pass" if signature_ok else "fail",
            {
                "flattened_jws_valid": jws is not None,
                "jwk_valid": key_valid,
                "kid_matches_key_hash": kid_matches,
                "signature_valid": signature_ok,
                "error": jws_error,
            },
            "Sign the flattened protected.payload input with its Ed25519 jwk and matching kid.",
        )
    )

    non_receipt_payloads = [
        event.payload for event in events if event.receipt is None and not event.done
    ]
    response_preimage = b"".join(payload + b"\n" for payload in non_receipt_payloads)
    payloads_lf_free = all(b"\n" not in payload for payload in non_receipt_payloads)
    request_hash = _b64url_encode(hashlib.sha256(request_body).digest())
    response_hash = _b64url_encode(hashlib.sha256(response_preimage).digest())
    claims = jws.claims if jws is not None else {}
    req = claims.get("req")
    resp = claims.get("resp")
    bindings_ok = (
        isinstance(req, dict)
        and req.get("alg") == "sha256"
        and req.get("of") == "body"
        and isinstance(req.get("hash"), str)
        and hmac.compare_digest(req["hash"], request_hash)
        and isinstance(resp, dict)
        and resp.get("alg") == "sha256"
        and resp.get("of") == "sse-data-v1"
        and isinstance(resp.get("hash"), str)
        and hmac.compare_digest(resp["hash"], response_hash)
        and type(resp.get("events")) is int
        and resp["events"] == len(non_receipt_payloads)
        and claims.get("nonce") == nonce
        and payloads_lf_free
    )
    results.append(
        _receipt_result(
            "receipt.bindings",
            "pass" if bindings_ok else "fail",
            {
                "request_body_captured": True,
                "response_events_captured": len(non_receipt_payloads),
                "request_hash_matches": isinstance(req, dict)
                and req.get("hash") == request_hash,
                "response_hash_matches": isinstance(resp, dict)
                and resp.get("hash") == response_hash,
                "response_event_count_matches": isinstance(resp, dict)
                and resp.get("events") == len(non_receipt_payloads),
                "nonce_matches": claims.get("nonce") == nonce,
                "response_payloads_lf_free": payloads_lf_free,
            },
            "Hash the exact received request body and every non-receipt SSE data payload, including the event count, and echo the nonce.",
        )
    )

    issuer_error: str | None = None
    try:
        pinned_origin = _canonical_https_origin(expected_origin or client.base_url)
    except ValueError as error:
        pinned_origin = None
        issuer_error = str(error)
    issuer = claims.get("iss")
    issuer_ok = (
        pinned_origin is not None
        and isinstance(issuer, str)
        and hmac.compare_digest(issuer, pinned_origin)
    )
    results.append(
        _receipt_result(
            "receipt.issuer",
            "pass" if issuer_ok else "fail",
            {
                "expected_origin": pinned_origin,
                "issuer": issuer,
                "exact_match": issuer_ok,
                "error": issuer_error,
            },
            "Set iss to the configured API endpoint's exact canonical HTTPS origin.",
        )
    )
    return results


__all__ = ["run_receipt_checks"]
