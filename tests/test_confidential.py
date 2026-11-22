import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from tr_provider_check import confidential as cc


PIN = "acme/model@sha256:" + "1" * 64
ARGS = [
    "--host",
    "enclave.example.org",
    "--model",
    "acme/model",
    "--repo-pin",
    PIN,
    "--send-canary",
]
GOOD = {
    "model": "acme/model",
    "choices": [{"finish_reason": "stop", "message": {"content": "PONG"}}],
}


@pytest.mark.parametrize(
    "case",
    [
        "good",
        "wrong_model",
        "truncated",
        "redirect",
        "too_big",
        "not_json",
        "rejected_pin",
    ],
)
def test_attested_probe(monkeypatch, capsys, case):
    sent = []

    def respond(request):
        sent.append(request)
        assert request.url == "https://enclave.example.org/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-marker"
        assert json.loads(request.content)["max_tokens"] == 16
        if case == "redirect":
            return httpx.Response(
                307, headers={"Location": "https://other.example.org/"}
            )
        if case == "too_big":
            return httpx.Response(200, content=b"x" * 65537)
        if case == "not_json":
            return httpx.Response(200, text="private-upstream-message")
        if case == "truncated":
            return httpx.Response(200, json={"model": "acme/model", "choices": []})
        return httpx.Response(
            200,
            json={**GOOD, "model": "wrong" if case == "wrong_model" else "acme/model"},
        )

    @contextmanager
    def direct(host, repo, project):
        assert (host, repo, project) == ("enclave.example.org", PIN, None)
        if case == "rejected_pin":
            raise ValueError("private-upstream-message")
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            yield SimpleNamespace(http_client=client)

    monkeypatch.setattr(cc, "_client", direct)
    monkeypatch.setenv("TR_PROVIDER_API_KEY", "secret-marker")
    assert cc.main(ARGS) == (0 if case == "good" else 1)
    output = capsys.readouterr().out
    assert "private-upstream-message" not in output and "secret-marker" not in output
    assert len(sent) == (0 if case == "rejected_pin" else 1)
    report = json.loads(output)
    if case == "good":
        assert (
            report["result"] == "attested_chat_passed"
            and report["marketplace_review"] == "required"
        )
    else:
        assert report["stage"] == (
            "attestation" if case == "rejected_pin" else "attested_chat"
        )
        assert report["result"] == "failed"


@pytest.mark.parametrize(
    "args",
    [
        ARGS[:-1],
        [*ARGS[:5], "acme/model@latest", ARGS[-1]],
        [ARGS[0], "example.org/path", *ARGS[2:]],
    ],
)
def test_bad_arguments_never_dial(monkeypatch, args):
    monkeypatch.setattr(cc, "_client", lambda *a: pytest.fail("must not dial"))
    with pytest.raises(SystemExit) as exc:
        cc.main(args)
    assert exc.value.code == 2


def test_installed_tinfoil_adapter_constructs_real_immutable_pin(monkeypatch):
    sdk = pytest.importorskip("tinfoil")
    seen = []
    monkeypatch.setattr(
        sdk, "EnclaveHTTPClient", lambda host, pin, **kw: seen.append((host, pin, kw))
    )
    for repo, project in ((PIN, None), (None, PIN)):
        cc._client("model.example.org", repo, project)
        host, pin, opts = seen[-1]
        assert host == "model.example.org"
        assert (pin.repo, pin.project) == (repo, project)
        assert opts == {"transport_mode": "tls", "timeout": 30.0}
