# Qualify one confidential model endpoint

Every new provider needs at least one model whose **actual inference path**
TrustedRouter can verify independently. This is not a brand, fleet-wide claim,
or replacement for the API conformance suite. Name the model revision, endpoint,
regions, replicas and release. A reseller must identify and cover the final
inference operator; attesting only its router is insufficient.

**Not set up yet?** [Contact provider operations](https://trustedrouter.com/support#support-inquiry).
We can help you choose a supported configuration, set up attestation and test
the endpoint together. Start with your intended model and hardware, not an API key.

## Evidence checklist (v1)

1. **Establish expectations offline.** Publish immutable source, signed release
   provenance, builder identity, hardware trust roots, allowed measurements and
   the exact model/weight commitment. Document reproducible verification commands.
   A measurement copied from the endpoint being checked is not a trust reference.
2. **Identify the workload.** A maintained verifier must validate current signed
   hardware evidence, endorsements, revocations, freshness (e.g. a verifier nonce)
   and production/non-debug mode. Compare the signed workload measurement with
   the independent reference. Inventory and verify all serving accelerators and
   relevant interconnects. Describe what can observe plaintext on each hop.
3. **Protect the model and keys.** Show how the verified release protects weights
   when loaded (for example a measured dm-verity root with no raw-device fallback).
   Show where keys are generated and who can access them. Publish retention terms
   for prompts, responses, caches and logs; attestation alone does not prove ZDR.
4. **Exercise the actual path.** Bind the peer TLS key on the inference connection
   or encrypt the request to an attested enclave key. Verify *before* sending any
   prompt. Check a bounded synthetic completion for the exact model and repeat
   for advertised streaming and eligible replicas. Do not send customer data.
5. **Demonstrate rejection.** In a controlled test deployment or harness, use a
   wrong release, a stale quote, a different TLS key and a missing/unqualified
   GPU or switch. Check the server's inference counter stays unchanged on each
   refusal and that there is no fallback to an ordinary endpoint. Corrupt a test
   weight block: readiness or inference must fail. Never damage a production volume.
6. **Submit repeatable evidence.** Include exact commands, UTC times, versions,
   release pins, redacted results for each check and limits or untested replicas.
   Keep credentials, raw device certificates, customer data and private logs out
   of GitHub and email. Arrange secure delivery of restricted material separately.
   TrustedRouter repeats the checks before approving the named route.

## Example: Tinfoil-compatible direct chat endpoint

[Tinfoil's client and verifier](https://github.com/tinfoilsh/tinfoil-python/tree/v0.16.0rc3)
provide an example of comparing signed releases to enclave measurements and
binding requests to the attested key. Alternatives are welcome: supply your
platform verifier and equivalent positive **and negative** evidence.

From a reviewed checkout of this checker, supply your **direct serving hostname**
(not a proxy/router), exact model and an **independently obtained** release
digest. Load a dedicated, restricted credential as `TR_PROVIDER_API_KEY` using
your secret manager. The following explicitly opts into a potentially billable
synthetic chat request (16 output tokens maximum). Do not run it on an
unapproved hostname or send it real prompts.

```sh
uv run --extra cc trustedrouter-cc-check \
  --host "$ENCLAVE_HOST" --model "$MODEL_ID" \
  --repo-pin "$OWNER_REPO@sha256:$RELEASE_DIGEST" --send-canary
```

Use `--project-pin "$ORG_PROJECT@sha256:$CONFIG_DIGEST"` instead for Tinfoil v3.
The SDK verifies the enclave before the probe uses its pinned TLS transport.
There is no ordinary HTTP fallback. Exit 0 with `attested_chat_passed` means
the synthetic chat passed on that connection; it does **not** prove weight
integrity, accelerator coverage, freshness beyond the SDK's policy, logging
policy, or fleet coverage. Those require the evidence above. Unknown platforms
cannot be qualified by pointing this Tinfoil-specific command at them.

For a repeatable negative control, run the same command against your staging
enclave with a deliberately different, well-formed sha256 release digest. Require
exit 1 at the attestation stage, **and** verify zero new inference calls on the
server. A DNS outage or a missing SDK is not evidence of correct rejection.

## Prompt for Claude

> Audit the model endpoint I name against this six-step checklist. Identify the
> immutable release, verifier version and trust roots before looking at the live
> measurement. Trace every hop where prompts could be read, including routers,
> GPU fabric and failover. Ask before making a billable call or sharing device
> evidence. Use only synthetic input, a restricted key in memory, and the
> platform's attested client; never print the key or run unreviewed provider code.
> In staging, run the positive and negative controls above. For each requirement
> give PASS, FAIL or NOT TESTED, the exact command with placeholders, UTC time,
> and redacted evidence. An unavailable test is NOT TESTED. Distinguish a
> successful connection from a qualified model/fleet. Do not change
> production, disable checks, or award a marketplace badge.
