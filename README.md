# trustedrouter-provider-check

`trustedrouter-provider-check` runs provider-side compatibility checks for the
OpenAI-compatible HTTP contract consumed by TrustedRouter. It checks the
catalog, a bounded billed sweep of advertised native models, non-streaming chat
semantics, and the streaming behavior that most often fails behind gateways and
reverse proxies.
Opt-in Tiers 5–6 add tools, structured output, and advisory production-style
performance sampling.

The implementation vendors production contract functions and a content-hashed
snapshot from TrustedRouter. The local test suite joins those functions to a
configurable mock HTTP server. The checks have also been exercised against nine
live OpenAI-compatible endpoints during development; that is point-in-time test
evidence, not certification. Tier 6 performance remains advisory.

## Running the checker

Install the project with [uv](https://docs.astral.sh/uv/), then pass the API root
that owns `/models` and `/chat/completions`. Include `/v1` when the endpoint uses
that prefix. Authentication is optional: when neither `--api-key` nor
`TR_PROVIDER_API_KEY` is set, the checker sends no `Authorization` header and
records that fact in both report formats.

```console
export TR_PROVIDER_API_KEY='your-provider-key'
uv run trustedrouter-provider-check \
  --base-url https://inference.example.com/v1 \
  --model deepseek-chat
```

`--api-key` is also accepted, but `TR_PROVIDER_API_KEY` keeps the key out of
shell history. When supplied, the key is recursively redacted from human and
JSON reports.

Available options:

```text
--base-url URL       OpenAI-compatible API root; required for a check run
--api-key KEY        Optional provider key; falls back to TR_PROVIDER_API_KEY
--model MODEL        Native model id for Tiers 3-6; defaults to the first tier-2 callable id
--catalog-url URL    Public Provider Contract v2 declaration; omitted means skip
--tier {1,2,3,4,5,6} Highest tier to run, including lower tiers; default 4
--max-sweep-models N Cap Tier 2 at N billed completions; default 25; 0 sweeps all
--perf-samples N     Billed Tier 6 completions; default 3
--json OUT           Also write redacted JSON to OUT; use - for JSON on stdout
```

Human-readable output is the default. The conformance gate and process exit
status use Tiers 1–4 only: a Tier 1–4 hard failure exits `1`; otherwise the run
exits `0`, even when an informational Tier 5 finding or advisory Tier 6 sample
is red. Invalid or missing arguments exit `2`. Running with no arguments prints
help and exits `2`; it never reports an empty success.

Examples:

```console
# Catalog discovery and an optional public declaration only
uv run trustedrouter-provider-check \
  --base-url https://inference.example.com/v1 \
  --tier 1

# Keyless local Ollama/llama.cpp/vLLM setup: no Authorization header is sent
uv run trustedrouter-provider-check \
  --base-url http://127.0.0.1:11434/v1 \
  --model llama3.2 \
  --tier 4

# All tiers, plus a JSON artifact
uv run trustedrouter-provider-check \
  --base-url https://inference.example.com/v1 \
  --model qwen3-4b:latest \
  --catalog-url https://inference.example.com/catalog.v2.json \
  --tier 6 \
  --perf-samples 3 \
  --json provider-report.json

# Package and exact contract identities
uv run trustedrouter-provider-check --version
uv run trustedrouter-provider-check --print-contract-version
```

## What the tiers check

| Tier | Checks | Result discipline |
| --- | --- | --- |
| 1 | Non-empty unique native IDs from `/models`; optional declared marketplace catalog against Catalog v2 and vendored exact-field/id/decimal rules | Invalid discovery or declaration is `fail`; an omitted `--catalog-url` is `skip`. The `owner/model` regex applies only to the declaration, never native engine IDs. |
| 2 | A billed, bounded subset of discovered native models on `/chat/completions` (`--max-sweep-models 0` selects all), classified by the vendored production route classifier | A DEAD route is `fail` only when a validated catalog declares that id as chat-served. Undeclared DEAD routes and transient FLAKY results are `warn`, because `/models` may include embeddings, speech, image, and other non-chat ids. |
| 3 | Non-empty output, deterministic PONG, usage consistency, response model, finish reason, `temperature=0`, forwarded optional fields, and the provider/model-specific max-token spelling | Enclave-breaking request or response behavior is `fail`; tolerated metadata drift is `warn`. Content lists and `reasoning_content`/`reasoning` shapes are accepted. |
| 4 | HTTP status before streaming, strict enclave-readable SSE framing, in-band errors, `[DONE]`, usage, first meaningful delta deadline, and incremental delivery | Silent/truncated output, missing usage, missed output budget, and errors embedded after HTTP 200 are `fail`. Missing `[DONE]` and whole-body buffering are `warn` because the enclave tolerates them but they remain risky. |
| 5 | Forced parallel tool-call deltas, an empty-string assistant tool round-trip, `json_object`, and strict `json_schema` output | A capability declared `false` skips without a completion. An undeclared capability that rejects or ignores the probe is not discovered and `skip`s; transient 429/5xx evidence `warn`s. A declared 4xx rejection or declared accepted-but-invalid output is `fail`. Well-formed single/alternate tool choices `warn` because they are model behavior. |
| 6 | TTFB, TTFT, effective throughput, declared deadlines, vendored leaderboard projection, and a catalog-pricing spend estimate | Entirely advisory. Samples that do not report at least 128 output tokens warn but never change the Tier 1–4 conformance gate or process exit status. |

Tier 4 passes the response through the vendored `_observe_provider_stream`.
Role-only chunks and `: ping` comments do not count as first output; the clock
stops only for a non-empty content, reasoning, or tool delta. Usage may appear
in any chunk. The accepted content/reasoning delta shapes are the same
multi-shape set exported by production.

Tier 5 mirrors two enclave-sensitive details. `stream_translate.go` keys tool
fragments by `index` and starts a tool block from the first observed name, so a
late `function.name` cannot be repaired after translation. `byok.go` replays a
tool-only assistant turn with `content: ""` (not `null`) followed by one
`role: "tool"` message per `tool_call_id`.

Tier 6 runs only when requested. Every `--perf-samples` unit is a billed long
completion with `stream: true`, `stream_options.include_usage: true`, and a
512-token cap. A throughput sample needs at least 128 provider-reported output
tokens. Effective throughput is `output_tokens * 1000 / elapsed_ms`, where
elapsed time starts before the request; this prevents batched SSE delivery from
inflating the result. The same observation is projected into vendored
`ProviderBenchmarkSample` rows and passed to the vendored
`aggregate_leaderboard`; declared time budgets are clamped with
`model_deadlines_declared`. When a validated catalog price matches the model,
the report includes both a cap estimate and an observed-token estimate.

### Completion cost

Every Tier 2 sweep entry and later probe is a real completion that may be
billed by the provider. Let `A` be the number of advertised models, `M` be
`--max-sweep-models`, `S = A` when `M=0` and otherwise `S = min(A, M)`, and `P`
be `--perf-samples`. The table gives the maximum cumulative completions for a
run through each tier; skips, rejected prerequisites, and models for which the
gateway omits temperature can reduce the actual count.

| Highest tier | Additional completions | Maximum cumulative completions |
| --- | ---: | ---: |
| 1 | 0 | 0 |
| 2 | `S` short callability probes | `S` |
| 3 | 7 chat probes | `S + 7` |
| 4 | 1 streaming probe | `S + 8` |
| 5 | 4 capability probes | `S + 12` |
| 6 | `P` long performance probes | `S + 12 + P` |

The default Tier 2 cap is 25. Setting `--max-sweep-models 0` removes the cap
and can therefore bill one completion for every advertised id, including ids
that are not chat routes. Negative values are rejected.

## JSON report

`--json` writes a versioned object with `report_version`, `suite_version`,
`contract_version`, `generated_at`, `target`, `summary`, `checks`,
`performance`, and a reserved unsigned `submission` block. The target records
whether bearer authentication was used without recording the key. The
submission block reserves JCS/RFC8785 canonicalization metadata; signing and
portal upload are intentionally not implemented.

The checked-in schema is
`src/tr_provider_check/data/provider-report.schema.json`. Summary totals cover
all requested tiers, while `summary.conformance_gate` is computed only from
Tiers 1–4. `summary.provider_owned_failures` counts hard failures attributed to
the provider across the report; it is informational and does not redefine the
gate.

The schema enumerates every public check id. The `measured` object is
check-specific; its stable keys are listed below. A branch may set only a
subset, and `reason` is present when evidence is missing, inconclusive, or a
prerequisite skipped.

| Check id | `measured` keys |
| --- | --- |
| `catalog.native-model-discovery` | `http_status`, `model_count`, `model_ids`, `unreadable_row_count`, `error`, `reason` |
| `catalog.declared-v2` | `schema_source`, `declared_model_count`, `error`, `reason` |
| `callability.advertised-models` | `advertised_count`, `swept_count`, `not_swept_count`, `not_swept`, `dead_count`, `declared_dead_count`, `flaky_count`, `models`, `reason` |
| `chat.unavailable` | `reason` |
| `chat.non-empty` | `http_status`, `extracted_characters`, `reason` |
| `chat.pong` | `http_status`, `matched`, `extracted_characters`, `error_type`, `reason` |
| `chat.usage` | `input_tokens`, `output_tokens`, `total_tokens`, `reason` |
| `chat.model` | `requested_model`, `response_model`, `reason` |
| `chat.finish-reason` | `finish_reason`, `reason` |
| `chat.temperature-zero` | `http_status`, `inconclusive`, `reason` |
| `chat.forwarded-fields` | `fields` |
| `chat.max-token-spelling` | `field`, `http_status`, `reason` |
| `stream.unavailable` | `reason` |
| `stream.response-status` | `http_status`, `route_verdict`, `error_type`, `reason` |
| `stream.sse-framing` | `content_type`, `data_line_count`, `invalid_data_line_count`, `wire_unobserved`, `genuinely_empty`, `capture_failed`, `reason` |
| `stream.error-signaling` | `http_status`, `error_before_stream`, `midstream_error_type`, `midstream_error_status` |
| `stream.done` | `done_sentinel`, `reason` |
| `stream.usage` | `input_tokens`, `output_tokens`, `reasoning_tokens`, `reason` |
| `stream.first-delta` | `ttfb_milliseconds`, `first_delta_milliseconds`, `first_content_or_reasoning_delta_milliseconds`, `first_tool_delta_milliseconds`, `budget_milliseconds`, `budget_source`, `reason` |
| `stream.incremental-delivery` | `first_delta_milliseconds`, `last_delta_milliseconds`, `elapsed_milliseconds`, `transport_chunk_count`, `reason` |
| `tools.parallel-deltas` | `capability_declared`, `capability_discovered`, `http_status`, `inconclusive`, `tool_delta_count`, `tool_call_count`, `indices`, `missing_index_count`, `invalid_delta_count`, `late_name_indices`, `argument_errors`, `reason` |
| `tools.round-trip` | `http_status`, `tool_call_count`, `assistant_content_type`, `assistant_content_length`, `completion_nonempty`, `alternative_tool_call_count`, `reason` |
| `structured.json-object` | `capability_declared`, `capability_discovered`, `http_status`, `error`, `inconclusive`, `body_parsed_as_json`, `schema_valid`, `parse_error`, `validation_error`, `reason` |
| `structured.json-schema` | `capability_declared`, `capability_discovered`, `http_status`, `error`, `inconclusive`, `body_parsed_as_json`, `schema_valid`, `parse_error`, `validation_error`, `reason` |
| `perf.production-benchmark` | `requested_samples`, `successful_samples`, `insufficient_samples`, `leaderboard_eligible`, `first_token_deadline_ms`, `reason` |

The mock corpus includes isolated modes for non-SSE HTTP 200 responses, invalid
`data:` framing, missing `[DONE]`, ignored `include_usage`, early role plus late
content, buffered output, finish-only streams, mid-stream error objects, bad
usage, native/catalog ID disagreement, dead/flaky routes, request-field
rejection, tool index/name/argument corruption, empty-string tool-turn
rejection, structured-output mismatch, insufficient throughput usage,
metadata drift, and every vendored stream delta shape. Tests assert
the complete result map for each mode so dependent checks cannot pass
vacuously.

## Relationship to TrustedRouter

The canonical packaged snapshot is
`src/tr_provider_check/data/contract_snapshot.json`. Selected functions in
`src/tr_provider_check/contract.py` are byte-for-byte copies from the production
`Lore-Hex/quill-router` checkout and are protected by normalized source hashes
and behavioral replays. `contract_version` is a hash of the exported contract,
not a branch name.

The HTTP and stream checks are justified against the enclave implementation in
`enclave-go/internal/llm/byok.go`, `stream_translate.go`, and `http_client.go`,
plus `enclave-go/cmd/enclave/provider_stream.go` for the pre-output retry and
first-byte gate. In the supplied enclave reference checkout,
`provider_stream.go` is under `cmd/enclave`, not `internal/llm`.

Run the cross-repository parity gate from the production checkout:

```console
cd /path/to/quill-router
PROVIDER_CHECK_REPO_PATH=/path/to/trustedrouter-provider-check \
  uv run --no-sync pytest tests/test_provider_check_contract_parity.py -q
```

### Synchronizing from production

1. Export the production contract with
   `uv run python scripts/export_provider_check_contract.py` in quill-router.
2. Copy `src/trusted_router/data/provider_check_contract.json` to
   `src/tr_provider_check/data/contract_snapshot.json` here.
3. Replace each source-hash mismatch with the exact production function body.
   Do not reformat vendored bodies.
4. Run the local and cross-repository gates below.

## Development

```console
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
uv build
```

`src/tr_provider_check/contract.py` is excluded from Ruff formatting because
its production function bodies are hash-guarded. See `CONTRIBUTING.md` for the
snapshot sync checklist.

## License

Apache License 2.0; see `LICENSE` and `NOTICE`.
