# Changelog

All notable verdict changes are listed here. Versions 0.1.0 through 0.1.7 map
to the existing release commits and local `v0.1.x` tags.

## Unreleased

- `stream.sse-framing`: empty HTTP 200 bodies change `warn → fail`; populated
  observations with a failed capture remain `warn`.
- `chat.pong` changes `skip → warn` for a transient base response;
  `chat.non-empty` and `chat.usage` change `fail → skip`, while `chat.model` and
  `chat.finish-reason` change `warn → skip`, because those dependent semantics
  were not observed. `chat.max-token-spelling` changes `fail → warn` on
  transient evidence; `chat.temperature-zero` remains `warn`.
- `tools.parallel-deltas` and `tools.round-trip` change `fail → warn` for a
  declared transient transport/429/5xx result; an undeclared transient tool
  probe changes `skip → warn`. `structured.*` retain the already-correct
  transient `warn` behavior through the shared ladder.
- `stream.response-status` and `stream.error-signaling` change `fail → warn` on
  a transport error. `stream.response-status` retains `warn` for 429/5xx.
- `structured.*`: undeclared HTTP 200 prose changes `fail → skip`; declared
  non-conforming output remains `fail`.
- `structured.*`: a declared permanent 4xx rejection changes `warn → fail`.
- `tools.round-trip`: a 200 follow-up tool choice changes `fail → warn`.
- `callability.advertised-models`: a dead model backed by validated chat-route
  evidence changes the previously unreachable advisory result to `fail`.
- `catalog.declared-v2`: an unreachable declaration changes `fail → warn`;
  invalid fetched content remains `fail`.
- `catalog.native-model-discovery`: transient 429/5xx changes `fail → warn`;
  readable rows survive malformed neighbors and remain `pass`.
- `stream.first-delta`: uses a declared first-token budget when available;
  status can change `pass ↔ fail` relative to the fallback budget.

## 0.1.7

- `catalog.native-model-discovery`: readable ids in a bare array or a `data[]`
  envelope without `object:list` change `fail → pass`.

## 0.1.6

- `stream.sse-framing`: conformant gzip-compressed SSE changes `fail → pass`
  after transport decoding. On the conformant gzip fixture, `stream.done`,
  `stream.usage`, and `stream.first-delta` change `skip → pass`, while
  `stream.incremental-delivery` changes `skip → warn` because that release's
  recorder still saw one compressed transport chunk.

## 0.1.5

- `stream.sse-framing`: an empty capture with otherwise observed response data
  changes `fail → warn`; captured malformed framing remains `fail`.

## 0.1.4

- `structured.json-object` and `structured.json-schema`: conforming JSON in the
  assistant content with prose reasoning changes `fail → pass`.

## 0.1.3

- `tools.parallel-deltas`: a well-formed single tool call changes `fail → warn`;
  malformed delta shapes remain `fail`.

## 0.1.2

- `chat.temperature-zero`: transient 429/5xx changes `fail → warn`.
- `structured.json-object` and `structured.json-schema`: transient 429/5xx
  on an undeclared capability changes `skip → warn` because support is unknown;
  a declared transient result remains `warn`.

## 0.1.1

- `chat.temperature-zero`: provider/model families for which the gateway omits
  temperature change from a possible `fail` probe to `skip`.

## 0.1.0

- Introduced the public check ids documented in the report interface. Initial
  statuses had no prior release direction.
