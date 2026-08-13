# Contributing

Thank you for helping make provider conformance failures easier to diagnose.
Changes should keep the project honest about what the current release actually
runs and preserve its exact relationship to TrustedRouter production.

## Set up and test

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```console
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
uv build
```

Tests should include a positive control as well as the intended failure. A mode
that emits no meaningful response is not a useful reproduction of a narrow
provider incompatibility.

## The vendored-contract rule

`src/tr_provider_check/contract.py` contains byte-for-byte copies of production
functions. Never reformat, refactor, or hand-edit a vendored function body,
even when a change appears behavior-preserving. Ruff's formatter exclusion is
intentional, and the fidelity suite hashes normalized function source.

Full-line comments around a copied body are permitted because the normalizer
strips them. Run the complete suite afterward to prove fidelity still holds.

## Re-syncing production

1. In a `Lore-Hex/quill-router` checkout, run:

   ```console
   uv run python scripts/export_provider_check_contract.py
   ```

2. Copy `src/trusted_router/data/provider_check_contract.json` to this
   repository as `src/tr_provider_check/data/contract_snapshot.json`.
3. Replace each function named by a fidelity failure with the exact function
   body from the quill-router source path in the error. Preserve documented
   public-package adaptations; do not make opportunistic cleanup changes.
4. Run the local gates above.
5. From quill-router, run:

   ```console
   PROVIDER_CHECK_REPO_PATH=/path/to/trustedrouter-provider-check \
     uv run --no-sync pytest tests/test_provider_check_contract_parity.py -q
   ```

Automatic cross-repository CI wiring is intentionally not claimed until this
repository is published.

## Review checklist

- The test would fail if the behavior named in it were broken.
- Negative-path fixtures retain a conforming positive shape or control.
- New provenance-tagged symbols are source-hashed or added to the explicit,
  reasoned completeness allowlist.
- The installed console script and built wheel contents remain covered.
- Public documentation distinguishes tested behavior from design intent.
