# ML repository instructions

Python/FastAPI code is under `app/`; tests are under `tests/`. Read
applicable parent instructions before editing, but do not copy host-local
runtime, domain, identity, or inference-provider choices here.

Preserve existing changes and work on `main`; the standard workflow has no
task branches or pull requests. Commit, push, deployment, destructive data
utilities, or live integration work requires separate direct approval.

Run `uv run --group dev ruff check app tests`, `uv run --group dev ruff format
app tests --check`, and `uv run --group dev pytest` when dependencies are
available. Build validation uses the Dockerfile.

Storage uses generic S3-compatible configuration. Autolabeling calls the
endpoint and credential selected in system settings through an
OpenAI-compatible vision inference contract; do not add a provider-specific
dependency or product name.

## Optional provider cancellation

The base OpenAI-compatible contract has no cancellation endpoint. A provider
extension may be used only after runtime capability discovery, and must remain
optional: when it is absent or unavailable, retain local RQ cancellation and
report that provider-side inference could not be cancelled. The currently
documented Unsloth extension uses a batch-scoped `session_id` on completion
requests and `POST /api/inference/cancel`; other providers may expose the same
contract, but must not be assumed to do so.
