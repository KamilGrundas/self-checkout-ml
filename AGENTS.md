# ML repository instructions

Python 3.13 FastAPI service. API code is under `app/api`, core storage/training/inference integrations under `app/core`, and dependency/configuration metadata is in `pyproject.toml` and `uv.lock`.

Read `../AGENTS.md` first. Inspect Git with `git -C self-checkout-ml`; never edit directly on `main` or `master`, combine repositories in one commit, or commit `.env`, `.venv`, caches, datasets, snapshots, trained model artifacts, Label Studio tokens, MLflow credentials, or S3 credentials. Destructive dataset/reset utilities require explicit user approval and must never be part of routine validation.

Current static validation is `uv run --group dev ruff check app` and `uv run --group dev ruff format app --check`; build validation uses the Dockerfile. No automated test suite or separate type checker is currently configured, so do not claim those checks exist. Exercise API health and affected integrations on remote dev.

Use `../ops/dev-sync.sh --repo ml --dry-run`, then `../ops/dev-test.sh --repo ml`. Keep commits focused and imperative; coordinate inference contracts, storage buckets, and model-registry changes with backend, admin, and infra PRs.

The base branch is `main` as recorded in `../repos.yaml`. Create short-lived branches from a freshly fetched `origin/main`, and never implement directly on `main` or `master`. Use Conventional Commits with scopes such as `ml`, `api`, `inference`, `training`, `datasets`, or `storage`.

Definition of Done: Ruff lint and formatting, image build, API health, and affected integration checks pass on remote dev; new behavior has tests when a test suite is introduced; model/dataset artifacts and credentials remain untracked; resource and model compatibility are documented; and rollback is stated.

The ML service uses only generic S3 settings, including separate bucket names
for captures, uploads, training data, and exports. It must support custom
endpoints, TLS verification, path-style addressing, timeouts, and retries
without provider-specific code. `MLFLOW_TRACKING_URI` is optional for API
health and storage-only scenarios but required for training, registry, and
model-loading workflows. MLflow backend and artifact configuration belongs to
infra. Coordinate storage, training, registry, and inference contract changes
with backend and infra and validate the current stack on dev.
