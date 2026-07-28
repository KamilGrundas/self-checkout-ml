# self-checkout-ml

`self-checkout-ml` stores raw checkout images in S3-compatible object storage, exposes upload and
inference APIs, prepares local datasets, and integrates with Label Studio and
MLflow.

The current repository covers:
- FastAPI API for session snapshots and classifier inference
- S3-compatible object storage storage for raw shelf, scale, upload, and training-release data
- local extraction and review pipeline
- durable Redis/RQ product-classifier training with MLflow logging
- durable, sequential Redis/RQ scale-image autolabeling through a configurable
  local VLM
- Label Studio export-to-bucket dataset build script

## Repository Layout

- `app/` - FastAPI application, S3-compatible object storage integration, inference loader
- `scripts/` - extraction, review, import, and reset utilities
- `ml/datasets/` - local generated datasets and external images
- `ml/manifests/` - local CSV manifests produced by the pipeline
- `ml/reports/` - optional local analysis outputs
- `app/core/training.py` - scikit-learn classifier training pipeline
- `app/core/training_queue.py` - durable Redis/RQ queue integration
- `app/core/training_worker.py` - worker entry point and persisted progress updates
- `app/core/autolabel.py` - prompt, strict response parser, inference client,
  catalog snapshot, and durable result sidecars
- `app/core/autolabel_queue.py` - idempotency and dedicated RQ queue integration
- `app/core/autolabel_worker.py` - per-image fault isolation and batch progress

## Raw Snapshot Storage

`ML_label` sessions are stored in S3-compatible object storage as raw captures.

Naming:
- empty shelf baseline: `0000-empty.<ext>`
- first labeled capture: `0001-product.<ext>`
- second labeled capture: `0002-product.<ext>`

S3-compatible object storage object path:
- `sessions/<session_id>/captures/<filename>`

The extension depends on the uploaded image format, for example `.png` or `.jpg`.

Additional raw buckets:
- scale images: `S3_SCALE_BUCKET`
- manually uploaded images: `S3_EXTERNAL_BUCKET`
- Label Studio raw exports: `S3_LABEL_STUDIO_EXPORT_BUCKET`
- built training releases: `S3_TRAINING_BUCKET`

## API

`GET /api/v1/utils/health-check/`
- returns `true`

`POST /api/v1/checkout-sessions/{session_id}/shelf-snapshots`
- multipart form-data
- fields:
  - `capture_index` required
  - `product_id` optional
  - `product_name` optional
  - `file` required image
- stores the snapshot in `S3_SHELF_BUCKET`

`GET /api/v1/checkout-sessions/{session_id}/shelf-snapshots`
- returns ordered shelf snapshots for the session

`POST /api/v1/checkout-sessions/{session_id}/scale-snapshots`
- multipart form-data
- fields:
  - `capture_index` required
  - `product_id` optional
  - `product_name` optional
  - `file` required image
- stores the snapshot in `S3_SCALE_BUCKET`

`GET /api/v1/checkout-sessions/{session_id}/scale-snapshots`
- returns ordered scale snapshots for the session

### Scale autolabeling

All endpoints below require a backend-issued superuser JWT:

- `GET /api/v1/autolabel/scale/images` — cursor-paginated image list from
  `S3_SCALE_BUCKET`, excluding non-images and the reserved sidecar prefix
- `GET /api/v1/autolabel/scale/images/content` — authenticated image delivery
  used by admin thumbnails; browsers never receive Compose-only S3 URLs
- `POST /api/v1/autolabel/scale/test` — invokes the configured VLM for one
  image without persisting a label
- `POST /api/v1/autolabel/scale/batches` — returns HTTP 202 and requires an
  `Idempotency-Key`
- `GET /api/v1/autolabel/scale/batches/latest` — restores the latest retained
  batch for the current superuser
- `GET /api/v1/autolabel/scale/batches/{batch_id}` — batch and per-image status

Before enqueueing, ML loads the complete paginated product catalog and the
global endpoint configuration from the backend. A batch stores snapshots of
both. Candidates use deterministic keys (`P0001`, `P0002`, ...); only a key
present in that snapshot can map to a backend product UUID.

The VLM request is bounded multipart form-data with `prompt`, `max_tokens`, and
the original image bytes. Redirects are disabled, connect/read timeouts are
explicit, and the response is size-limited. The parser accepts the directly
observed endpoint envelope, whose `response` field contains either plain JSON
or one fenced JSON object. Descriptive output, extra fields, invalid JSON, and
unknown keys never produce a product assignment.

`scale-autolabel` is consumed by a dedicated concurrency-1 RQ worker so it
cannot block classifier training. Queue state may expire, but final
`matched`/`unmatched` results are stored as schema-versioned JSON sidecars below
`_autolabel/scale/v1/`, keyed by a hash of the complete source object name.
Sidecars contain the source fingerprint, endpoint/prompt snapshot, safe
response diagnostics, result, and batch identifier. A source ETag/size change
invalidates the old result. Source images and their manual metadata are never
overwritten.

`POST /api/v1/datasets/shelf-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw shelf images in `S3_SHELF_BUCKET` under `raw/shelf/`

`POST /api/v1/datasets/scale-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw scale images in `S3_SCALE_BUCKET` under `raw/scale/`

`POST /api/v1/datasets/external-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw uploaded images in `S3_EXTERNAL_BUCKET` under `raw/uploaded/`

Each snapshot item includes:
- `capture_index`
- `product_id`
- `product_name`
- `filename`
- `object_name`
- `image_url`

`POST /api/v1/inference/classify`
- multipart form-data
- field:
  - `file` required image
- returns ordered class probabilities, for example:

```json
{
  "scores": {
    "Banan": 0.84,
    "Ananas": 0.10,
    "Kiwi": 0.06
  },
  "run_id": "..."
}
```

The inference API loads the latest registered version of
`self-checkout-classifier` from MLflow Model Registry and reads label order
from MLflow model metadata.

`POST /api/v1/inference/detect`
- multipart form-data
- field:
  - `file` required image
- returns multilabel shelf scores, for example:

```json
{
  "scores": {
    "Kiwi": 0.99,
    "Banan": 0.99
  },
  "run_id": "..."
}
```

`POST /api/v1/inference/refresh-classify-model`
- forces reload of the latest registered `self-checkout-classifier` model
- rate limited to once per minute per API process
- returns the model name, version, and MLflow run id

`POST /api/v1/inference/refresh-detect-model`
- forces reload of the latest registered `self-checkout-shelf-classifier` model from MLflow Model Registry
- rate limited to once per minute per API process
- returns the shelf model name, version, and MLflow run id

Runtime behavior:
- `classify` uses the model cached in memory
- `detect` uses the shelf classifier cached in memory
- if the process restarts, both models load from local disk cache when available
- MLflow is only required for training and refresh endpoints, not for every inference request

## Verification

```bash
uv run --group dev ruff check app tests
uv run --group dev ruff format app tests --check
uv run --group dev pytest
```

Docker, object-storage integration, and full-stack validation run through the
workspace-controlled `../ops/dev-test.sh --repo ml` command on `ssh dev`.

## Local Data Layout

Generated local files are kept in the repository, but ignored by Git.

- `ml/datasets/extracted/` - auto-cropped products from labeled sessions
- `ml/datasets/reviewed/approved/` - manually approved crops
- `ml/datasets/reviewed/rejected/` - manually rejected crops
- `ml/datasets/external/` - manually added external images grouped by folder name
- `ml/manifests/extracted_objects.csv` - extraction manifest with review status
- `ml/manifests/external_objects.csv` - external dataset manifest

Older local runs may still contain data under `data/extracted/`. Review and
training scripts still accept that legacy location, but new outputs should go to
`ml/datasets/extracted/`.

## Extraction

`extract_labeled_objects.py` compares consecutive snapshots from one session,
detects the newly appeared region, and saves the crop under the product label.

Single session:

```bash
uv run python scripts/extract_labeled_objects.py \
  --api-base-url http://127.0.0.1:8001 \
  --session-id <SESSION_ID> \
  --output-dir ml/datasets/extracted \
  --manifest-path ml/manifests/extracted_objects.csv \
  --threshold 25 \
  --min-area 5000 \
  --padding 12
```

Batch mode:

```bash
uv run python scripts/extract_all_sessions.py \
  --api-base-url http://127.0.0.1:8001 \
  --limit 50 \
  --output-dir ml/datasets/extracted \
  --manifest-path ml/manifests/extracted_objects.csv
```

If `--limit` is omitted, the script processes all sessions it discovers in S3-compatible object storage.

The extraction manifest stores:
- `file_path`
- `session_id`
- `capture_index`
- `product_id`
- `product_name`
- bounding box coordinates
- source image references
- `review_status`
- review metadata

## Review

`review_extracted_objects.py` lets you confirm or reject extracted crops before training.

```bash
uv run python scripts/review_extracted_objects.py \
  --manifest-path ml/manifests/extracted_objects.csv \
  --dataset-root ml/datasets/extracted \
  --reviewed-root ml/datasets/reviewed
```

Review keys:
- `A` - approve
- `R` - reject
- `S` - skip and keep the row as `pending`
- `Q` - quit

The review window uses a fixed size and only scales images down when needed.

## External Dataset

You can add images manually without any mapping file. Folder names are used as class names.

Expected structure:

```text
ml/datasets/external/
├── Ananas/
├── Banan/
└── Kiwi/
```

Then build the manifest:

```bash
uv run python scripts/import_external_dataset.py
```

This creates:
- `ml/manifests/external_objects.csv`

This local folder-based flow is still supported, but the preferred cloud flow is:
- upload raw images to `POST /api/v1/datasets/external-images`
- annotate them in Label Studio from S3-compatible object storage source storage
- export a reviewed release with `scripts/build_dataset.py`

## Training

The product classifier uses a scikit-learn pipeline with HOG, colour-histogram,
and coarse spatial image features. Training runs in a dedicated RQ worker, and
job state and progress survive an ML API restart. The model is logged directly
to MLflow through the scikit-learn flavor and is not stored locally as the
source of truth.

Training is triggered via `POST /api/v1/train/classifier` with S3-compatible object storage dataset prefixes:

```json
{
  "yolo_datasets": ["datasets/releases/shelf-products/...", "datasets/releases/external-products/..."],
  "csv_datasets": ["datasets/releases/scale-products/..."],
  "image_size": 160,
  "epochs": 12,
  "batch_size": 16,
  "validation_ratio": 0.2
}
```

- `yolo_datasets` — YOLO releases (shelf, external); images are **cropped** from bounding boxes
- `csv_datasets` — CSV releases (scale); **whole images** are used directly

Both dataset types can be combined in a single training run. Class indices are unified across all sources before training.

The trainer stores the model directly in MLflow Model Registry and keeps label
order in MLflow model metadata. Registering a candidate does not activate it;
activation remains an explicit operation through the model-version API.

MLflow logging includes:
- parameters
- metrics
- input datasets visible in MLflow UI
- dataset summary
- manifest artifacts
- model artifact
- report artifact

Logged MLflow datasets:
- `classifier_input_dataset` - full input dataset used by the run
- `classifier_train_split` - training split
- `classifier_validation_split` - validation split, if validation data exists
- `classifier_extracted_dataset` - only samples coming from `ml/datasets/extracted`
- `classifier_external_dataset` - only samples coming from `ml/datasets/external`

Each logged dataset row includes dataset provenance, including:
- `source` such as `extracted` or `external`
- `product_name`
- `session_id`
- `capture_index`
- `file_path`

## Label Studio

In the shared local setup, Label Studio is started as part of `ml-dev` from
`self-checkout-infra`.

Start the default stack:

```bash
cd /Users/kamilgrundas/Repositories/self-checkout/self-checkout-infra
./scripts/up.sh
```

Start `ml-dev`:

```bash
cd /Users/kamilgrundas/Repositories/self-checkout/self-checkout-infra
./scripts/up-ml-dev.sh
```

`label-studio-init` automatically:
- creates or updates the `scale-products`, `shelf-products`, and `external-products` projects
- connects `scale-images`, `uploaded-images`, and `session-images` S3-compatible object storage buckets
- connects a raw export bucket for Label Studio snapshot exports

Project labeling schema:
- `scale-products` — **Choices** (single-label image classification, whole images)
- `shelf-products` — **RectangleLabels** (bounding box detection)
- `external-products` — **RectangleLabels** (bounding box detection)

Labels are fetched automatically from the backend product catalog (`GET /api/v1/products/`) on each sync.
`LABEL_STUDIO_LABELS` is no longer used.

Default local endpoint:
- `http://127.0.0.1:8080`

## Build Dataset

`POST /api/v1/label-studio/export` creates a reviewed export snapshot in Label
Studio and uploads the release to the training bucket in S3-compatible object
storage. The authenticated superuser's saved Label Studio personal access token
is loaded from the backend for each operation.

The export format depends on the project:
- `scale-products` → **CSV** (`dataset.csv` + `images/`) — whole images for classification
- `shelf-products` → **YOLO with Images** (`classes.txt`, `images/`, `labels/`, `dataset.yaml`)
- `external-products` → **YOLO with Images**

The release is uploaded under:
- `datasets/releases/<project-slug>/<release-name>/`

## MLflow

MLflow is not required to be running all the time for inference.
The runtime flow is:
- train or register a model through MLflow
- call `POST /api/v1/inference/refresh-classify-model` or `POST /api/v1/inference/refresh-detect-model`
- keep serving `classify` and `detect` requests from in-memory or disk cache

In the shared local setup, MLflow is started separately from `self-checkout-infra`:

Without MLflow:

```bash
cd /Users/kamilgrundas/Repositories/self-checkout/self-checkout-infra
./scripts/up.sh
```

With `ml-dev`:

```bash
cd /Users/kamilgrundas/Repositories/self-checkout/self-checkout-infra
./scripts/up-ml-dev.sh
```

For local MLflow access from the host machine, use
`http://127.0.0.1:5002`. The ML API and worker use `http://mlflow:5000`
inside the shared Compose stack.

Submit training through `POST /api/v1/train/classifier`; the API queues the
request and the dedicated RQ worker executes it. Poll
`GET /api/v1/train/{job_id}` for durable progress and the final result.

If you see an error like `403` while the trainer tries to create or read an
experiment, it usually means the tracking URI points to the wrong service or to
the wrong port on the host machine.

Current naming:
- experiment: `self-checkout-classifier`
- registered model: `self-checkout-classifier`

Inference selection:
- the API does not use the latest run anymore
- it uses the latest registered model version from MLflow Model Registry for
  `self-checkout-classifier`
- shelf inference uses the latest registered version of `self-checkout-shelf-classifier`

## Reset Local Data

To remove local extracted data, review outputs, manifests, cache, and reports:

```bash
uv run python scripts/reset_local_data.py
```

To also remove the local `uv` cache used in this repository:

```bash
uv run python scripts/reset_local_data.py --include-cache
```

This reset does not remove raw session snapshots stored in S3-compatible object storage.

## Configuration

Copy `.env.example` to `.env`.

Important variables:
- `S3_ENDPOINT_URL`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_PUBLIC_BASE_URL`
- `S3_SHELF_BUCKET`
- `S3_SCALE_BUCKET`
- `S3_EXTERNAL_BUCKET`
- `S3_TRAINING_BUCKET`
- `S3_LABEL_STUDIO_EXPORT_BUCKET`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_REGISTERED_MODEL_NAME`
- `MLFLOW_SHELF_MODEL_NAME`
- `TRAINING_QUEUE_URL`
- `LABEL_STUDIO_URL`
- `BACKEND_URL`

The Label Studio personal access token is saved for the authenticated
superuser through the admin UI. The backend encrypts it at rest and does not
include it in the public user profile. The ML service loads it from the backend
for the current authenticated operation, exchanges it through
`/api/token/refresh` when required, and does not persist it locally.

`BACKEND_URL` points to the backend API used to fetch product names as Label
Studio labels during sync. In Docker it is set to `http://backend:8000` directly
in `compose.yml` and does not need to be set in `.env`.

For the shared local stack from `self-checkout-infra`, the relevant host endpoints are:
- ML API: `http://127.0.0.1:8001`
- MLflow: `http://127.0.0.1:5002`
- Label Studio: `http://127.0.0.1:8080`

Inside Docker in the shared stack:
- S3-compatible object storage: `s3-provider:8080`
- MLflow: `mlflow:5000`
- Redis/RQ: `redis:6379`

## Python Version

The repository is pinned to Python `3.13.14` via `.python-version`.

Core ML stack:
- `scikit-learn` for product classifier training and inference
- `scikit-learn` with HOG, colour histogram, and coarse spatial features for
  both product and shelf classifiers
- Redis/RQ for durable, multi-worker training jobs and progress
- `mlflow`

## Run Locally

Install dependencies:

```bash
uv sync --python 3.13.14
```

Run the API:

```bash
uv run fastapi dev app/main.py
```

For a standalone local API and worker, start Redis and then run:

```bash
uv run rq worker --url redis://127.0.0.1:6379/0 classifier-training
```
