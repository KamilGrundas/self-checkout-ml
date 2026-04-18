# self-checkout-ml

`self-checkout-ml` stores raw checkout images in MinIO, exposes upload and
inference APIs, prepares local datasets, and integrates with Label Studio and
MLflow.

The current repository covers four areas:
- FastAPI API for session snapshots and classifier inference
- MinIO storage for raw shelf, scale, upload, and training-release data
- local extraction and review pipeline
- TensorFlow/Keras classifier training with MLflow logging
- TensorFlow/Keras multilabel shelf classifier training with MLflow logging
- Label Studio export-to-bucket dataset build script

## Repository Layout

- `app/` - FastAPI application, MinIO integration, inference loader
- `scripts/` - extraction, review, import, and reset utilities
- `ml/datasets/` - local generated datasets and external images
- `ml/manifests/` - local CSV manifests produced by the pipeline
- `ml/reports/` - optional local analysis outputs
- `train/train_classifier/` - current TensorFlow/Keras baseline classifier trainer
- `train/train_detector/` - current multilabel shelf classifier trainer

## Raw Snapshot Storage

`ML_label` sessions are stored in MinIO as raw captures.

Naming:
- empty shelf baseline: `0000-empty.<ext>`
- first labeled capture: `0001-product.<ext>`
- second labeled capture: `0002-product.<ext>`

MinIO object path:
- `sessions/<session_id>/captures/<filename>`

The extension depends on the uploaded image format, for example `.png` or `.jpg`.

Additional raw buckets:
- scale images: `ML_MINIO_SCALE_BUCKET_NAME`
- manually uploaded images: `ML_MINIO_EXTERNAL_BUCKET_NAME`
- Label Studio raw exports: `ML_MINIO_LABELSTUDIO_EXPORT_BUCKET_NAME`
- built training releases: `ML_MINIO_TRAINING_BUCKET_NAME`

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
- stores the snapshot in `ML_MINIO_SHELF_BUCKET_NAME`

`GET /api/v1/checkout-sessions/{session_id}/shelf-snapshots`
- returns ordered shelf snapshots for the session

`POST /api/v1/checkout-sessions/{session_id}/scale-snapshots`
- multipart form-data
- fields:
  - `capture_index` required
  - `product_id` optional
  - `product_name` optional
  - `file` required image
- stores the snapshot in `ML_MINIO_SCALE_BUCKET_NAME`

`GET /api/v1/checkout-sessions/{session_id}/scale-snapshots`
- returns ordered scale snapshots for the session

`POST /api/v1/datasets/shelf-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw shelf images in `ML_MINIO_SHELF_BUCKET_NAME` under `raw/shelf/`

`POST /api/v1/datasets/scale-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw scale images in `ML_MINIO_SCALE_BUCKET_NAME` under `raw/scale/`

`POST /api/v1/datasets/external-images`
- multipart form-data
- field:
  - `files` required, multiple images
- stores raw uploaded images in `ML_MINIO_EXTERNAL_BUCKET_NAME` under `raw/uploaded/`

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

If `--limit` is omitted, the script processes all sessions it discovers in MinIO.

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
- annotate them in Label Studio from MinIO source storage
- export a reviewed release with `scripts/build_dataset.py`

## Training

The current baseline trainer lives in `train/train_classifier/` and uses stable
TensorFlow with Keras. The model is logged directly to MLflow through the Keras
flavor and is not stored locally as the source of truth.

Training from approved extracted samples and external samples:

```bash
uv run python train/train_classifier/train.py \
  --manifest-path ml/manifests/extracted_objects.csv \
  --dataset-root ml/datasets/extracted \
  --external-manifest ml/manifests/external_objects.csv \
  --external-dataset-root ml/datasets/external
```

Training only from external samples also works. If `ml/manifests/extracted_objects.csv`
does not exist, the trainer skips it automatically and uses only the external manifest.

The trainer stores the model directly in MLflow Model Registry and keeps label
order in MLflow model metadata.

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

## Shelf Classifier Training

The shelf classifier lives in `train/train_detector/` and trains a multilabel
Keras model for the whole shelf image.

It uses:
- cumulative full-scene shelf snapshots derived from `source_image_curr`
- optional external one-label images for additional support per class

Typical command:

```bash
uv run python -m train.train_detector.train \
  --manifest-path ml/manifests/extracted_objects.csv \
  --external-manifest ml/manifests/external_objects.csv \
  --external-dataset-root ml/datasets/external \
  --mlflow-tracking-uri http://127.0.0.1:5002
```

The shelf trainer:
- caches downloaded shelf source images under `ml/datasets/shelf/source_images/`
- trains a multilabel classifier with sigmoid outputs
- logs datasets, reports, and the registered Keras model to MLflow
- registers the trained model as `self-checkout-shelf-classifier` in MLflow Model Registry

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
- creates or updates the `scale-products` and `shelf-products` projects
- connects `scale-images`, `uploaded-images`, and `session-images` MinIO buckets
- connects a raw export bucket for Label Studio snapshot exports

Default local endpoint:
- `http://127.0.0.1:8080`

## Build Dataset

`scripts/build_dataset.py` creates a reviewed export snapshot in Label Studio,
converts it to `YOLO with Images`, and uploads the extracted release to the
training bucket in MinIO.

Example:

```bash
uv run python scripts/build_dataset.py \
  --label-studio-url http://127.0.0.1:8080 \
  --label-studio-api-key <API_KEY> \
  --project-id 1 \
  --project-slug scale-products
```

The release is uploaded under:
- `datasets/releases/<project-slug>/<release-name>/`

Typical contents:
- `classes.txt`
- `images/`
- `labels/`
- `notes.json`
- `dataset.yaml`
- `manifest.json`

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

The default local stack does not start `ml-dev` automatically. `./scripts/up.sh`
keeps the main stack running and stops `mlflow` and `label-studio` if they were started earlier.

For local training from the host machine, use:
- `http://127.0.0.1:5002`

For `self-checkout-ml` running inside Docker in the shared infra stack, use:
- `http://mlflow:5000`

You can override the tracking URI explicitly:

```bash
uv run python train/train_classifier/train.py \
  --mlflow-tracking-uri http://127.0.0.1:5002
```

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

This reset does not remove raw session snapshots stored in MinIO.

## Configuration

Copy `.env.example` to `.env`.

Important variables:
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_PUBLIC_URL`
- `ML_MINIO_SHELF_BUCKET_NAME`
- `ML_MINIO_SCALE_BUCKET_NAME`
- `ML_MINIO_EXTERNAL_BUCKET_NAME`
- `ML_MINIO_TRAINING_BUCKET_NAME`
- `ML_MINIO_LABELSTUDIO_EXPORT_BUCKET_NAME`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_REGISTERED_MODEL_NAME`
- `MLFLOW_SHELF_MODEL_NAME`
- `LABEL_STUDIO_URL`
- `LABEL_STUDIO_API_KEY`

`LABEL_STUDIO_API_KEY` should be a personal access token from Label Studio.
`build_dataset.py` exchanges it through `/api/token/refresh` and then uses the
returned Bearer access token for API calls.

For the shared local stack from `self-checkout-infra`, the relevant host endpoints are:
- ML API: `http://127.0.0.1:8001`
- MLflow: `http://127.0.0.1:5002`
- Label Studio: `http://127.0.0.1:8080`

Inside Docker in the shared stack:
- MinIO: `minio:9000`
- MLflow: `mlflow:5000`

## Python Version

The repository is pinned to Python `3.12.13` via `.python-version`.

Core ML stack:
- `tensorflow`
- `keras`
- `mlflow`

## Run Locally

Install dependencies:

```bash
uv sync --python 3.12.13
```

Run the API:

```bash
uv run fastapi dev app/main.py
```

Train locally from the host machine:

```bash
uv run python train/train_classifier/train.py \
  --mlflow-tracking-uri http://127.0.0.1:5002
```
