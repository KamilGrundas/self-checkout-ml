# Self-checkout ML service

FastAPI service responsible for checkout-session snapshots, native image
labeling, dataset creation, classifier training, model storage, and inference.
It has no Label Studio or MLflow dependency.

## Runtime

The API runs with two Redis/RQ workers:

- `classifier-training` trains product classifiers;
- `scale-autolabel` calls the configured vision inference provider sequentially
  for scale images.

All durable images, annotations, datasets, trained model artifacts, metrics,
and active-model pointers use generic S3-compatible object storage. Provider
selection remains an infrastructure decision.

## Labeling workflow

Scale snapshots are listed by `GET /api/v1/autolabel/scale/images`. The native
admin UI shows capture time, the label selected during checkout, and an
editable final Label. Autolabeling fills only empty Labels. A different
autolabel result is highlighted for review.

Important endpoints:

- `POST /api/v1/autolabel/scale/batches` starts vision-inference autolabeling;
- `PATCH /api/v1/autolabel/scale/images/label` corrects a Label with an existing product;
- `POST /api/v1/autolabel/scale/images/finalize` moves reviewed images to the labeled collection;
- `POST /api/v1/datasets/scale-images` imports images without a known Label for review;
- `GET /api/v1/datasets/images` lists finalized labeled images;
- `POST /api/v1/datasets/images/import` imports up to 100 images with one existing product Label;
- `PATCH /api/v1/datasets/images/label` corrects a finalized image;
- `POST /api/v1/datasets/images/export` exports selected finalized images.
- `POST /api/v1/datasets/images/duplicates` starts a background 99% similarity scan;
- `DELETE /api/v1/datasets/images/duplicates/{job_id}` removes only duplicates reported by the completed scan.

Dataset releases are self-contained under
`datasets/releases/labeled-images/<release>/` in `S3_TRAINING_BUCKET` and
contain `images/`, `dataset.csv`, and `manifest.json`.

## Training and models

`POST /api/v1/train/classifier` queues training from selected CSV and/or YOLO
datasets. A completed job writes the model artifact and metadata under
`models/classifier/versions/<version>/` in `S3_TRAINING_BUCKET` and activates
the new version. Metadata includes labels, image size, parameters, training and
validation metrics, creation time, and a stable model ID.

`GET /api/v1/inference/classify-models` lists versions and metrics;
`POST /api/v1/inference/set-classify-model` changes the active version. The
inference process lazily loads the active artifact and caches it in memory.

## Configuration

Required non-local settings are `S3_ENDPOINT_URL`, `S3_SHELF_BUCKET`,
`S3_SCALE_BUCKET`, `S3_EXTERNAL_BUCKET`, `S3_TRAINING_BUCKET`, and
`TRAINING_QUEUE_URL`. The generic S3 region, credentials, session token, TLS,
path-style addressing, timeout, retry, and bucket-creation settings are shared
with the rest of the project. Never commit `.env`, datasets, snapshots, model
artifacts, or credentials.

## Validation

```bash
uv run --group dev ruff check app tests
uv run --group dev ruff format app tests --check
uv run --group dev pytest
```

Docker builds and integration validation run only on `dev` through the parent
workspace scripts.
