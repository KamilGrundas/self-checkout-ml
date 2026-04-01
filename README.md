# self-checkout-ml

Minimal FastAPI service for storing checkout session snapshots in MinIO.

The service is intentionally small:
- accepts image uploads for a checkout session,
- stores them in MinIO with deterministic names,
- lists stored snapshots for a given session,
- provides a local extraction script for dataset bootstrapping.

Snapshot naming:
- empty shelf: `0000-empty.<ext>`
- first product capture: `0001-product.<ext>`
- second product capture: `0002-product.<ext>`

The extension depends on the uploaded image format, for example `.png` or `.jpg`.

The object path in MinIO is:
- `sessions/<session_id>/captures/<filename>`

## API

`GET /api/v1/utils/health-check/`
- returns `true`

`POST /api/v1/checkout-sessions/{session_id}/snapshots`
- multipart form-data
- fields:
  - `capture_index` required
  - `product_id` optional
  - `product_name` optional
  - `file` required image

`GET /api/v1/checkout-sessions/{session_id}/snapshots`
- returns ordered snapshots for the session

Each response item includes:
- `capture_index`
- `product_id`
- `product_name`
- `filename`
- `object_name`
- `image_url`

## Extraction Script

The repository also includes a script that compares consecutive session snapshots,
detects the newly appeared object, crops it from the current frame, and saves it
under a directory named after the product label.

Run it with:

```bash
uv run python scripts/extract_labeled_objects.py \
  --api-base-url http://127.0.0.1:8001 \
  --session-id <SESSION_ID> \
  --output-dir data/extracted \
  --threshold 25 \
  --min-area 5000 \
  --padding 12
```

The script creates:
- cropped product images under `data/extracted/<product_name>/`
- `data/extracted/labels.csv` with file path, session, capture index, product id,
  product name, and the detected bounding box

How it works:
- loads ordered session snapshots from the ML API,
- compares each image with the previous one,
- extracts the largest newly appeared region,
- saves the crop under a directory named after `product_name`.

## Configuration

Copy `.env.example` to `.env` and set:
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_PUBLIC_URL`
- `ML_MINIO_BUCKET_NAME`

For the shared local stack from `self-checkout-infra`, the default local endpoint is:
- `http://127.0.0.1:8001`

The extracted dataset under `data/` is local output and should not be committed.

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py
```
