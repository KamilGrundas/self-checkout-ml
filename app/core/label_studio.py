from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx

from app.core.config import settings

TIMEOUT = 60.0
EXPORT_TIMEOUT = 300.0
logger = logging.getLogger(__name__)


def _client(headers: dict[str, str]) -> httpx.Client:
    return httpx.Client(
        base_url=settings.LABEL_STUDIO_URL.rstrip("/"),
        headers=headers,
        timeout=TIMEOUT,
    )


def _resolve_auth_headers() -> dict[str, str]:
    api_key = settings.LABEL_STUDIO_API_KEY
    if not api_key:
        raise RuntimeError("LABEL_STUDIO_API_KEY is not configured")

    for headers in (
        {"Authorization": f"Token {api_key}"},
        {"Authorization": f"Bearer {api_key}"},
    ):
        with _client(headers) as c:
            r = c.get("/api/projects/", params={"page_size": 1})
            if r.status_code not in {400, 401, 403}:
                r.raise_for_status()
                return headers

    with _client({}) as c:
        r = c.post("/api/token/refresh", json={"refresh": api_key})
        if r.status_code in {400, 401}:
            raise RuntimeError("Failed to authenticate to Label Studio")
        r.raise_for_status()
        access_token = r.json().get("access")
        if not access_token:
            raise RuntimeError("Failed to authenticate to Label Studio")
        return {"Authorization": f"Bearer {access_token}"}


def _ensure_project(c: httpx.Client, title: str, label_config: str) -> dict:
    r = c.get("/api/projects/", params={"page_size": 100})
    r.raise_for_status()
    data = r.json()
    projects = data.get("results", data) if isinstance(data, dict) else data

    for project in projects:
        if project.get("title") == title:
            pid = project["id"]
            patch_r = c.patch(
                f"/api/projects/{pid}/",
                json={"label_config": label_config},
            )
            if not patch_r.is_success:
                error_text = patch_r.text
                if patch_r.status_code == 400 and "incompatible" in error_text:
                    logger.warning(
                        "Project '%s' (id=%d) has annotations incompatible with new "
                        "label config — deleting all tasks and retrying. Detail: %s",
                        title,
                        pid,
                        error_text,
                    )
                    c.delete(f"/api/projects/{pid}/tasks/").raise_for_status()
                    c.patch(
                        f"/api/projects/{pid}/",
                        json={"label_config": label_config},
                    ).raise_for_status()
                else:
                    logger.error(
                        "Failed to update label config for project '%s' (id=%d): %s",
                        title,
                        pid,
                        error_text,
                    )
                    patch_r.raise_for_status()
            return c.get(f"/api/projects/{pid}/").json()

    r = c.post("/api/projects/", json={"title": title, "label_config": label_config})
    if not r.is_success:
        logger.error("Failed to create project '%s': %s", title, r.text)
        r.raise_for_status()
    return r.json()


def _import_storage_payload(project_id: int, title: str, bucket: str) -> dict:
    return {
        "project": project_id,
        "title": title,
        "bucket": bucket,
        "prefix": "",
        "aws_access_key_id": settings.MINIO_ACCESS_KEY,
        "aws_secret_access_key": settings.MINIO_SECRET_KEY,
        "s3_endpoint": f"http://{settings.MINIO_ENDPOINT}",
        "region_name": "us-east-1",
        "recursive_scan": True,
        "regex_filter": r".*\.(png|jpg|jpeg|webp)$",
        "presign": False,
        "presign_ttl": 1,
        "use_blob_urls": True,
    }


def _ensure_import_storage_and_sync(
    c: httpx.Client,
    project_id: int,
    title: str,
    bucket: str,
) -> dict:
    """Create or update an S3 import storage and sync it.

    Removes any other S3 import storages from the project that don't match
    the expected title (cleanup from old configurations).

    Returns the sync response so callers can see how many tasks were created.
    """
    r = c.get("/api/storages", params={"project": project_id})
    r.raise_for_status()
    raw = r.json()
    existing = (
        [s for s in raw if s.get("type") == "s3"] if isinstance(raw, list) else []
    )

    payload = _import_storage_payload(project_id, title, bucket)
    storage_id: int | None = None

    for storage in existing:
        if storage.get("title") == title:
            storage_id = storage["id"]
            c.patch(f"/api/storages/s3/{storage_id}", json=payload).raise_for_status()
        else:
            c.delete(f"/api/storages/s3/{storage['id']}").raise_for_status()
            logger.info(
                "Removed old storage '%s' (id=%d)", storage.get("title"), storage["id"]
            )

    if storage_id is None:
        r = c.post("/api/storages/s3/", json=payload)
        r.raise_for_status()
        storage_id = r.json()["id"]

    sync_r = c.post(f"/api/storages/s3/{storage_id}/sync")
    sync_r.raise_for_status()
    return sync_r.json()


def _ensure_export_storage(
    c: httpx.Client,
    project_id: int,
    title: str,
    bucket: str,
    prefix: str,
) -> None:
    r = c.get("/api/storages/export", params={"project": project_id})
    r.raise_for_status()
    raw = r.json()
    existing = (
        [s for s in raw if s.get("type") == "s3"] if isinstance(raw, list) else []
    )

    for storage in existing:
        if storage.get("title") == title:
            return

    c.post(
        "/api/storages/export/s3",
        json={
            "project": project_id,
            "title": title,
            "bucket": bucket,
            "prefix": prefix,
            "aws_access_key_id": settings.MINIO_ACCESS_KEY,
            "aws_secret_access_key": settings.MINIO_SECRET_KEY,
            "s3_endpoint": f"http://{settings.MINIO_ENDPOINT}",
            "region_name": "us-east-1",
            "can_delete_objects": False,
        },
    ).raise_for_status()


def _build_detect_label_config(labels: list[str]) -> str:
    labels_markup = "\n".join(
        f'    <Label value="{label}" background="green"/>' for label in labels
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels_markup}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def _build_classify_label_config(labels: list[str]) -> str:
    choices_markup = "\n".join(f'    <Choice value="{label}"/>' for label in labels)
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <Choices name="choice" toName="image" choice="single" showInLine="true">\n'
        f"{choices_markup}\n"
        "  </Choices>\n"
        "</View>\n"
    )


def _fetch_labels_from_backend() -> list[str]:
    url = f"{settings.BACKEND_URL.rstrip('/')}/api/v1/products/"
    with httpx.Client(timeout=10.0) as c:
        r = c.get(url, params={"limit": 1000})
        r.raise_for_status()
        data = r.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        labels = [item["name"] for item in items if item.get("name")]
    if not labels:
        raise ValueError(
            "No products found in backend — cannot configure Label Studio labels"
        )
    return labels


# Each project: (title_setting, bucket_setting, import_storage_title, export_prefix, is_classify)
_PROJECT_DEFS = [
    (
        "LABEL_STUDIO_SCALE_PROJECT_TITLE",
        "ML_MINIO_SCALE_BUCKET_NAME",
        "scale-images",
        "projects/scale-products",
        True,
    ),
    (
        "LABEL_STUDIO_SHELF_PROJECT_TITLE",
        "ML_MINIO_SHELF_BUCKET_NAME",
        "shelf-images",
        "projects/shelf-products",
        False,
    ),
    (
        "LABEL_STUDIO_EXTERNAL_PROJECT_TITLE",
        "ML_MINIO_EXTERNAL_BUCKET_NAME",
        "external-images",
        "projects/external-products",
        False,
    ),
]


def list_projects() -> list[dict]:
    headers = _resolve_auth_headers()
    with _client(headers) as c:
        r = c.get("/api/projects/", params={"page_size": 100})
        r.raise_for_status()
        data = r.json()
        projects = data.get("results", data) if isinstance(data, dict) else data
        return [{"id": p["id"], "title": p["title"]} for p in projects]


def sync_label_studio() -> dict:
    """Sync MinIO buckets with Label Studio projects.

    Creates/updates 3 projects (scale, shelf, external), each with one
    S3 import storage and one S3 export storage, then syncs all imports.
    Labels are fetched from the backend product catalog.

    Raises httpx.ConnectError if Label Studio is unreachable.
    """
    labels = _fetch_labels_from_backend()
    headers = _resolve_auth_headers()
    classify_config = _build_classify_label_config(labels)
    detect_config = _build_detect_label_config(labels)

    results: dict[str, dict] = {}

    with _client(headers) as c:
        for (
            title_attr,
            bucket_attr,
            storage_title,
            export_prefix,
            is_classify,
        ) in _PROJECT_DEFS:
            project_title = getattr(settings, title_attr)
            bucket_name = getattr(settings, bucket_attr)
            label_config = classify_config if is_classify else detect_config

            project = _ensure_project(c, project_title, label_config)
            pid = project["id"]
            logger.info("Project '%s' (id=%d) ready", project_title, pid)

            sync_result = _ensure_import_storage_and_sync(
                c,
                pid,
                storage_title,
                bucket_name,
            )
            logger.info(
                "Synced storage '%s' for project '%s': %s",
                storage_title,
                project_title,
                sync_result,
            )

            _ensure_export_storage(
                c,
                pid,
                f"{storage_title}-exports",
                settings.ML_MINIO_LABELSTUDIO_EXPORT_BUCKET_NAME,
                export_prefix,
            )

            results[project_title] = {
                "project_id": pid,
                "bucket": bucket_name,
                "status": sync_result.get("status", "unknown"),
                "last_sync_count": sync_result.get("last_sync_count"),
                "tasks_existed": (sync_result.get("meta") or {}).get("tasks_existed"),
            }

    return {"projects": results, "status": "configured"}


def _find_project_by_title(c: httpx.Client, title: str) -> dict:
    r = c.get("/api/projects/", params={"page_size": 100})
    r.raise_for_status()
    data = r.json()
    projects = data.get("results", data) if isinstance(data, dict) else data
    for project in projects:
        if project.get("title") == title:
            return project
    raise ValueError(f"Project '{title}' not found in Label Studio")


def _build_dataset_yaml(classes_path: Path) -> str:
    class_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names_lines = "\n".join(
        f"  {i}: {json.dumps(name, ensure_ascii=False)}"
        for i, name in enumerate(class_names)
    )
    return f"path: .\ntrain: images\nval: images\nnames:\n{names_lines}\n"


def _upload_directory_to_minio(local_root: Path, bucket_name: str, prefix: str) -> int:
    from app.core.object_storage import ensure_bucket_exists, get_minio_client

    ensure_bucket_exists(bucket_name)
    client = get_minio_client()
    uploaded = 0
    ext_to_ct = {
        ".txt": "text/plain",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".yaml": "application/x-yaml",
    }
    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        object_name = f"{prefix.rstrip('/')}/{path.relative_to(local_root).as_posix()}"
        data = path.read_bytes()
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=ext_to_ct.get(path.suffix.lower(), "application/octet-stream"),
        )
        uploaded += 1
    return uploaded


def _download_images_for_labels(
    headers: dict[str, str],
    project_id: int,
    labels_dir: Path,
    images_dir: Path,
) -> None:
    """Download task images from MinIO for each label file.

    YOLO export label filenames match the image filename stem in the S3 URL
    (e.g. label ``abc123.txt`` corresponds to ``s3://bucket/.../abc123.jpg``).
    We fetch all project tasks, build a stem→s3_url map, and download matching
    images directly from MinIO.
    """
    from app.core.object_storage import get_minio_client

    label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix == ".txt"}
    if not label_stems:
        return

    # Build stem→s3_url map from project tasks
    stem_to_s3: dict[str, str] = {}
    with _client(headers) as c:
        r = c.get(f"/api/projects/{project_id}/tasks/", params={"page_size": 10000})
        r.raise_for_status()
        data = r.json()
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        for task in tasks:
            image_url = task.get("data", {}).get("image", "")
            if not image_url:
                continue
            stem = Path(image_url).stem
            if stem in label_stems:
                stem_to_s3[stem] = image_url

    # Download images from MinIO
    minio = get_minio_client()
    for stem, s3_url in stem_to_s3.items():
        ext = Path(s3_url).suffix.lower() or ".jpg"
        dest = images_dir / f"{stem}{ext}"
        if dest.exists():
            continue
        # Parse s3://bucket/key
        parts = s3_url.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            logger.warning("Invalid S3 URL: %s", s3_url)
            continue
        bucket_name, key = parts
        try:
            minio.fget_object(bucket_name, key, str(dest))
        except Exception as exc:
            logger.warning("Failed to download %s: %s", s3_url, exc)

    logger.info(
        "Downloaded %d/%d images for labels",
        len(list(images_dir.iterdir())),
        len(label_stems),
    )


def _parse_classify_tasks(tasks: list[dict], images_dir: Path) -> list[tuple[str, str]]:
    """Extract (filename, label) pairs from Label Studio JSON export of a Choices project.

    Downloads each image from MinIO into images_dir.
    Skips tasks without a choice annotation or whose image cannot be downloaded.
    """
    from app.core.object_storage import get_minio_client

    minio = get_minio_client()
    rows: list[tuple[str, str]] = []

    for task in tasks:
        image_url = task.get("data", {}).get("image", "")
        if not image_url:
            continue

        label: str | None = None
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") == "choices":
                    choices = result.get("value", {}).get("choices", [])
                    if choices:
                        label = choices[0]
                        break
            if label:
                break

        if not label:
            continue

        ext = Path(image_url).suffix.lower() or ".jpg"
        filename = f"{Path(image_url).stem}{ext}"
        dest = images_dir / filename

        if not dest.exists():
            parts = image_url.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                logger.warning("Invalid S3 URL: %s", image_url)
                continue
            bucket_name, key = parts
            try:
                minio.fget_object(bucket_name, key, str(dest))
            except Exception as exc:
                logger.warning("Failed to download %s: %s", image_url, exc)
                continue

        rows.append((filename, label))

    return rows


def _wait_for_export_conversion(
    c: httpx.Client, pid: int, export_id: int, export_type: str
) -> None:
    deadline = time.time() + EXPORT_TIMEOUT
    while time.time() < deadline:
        r = c.get(f"/api/projects/{pid}/exports/{export_id}")
        r.raise_for_status()
        snapshot = r.json()
        for fmt in snapshot.get("converted_formats", []):
            if fmt.get("export_type") != export_type:
                continue
            if fmt["status"] == "completed":
                return
            if fmt["status"] == "failed":
                raise RuntimeError(
                    f"{export_type} conversion failed: {fmt.get('traceback', 'unknown error')}"
                )
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {export_type} conversion")


def export_csv_dataset(
    project_title: str,
    release_name: str | None = None,
) -> dict:
    """Export reviewed Choices annotations from a Label Studio project as a CSV
    dataset (filename,label) with images and upload to the MinIO training bucket.
    """
    headers = _resolve_auth_headers()
    release_name = release_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    export_type = "JSON"

    with _client(headers) as c:
        c.timeout = httpx.Timeout(EXPORT_TIMEOUT)

        project = _find_project_by_title(c, project_title)
        pid = project["id"]
        logger.info("Exporting CSV for project '%s' (id=%d)", project_title, pid)

        r = c.post(
            f"/api/projects/{pid}/exports/",
            json={"annotation_filter_options": {"reviewed": "only"}},
        )
        r.raise_for_status()
        export_id = r.json()["id"]

        r = c.post(
            f"/api/projects/{pid}/exports/{export_id}/convert",
            json={"export_type": export_type, "download_resources": False},
        )
        if r.status_code != 200:
            logger.error("Convert failed (%d): %s", r.status_code, r.text)
            r.raise_for_status()

        _wait_for_export_conversion(c, pid, export_id, export_type)

        r = c.get(
            f"/api/projects/{pid}/exports/{export_id}/download",
            params={"exportType": export_type},
        )
        r.raise_for_status()
        tasks: list[dict] = r.json()

    bucket = settings.ML_MINIO_TRAINING_BUCKET_NAME
    project_slug = project_title.lower().replace(" ", "-")
    release_prefix = f"datasets/releases/{project_slug}/{release_name}"

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp) / "dataset"
        images_dir = extract_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        rows = _parse_classify_tasks(tasks, images_dir)
        if not rows:
            raise ValueError("No reviewed classify annotations found in export")

        csv_path = extract_dir / "dataset.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)

        manifest = {
            "project_id": pid,
            "project_title": project_title,
            "export_id": export_id,
            "export_type": "CSV",
            "release_name": release_name,
            "sample_count": len(rows),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (extract_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        uploaded_files = _upload_directory_to_minio(extract_dir, bucket, release_prefix)

    return {
        "project_id": pid,
        "project_title": project_title,
        "export_id": export_id,
        "release_name": release_name,
        "bucket": bucket,
        "release_prefix": release_prefix,
        "sample_count": len(rows),
        "uploaded_files": uploaded_files,
    }


def export_yolo_dataset(
    project_title: str,
    release_name: str | None = None,
) -> dict:
    """Export reviewed annotations from a Label Studio project as a YOLO
    dataset and upload the result to the MinIO training bucket.

    Steps:
        1. Find project by title
        2. Create export snapshot (reviewed annotations only)
        3. Convert to "YOLO with Images"
        4. Download the zip
        5. Extract, generate dataset.yaml, upload to MinIO

    Raises httpx.ConnectError if Label Studio is unreachable.
    """
    headers = _resolve_auth_headers()
    release_name = release_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    export_type = "YOLO_WITH_IMAGES"

    with _client(headers) as c:
        # Use a longer timeout for export operations
        c.timeout = httpx.Timeout(EXPORT_TIMEOUT)

        project = _find_project_by_title(c, project_title)
        pid = project["id"]
        logger.info("Exporting project '%s' (id=%d)", project_title, pid)

        # 1. Create export snapshot (reviewed only)
        r = c.post(
            f"/api/projects/{pid}/exports/",
            json={"annotation_filter_options": {"reviewed": "only"}},
        )
        r.raise_for_status()
        export_id = r.json()["id"]

        # 2. Convert to YOLO
        r = c.post(
            f"/api/projects/{pid}/exports/{export_id}/convert",
            json={"export_type": export_type, "download_resources": True},
        )
        if r.status_code != 200:
            logger.error("Convert failed (%d): %s", r.status_code, r.text)
            r.raise_for_status()

        # 3. Wait for conversion
        _wait_for_export_conversion(c, pid, export_id, export_type)

        # 4. Download zip
        r = c.get(
            f"/api/projects/{pid}/exports/{export_id}/download",
            params={"exportType": export_type},
        )
        r.raise_for_status()
        archive_bytes = r.content

    # 5. Extract and upload to MinIO
    bucket = settings.ML_MINIO_TRAINING_BUCKET_NAME
    # Use project title as slug (safe for paths)
    project_slug = project_title.lower().replace(" ", "-")
    release_prefix = f"datasets/releases/{project_slug}/{release_name}"

    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp) / "dataset"
        extract_dir.mkdir()
        with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
            zf.extractall(extract_dir)

        # Label Studio YOLO export from S3 storage often doesn't include
        # image files in the zip. Download them from task data if missing.
        images_dir = extract_dir / "images"
        labels_dir = extract_dir / "labels"
        if labels_dir.exists() and (
            not images_dir.exists() or not any(images_dir.iterdir())
        ):
            images_dir.mkdir(exist_ok=True)
            _download_images_for_labels(headers, pid, labels_dir, images_dir)

        classes_path = extract_dir / "classes.txt"
        if classes_path.exists():
            (extract_dir / "dataset.yaml").write_text(
                _build_dataset_yaml(classes_path),
                encoding="utf-8",
            )

        manifest = {
            "project_id": pid,
            "project_title": project_title,
            "export_id": export_id,
            "export_type": export_type,
            "release_name": release_name,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (extract_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        uploaded_files = _upload_directory_to_minio(extract_dir, bucket, release_prefix)

    return {
        "project_id": pid,
        "project_title": project_title,
        "export_id": export_id,
        "release_name": release_name,
        "bucket": bucket,
        "release_prefix": release_prefix,
        "uploaded_files": uploaded_files,
    }


def export_dataset(
    project_title: str,
    release_name: str | None = None,
) -> dict:
    """Export a Label Studio project dataset.

    Routes to CSV export for classify projects (scale) and YOLO export for
    detect projects (shelf, external), based on _PROJECT_DEFS.
    """
    for title_attr, _, _, _, is_classify in _PROJECT_DEFS:
        if getattr(settings, title_attr) == project_title:
            if is_classify:
                return export_csv_dataset(project_title, release_name)
            else:
                return export_yolo_dataset(project_title, release_name)
    known = [getattr(settings, t) for t, *_ in _PROJECT_DEFS]
    raise ValueError(
        f"Unknown project title: '{project_title}'. Known projects: {known}"
    )
