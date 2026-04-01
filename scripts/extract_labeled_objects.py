from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


def fetch_session_snapshots(api_base_url: str, session_id: str) -> list[dict]:
    url = f"{api_base_url.rstrip('/')}/api/v1/checkout-sessions/{session_id}/snapshots"
    with urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def download_image(url: str) -> np.ndarray:
    with urlopen(url) as response:
        image_bytes = response.read()

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image from {url}")
    return image


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def extract_new_object(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    *,
    threshold_value: int,
    min_area: int,
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if previous_image.shape != current_image.shape:
        raise ValueError("Previous and current image sizes differ")

    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    previous_blur = cv2.GaussianBlur(previous_gray, (11, 11), 0)
    current_blur = cv2.GaussianBlur(current_gray, (11, 11), 0)

    diff = cv2.absdiff(previous_blur, current_blur)
    _, threshold = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        raise ValueError("No new object contour found")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(current_image.shape[1], x + w + padding)
    y2 = min(current_image.shape[0], y + h + padding)

    return current_image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def save_crop(
    output_dir: Path,
    session_id: str,
    snapshot: dict,
    crop: np.ndarray,
) -> Path:
    product_name = snapshot.get("product_name") or "unknown"
    capture_index = snapshot["capture_index"]
    label_dir = output_dir / slugify(product_name)
    label_dir.mkdir(parents=True, exist_ok=True)

    file_path = label_dir / f"{session_id}_{capture_index:04d}.png"
    if not cv2.imwrite(str(file_path), crop):
        raise ValueError(f"Failed to save crop to {file_path}")
    return file_path


def append_manifest_row(
    manifest_path: Path,
    saved_path: Path,
    output_dir: Path,
    session_id: str,
    snapshot: dict,
    bbox: tuple[int, int, int, int],
) -> None:
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        if not file_exists:
            writer.writerow(
                [
                    "file_path",
                    "session_id",
                    "capture_index",
                    "product_id",
                    "product_name",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                ]
            )
        writer.writerow(
            [
                str(saved_path.relative_to(output_dir)),
                session_id,
                snapshot["capture_index"],
                snapshot.get("product_id") or "",
                snapshot.get("product_name") or "",
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
            ]
        )


def process_session(
    api_base_url: str,
    session_id: str,
    output_dir: Path,
    *,
    threshold_value: int,
    min_area: int,
    padding: int,
) -> None:
    snapshots = sorted(
        fetch_session_snapshots(api_base_url, session_id),
        key=lambda item: item["capture_index"],
    )
    if len(snapshots) < 2:
        raise ValueError("At least two snapshots are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "labels.csv"

    previous_snapshot = snapshots[0]
    previous_image = download_image(previous_snapshot["image_url"])

    for snapshot in snapshots[1:]:
        current_image = download_image(snapshot["image_url"])

        product_name = snapshot.get("product_name")
        if not product_name:
            previous_image = current_image
            continue

        crop, bbox = extract_new_object(
            previous_image,
            current_image,
            threshold_value=threshold_value,
            min_area=min_area,
            padding=padding,
        )
        saved_path = save_crop(output_dir, session_id, snapshot, crop)
        append_manifest_row(
            manifest_path,
            saved_path,
            output_dir,
            session_id,
            snapshot,
            bbox,
        )
        previous_image = current_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract newly appeared objects from session snapshots and save them with product labels."
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", default="data/extracted")
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--padding", type=int, default=12)
    args = parser.parse_args()

    process_session(
        api_base_url=args.api_base_url,
        session_id=args.session_id,
        output_dir=Path(args.output_dir),
        threshold_value=args.threshold,
        min_area=args.min_area,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
