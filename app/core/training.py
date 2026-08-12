"""Train an image classifier from datasets stored in S3-compatible object storage."""

from __future__ import annotations

import csv
import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

from app.core.config import settings

logger = logging.getLogger(__name__)

TrainingProgressCallback = Callable[[dict[str, object]], None]


def _download_datasets(prefixes: list[str], dest: Path) -> None:
    from app.core.object_storage import get_object_storage

    client = get_object_storage()
    bucket = settings.S3_TRAINING_BUCKET
    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_classes: list[str] = []

    for prefix in prefixes:
        prefix = prefix.rstrip("/") + "/"
        for obj in client.list_objects(bucket, prefix=prefix):
            rel = obj.object_name[len(prefix) :]
            if rel.startswith("images/"):
                client.download_file(
                    bucket, obj.object_name, str(images_dir / Path(rel).name)
                )
            elif rel.startswith("labels/"):
                client.download_file(
                    bucket, obj.object_name, str(labels_dir / Path(rel).name)
                )
            elif rel == "classes.txt":
                data = client.get_bytes(bucket, obj.object_name)
                for line in data.decode("utf-8").strip().splitlines():
                    if line.strip() and line.strip() not in all_classes:
                        all_classes.append(line.strip())

    (dest / "classes.txt").write_text("\n".join(all_classes) + "\n", encoding="utf-8")


def _load_yolo_dataset(
    dataset_dir: Path,
    image_size: int,
) -> tuple[list, list, list[str]]:
    import cv2
    import numpy as np

    class_names = [
        line.strip()
        for line in (dataset_dir / "classes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 classes, got {len(class_names)}: {class_names}"
        )

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    loaded_images = []
    loaded_labels = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
        if not lines:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_idx = int(parts[0])
            if class_idx >= len(class_names):
                continue

            xc, yc, bw, bh = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))

            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            crop = rgb[y1:y2, x1:x2]
            resized = cv2.resize(
                crop, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            loaded_images.append(resized.astype(np.float32) / 255.0)
            loaded_labels.append(class_idx)

    return loaded_images, loaded_labels, class_names


def _download_csv_datasets(prefixes: list[str], dest: Path) -> None:
    from app.core.object_storage import get_object_storage

    client = get_object_storage()
    bucket = settings.S3_TRAINING_BUCKET
    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, str]] = []

    for idx, prefix in enumerate(prefixes):
        prefix = prefix.rstrip("/") + "/"
        csv_data: str | None = None
        image_objects: dict[str, str] = {}  # filename -> object_name

        for obj in client.list_objects(bucket, prefix=prefix):
            rel = obj.object_name[len(prefix) :]
            if rel == "dataset.csv":
                csv_data = client.get_bytes(bucket, obj.object_name).decode("utf-8")
            elif rel.startswith("images/"):
                image_objects[Path(rel).name] = obj.object_name

        if csv_data is None:
            logger.warning("No dataset.csv in prefix %s, skipping", prefix)
            continue

        for row in csv.DictReader(io.StringIO(csv_data)):
            filename = row.get("filename", "")
            label = row.get("label", "")
            if not filename or not label:
                continue
            unique_name = f"{idx}_{filename}"
            if filename in image_objects:
                dest_path = images_dir / unique_name
                if not dest_path.exists():
                    client.download_file(bucket, image_objects[filename], dest_path)
                all_rows.append((unique_name, label))

    with (dest / "dataset.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(all_rows)


def _load_csv_dataset(
    dataset_dir: Path,
    image_size: int,
) -> tuple[list, list, list[str]]:
    import cv2
    import numpy as np

    csv_path = dataset_dir / "dataset.csv"
    if not csv_path.exists():
        return [], [], []

    images_dir = dataset_dir / "images"
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    class_names = sorted({row["label"] for row in rows if row.get("label")})
    if not class_names:
        return [], [], []
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    loaded_images = []
    loaded_labels = []

    for row in rows:
        filename = row.get("filename", "")
        label = row.get("label", "")
        if not filename or label not in class_to_idx:
            continue
        img_path = images_dir / filename
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        loaded_images.append(resized.astype(np.float32) / 255.0)
        loaded_labels.append(class_to_idx[label])

    return loaded_images, loaded_labels, class_names


def _merge_datasets(
    *groups: tuple[list, list, list[str]],
) -> tuple[list, list, list[str]]:
    """Merge multiple (images, labels, class_names) groups into a unified dataset."""
    import numpy as np

    all_classes = sorted({name for _, _, names in groups for name in names})
    if not all_classes:
        return [], [], []
    class_to_idx = {name: i for i, name in enumerate(all_classes)}

    merged_images: list = []
    merged_labels: list = []

    for images, labels, class_names in groups:
        old_to_new = {i: class_to_idx[name] for i, name in enumerate(class_names)}
        merged_images.extend(images)
        merged_labels.extend(old_to_new[lbl] for lbl in labels)

    return merged_images, np.array(merged_labels, dtype=np.int32).tolist(), all_classes


def _split_data(images: list, labels: list, val_ratio: float) -> tuple:
    import numpy as np

    labels_array = np.asarray(labels, dtype=np.int32)
    random = np.random.default_rng(42)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(set(labels)):
        class_indices = np.flatnonzero(labels_array == label)
        random.shuffle(class_indices)
        desired_validation = int(round(len(class_indices) * val_ratio))
        validation_count = min(desired_validation, max(0, len(class_indices) - 1))
        val_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    return (
        np.stack([images[index] for index in train_indices]),
        labels_array[train_indices],
        (
            np.stack([images[index] for index in val_indices])
            if val_indices
            else np.empty((0,))
        ),
        labels_array[val_indices],
    )


def _fit_classifier(
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[Any, dict[str, float]]:
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.pipeline import Pipeline

    from app.core.image_features import ProductImageFeatures

    x_train_flat = x_train.reshape(len(x_train), -1)
    has_val = len(x_val) > 0
    x_val_flat = x_val.reshape(len(x_val), -1) if has_val else x_val
    image_size = x_train.shape[1]
    feature_transformer = ProductImageFeatures(image_size=image_size)
    x_train_features = feature_transformer.fit_transform(x_train_flat)
    x_val_features = (
        feature_transformer.transform(x_val_flat) if has_val else x_val_flat
    )
    classes = np.arange(num_classes, dtype=np.int32)
    classifier = SGDClassifier(
        loss="log_loss",
        random_state=42,
        alpha=0.001,
        learning_rate="adaptive",
        eta0=0.01,
        average=True,
    )

    counts = np.bincount(y_train, minlength=num_classes)
    total = counts.sum()
    class_weight = {
        i: total / (num_classes * count) if count > 0 else 1.0
        for i, count in enumerate(counts)
    }
    logger.info(
        "Class distribution: %s, weights: %s", dict(enumerate(counts)), class_weight
    )

    random = np.random.default_rng(42)
    first_batch = True
    last_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        indices = random.permutation(len(x_train_features))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_y = y_train[batch_indices]
            sample_weight = np.array(
                [class_weight[int(label)] for label in batch_y],
                dtype=np.float64,
            )
            classifier.partial_fit(
                x_train_features[batch_indices],
                batch_y,
                classes=classes if first_batch else None,
                sample_weight=sample_weight,
            )
            first_batch = False

        train_probabilities = classifier.predict_proba(x_train_features)
        last_metrics = {
            "accuracy": float(
                accuracy_score(y_train, classifier.predict(x_train_features))
            ),
            "loss": float(log_loss(y_train, train_probabilities, labels=classes)),
        }
        if has_val:
            validation_probabilities = classifier.predict_proba(x_val_features)
            last_metrics["val_accuracy"] = float(
                accuracy_score(y_val, classifier.predict(x_val_features))
            )
            last_metrics["val_loss"] = float(
                log_loss(y_val, validation_probabilities, labels=classes)
            )
        if epoch_callback:
            epoch_callback(epoch + 1, last_metrics)

    model = Pipeline(
        [
            ("image_features", feature_transformer),
            ("classifier", classifier),
        ]
    )
    return model, last_metrics


def train_classifier(
    yolo_datasets: list[str],
    csv_datasets: list[str] | None = None,
    *,
    image_size: int = 160,
    epochs: int = 12,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
    progress_callback: TrainingProgressCallback | None = None,
) -> dict:
    """Train and store a classifier in generic S3-compatible object storage."""

    def report(
        stage: str,
        message: str,
        progress: float,
        *,
        current_epoch: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        if progress_callback:
            progress_callback(
                {
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "current_epoch": current_epoch,
                    "total_epochs": epochs,
                    "metrics": metrics,
                }
            )

    with tempfile.TemporaryDirectory() as tmp:
        groups = []

        if yolo_datasets:
            report("downloading", "Downloading YOLO datasets", 5)
            yolo_dir = Path(tmp) / "yolo"
            _download_datasets(yolo_datasets, yolo_dir)
            report("loading", "Loading YOLO images and annotations", 12)
            groups.append(_load_yolo_dataset(yolo_dir, image_size))

        if csv_datasets:
            report("downloading", "Downloading CSV datasets", 5)
            csv_dir = Path(tmp) / "csv"
            _download_csv_datasets(csv_datasets, csv_dir)
            report("loading", "Loading CSV images and labels", 12)
            groups.append(_load_csv_dataset(csv_dir, image_size))

        if not groups:
            raise ValueError("No datasets provided")

        report("preparing", "Preparing training and validation data", 20)
        images, labels, class_names = _merge_datasets(*groups)

    if len(images) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(images)}")
    if len(class_names) < 2 or len(set(labels)) < 2:
        raise ValueError(f"Need at least 2 represented classes, got {len(set(labels))}")

    logger.info(
        "Loaded %d samples, %d classes: %s", len(images), len(class_names), class_names
    )

    x_train, y_train, x_val, y_val = _split_data(images, labels, validation_ratio)
    has_val = len(x_val) > 0
    report("training", f"Starting epoch 1 of {epochs}", 25, current_epoch=0)

    def report_epoch(current_epoch: int, metrics: dict[str, float]) -> None:
        report(
            "training",
            f"Completed epoch {current_epoch} of {epochs}",
            25 + (current_epoch / epochs) * 60,
            current_epoch=current_epoch,
            metrics=metrics,
        )

    model, last_metrics = _fit_classifier(
        x_train,
        y_train,
        x_val,
        y_val,
        num_classes=len(class_names),
        epochs=epochs,
        batch_size=batch_size,
        epoch_callback=report_epoch,
    )

    result: dict = {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "num_classes": len(class_names),
        "classes": class_names,
        "image_size": image_size,
        "epochs_ran": epochs,
        "yolo_datasets": yolo_datasets,
        "csv_datasets": csv_datasets or [],
        "accuracy": last_metrics["accuracy"],
        "loss": last_metrics["loss"],
    }
    if has_val:
        report("evaluating", "Evaluating the trained model", 88)
        result["val_loss"] = last_metrics["val_loss"]
        result["val_accuracy"] = last_metrics["val_accuracy"]

    report("saving", "Saving the trained model", 92)
    from app.core.inference import classifier_model_store

    registered = classifier_model_store.register(
        model=model,
        labels=class_names,
        image_size=image_size,
        metrics={name: float(value) for name, value in last_metrics.items()},
        parameters={
            "image_size": image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_ratio": validation_ratio,
            "num_classes": len(class_names),
            "yolo_datasets": yolo_datasets,
            "csv_datasets": csv_datasets or [],
        },
    )
    result["model_id"] = registered["model_id"]
    result["model_name"] = registered["name"]
    result["model_version"] = registered["version"]

    return result
