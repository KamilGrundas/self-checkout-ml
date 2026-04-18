"""Train a Keras image classifier from YOLO datasets stored in MinIO."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


def _download_datasets(prefixes: list[str], dest: Path) -> None:
    from app.core.object_storage import get_minio_client

    client = get_minio_client()
    bucket = settings.ML_MINIO_TRAINING_BUCKET_NAME
    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_classes: list[str] = []

    for prefix in prefixes:
        prefix = prefix.rstrip("/") + "/"
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            rel = obj.object_name[len(prefix) :]
            if rel.startswith("images/"):
                client.fget_object(
                    bucket, obj.object_name, str(images_dir / Path(rel).name)
                )
            elif rel.startswith("labels/"):
                client.fget_object(
                    bucket, obj.object_name, str(labels_dir / Path(rel).name)
                )
            elif rel == "classes.txt":
                data = client.get_object(bucket, obj.object_name).read()
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


def _split_data(images: list, labels: list, val_ratio: float) -> tuple:
    import numpy as np

    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []

    for i, (img, label) in enumerate(zip(images, labels)):
        h = int(hashlib.sha1(str(i).encode()).hexdigest(), 16) % 100
        if h < int(val_ratio * 100):
            val_imgs.append(img)
            val_labels.append(label)
        else:
            train_imgs.append(img)
            train_labels.append(label)

    return (
        np.stack(train_imgs) if train_imgs else np.empty((0,)),
        np.array(train_labels, dtype=np.int32),
        np.stack(val_imgs) if val_imgs else np.empty((0,)),
        np.array(val_labels, dtype=np.int32),
    )


def _build_model(image_size: int, num_classes: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def check_mlflow() -> None:
    """Raise RuntimeError if MLflow is not reachable."""
    import urllib.error
    import urllib.request

    url = f"{settings.MLFLOW_TRACKING_URI}/health"
    try:
        urllib.request.urlopen(url, timeout=5)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"MLflow is not reachable at {settings.MLFLOW_TRACKING_URI}"
        ) from exc


def train_classifier(
    datasets: list[str],
    *,
    image_size: int = 160,
    epochs: int = 12,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
) -> dict:
    """Train a classifier from YOLO datasets in MinIO and register in MLflow."""
    import os

    import numpy as np

    import mlflow
    import mlflow.keras
    import tensorflow as tf
    from mlflow.models import infer_signature

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

    check_mlflow()
    tf.random.set_seed(42)

    model_name = settings.MLFLOW_REGISTERED_MODEL_NAME

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp) / "dataset"
        _download_datasets(datasets, dataset_dir)
        images, labels, class_names = _load_yolo_dataset(dataset_dir, image_size)

    if len(images) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(images)}")

    logger.info(
        "Loaded %d samples, %d classes: %s", len(images), len(class_names), class_names
    )

    x_train, y_train, x_val, y_val = _split_data(images, labels, validation_ratio)

    model = _build_model(image_size, len(class_names))
    has_val = len(x_val) > 0
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if has_val else "accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]
    # Compute class weights to handle imbalanced classes
    counts = np.bincount(y_train, minlength=len(class_names))
    total = counts.sum()
    class_weight = {
        i: total / (len(class_names) * c) if c > 0 else 1.0
        for i, c in enumerate(counts)
    }
    logger.info(
        "Class distribution: %s, weights: %s", dict(enumerate(counts)), class_weight
    )

    fit_kwargs: dict = {
        "x": x_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "class_weight": class_weight,
        "verbose": 1,
    }
    if has_val:
        fit_kwargs["validation_data"] = (x_val, y_val)

    history = model.fit(**fit_kwargs)

    result: dict = {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "num_classes": len(class_names),
        "classes": class_names,
        "image_size": image_size,
        "epochs_ran": len(history.history["loss"]),
        "datasets": datasets,
    }
    if has_val:
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        result["val_loss"] = float(loss)
        result["val_accuracy"] = float(acc)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "pipeline": "train_classifier",
                "framework": "tensorflow",
                "datasets": ",".join(datasets),
            }
        )
        mlflow.log_params(
            {
                "image_size": image_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "validation_ratio": validation_ratio,
                "num_classes": len(class_names),
            }
        )
        mlflow.log_metrics(
            {k: v for k, v in result.items() if isinstance(v, (int, float))}
        )

        input_example = x_train[:1]
        signature = infer_signature(
            input_example,
            model.predict(input_example, verbose=0),
        )
        mlflow.keras.log_model(
            model,
            name=model_name,
            registered_model_name=model_name,
            signature=signature,
            metadata={
                "labels": class_names,
                "image_size": image_size,
                "num_classes": len(class_names),
            },
        )
        result["run_id"] = run.info.run_id
        result["model_name"] = model_name

    return result
