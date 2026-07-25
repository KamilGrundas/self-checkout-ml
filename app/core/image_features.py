"""CPU-friendly image features used by scikit-learn classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ProductImageFeatures(TransformerMixin, BaseEstimator):
    """Combine shape, colour, and coarse spatial information for product images."""

    def __init__(
        self,
        image_size: int,
        *,
        orientations: int = 9,
        pixels_per_cell: int = 16,
        histogram_bins: int = 16,
        spatial_size: int = 16,
    ) -> None:
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.histogram_bins = histogram_bins
        self.spatial_size = spatial_size

    def fit(self, x: Any, y: Any = None) -> ProductImageFeatures:
        return self

    def transform(self, x: Any) -> np.ndarray:
        import cv2
        from skimage.feature import hog

        images = np.asarray(x, dtype=np.float32).reshape(
            -1, self.image_size, self.image_size, 3
        )
        features: list[np.ndarray] = []
        cells_per_block = 2 if self.image_size >= self.pixels_per_cell * 2 else 1

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            shape_features = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                block_norm="L2-Hys",
                feature_vector=True,
            ).astype(np.float32)
            colour_features = np.concatenate(
                [
                    np.histogram(
                        image[:, :, channel],
                        bins=self.histogram_bins,
                        range=(0.0, 1.0),
                        density=True,
                    )[0].astype(np.float32)
                    for channel in range(3)
                ]
            )
            spatial_features = cv2.resize(
                image,
                (self.spatial_size, self.spatial_size),
                interpolation=cv2.INTER_AREA,
            ).reshape(-1)
            shape_features = self._normalize_group(shape_features)
            colour_features = self._normalize_group(colour_features)
            spatial_features = self._normalize_group(spatial_features)
            features.append(
                np.concatenate([shape_features, colour_features, spatial_features])
            )

        return np.stack(features)

    @staticmethod
    def _normalize_group(features: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(features))
        return features / norm if norm > 0 else features
