from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LabelPrototype:
    label: str
    vector: np.ndarray


class VisualClassifier:
    def __init__(self) -> None:
        # Feature order:
        # brightness, contrast, saturation, warm_ratio, cool_ratio,
        # green_ratio, edge_density, portrait_bias, document_bias, center_focus
        self.prototypes = [
            LabelPrototype(
                "Landscape",
                np.array([0.62, 0.28, 0.50, 0.25, 0.36, 0.48, 0.26, 0.34, 0.10, 0.42], dtype=np.float32),
            ),
            LabelPrototype(
                "Portrait",
                np.array([0.58, 0.24, 0.40, 0.42, 0.16, 0.12, 0.18, 0.82, 0.05, 0.71], dtype=np.float32),
            ),
            LabelPrototype(
                "Document",
                np.array([0.86, 0.22, 0.05, 0.12, 0.10, 0.08, 0.64, 0.24, 0.92, 0.38], dtype=np.float32),
            ),
            LabelPrototype(
                "Night",
                np.array([0.18, 0.34, 0.25, 0.08, 0.56, 0.10, 0.22, 0.31, 0.02, 0.48], dtype=np.float32),
            ),
            LabelPrototype(
                "Abstract",
                np.array([0.52, 0.58, 0.81, 0.40, 0.32, 0.28, 0.71, 0.44, 0.06, 0.35], dtype=np.float32),
            ),
        ]
        self.prototype_matrix = np.stack([item.vector for item in self.prototypes])

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        features = self._extract_features(image_array, width, height)
        feature_vector = np.array(
            [
                features["brightness"],
                features["contrast"],
                features["saturation"],
                features["warm_ratio"],
                features["cool_ratio"],
                features["green_ratio"],
                features["edge_density"],
                features["portrait_bias"],
                features["document_bias"],
                features["center_focus"],
            ],
            dtype=np.float32,
        )

        distances = np.linalg.norm(self.prototype_matrix - feature_vector, axis=1)
        logits = -4.0 * distances
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()

        ranked = sorted(
            (
                {
                    "label": prototype.label,
                    "confidence": round(float(probability), 4),
                }
                for prototype, probability in zip(self.prototypes, probabilities, strict=True)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "label": ranked[0]["label"],
            "confidence": ranked[0]["confidence"],
            "predictions": ranked[:3],
            "features": {
                key: round(float(value), 4) for key, value in features.items()
            },
            "width": width,
            "height": height,
        }

    def _extract_features(
        self, image_array: np.ndarray, width: int, height: int
    ) -> dict[str, float]:
        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]

        brightness_map = (red + green + blue) / 3.0
        brightness = float(brightness_map.mean())
        contrast = float(brightness_map.std())

        max_channel = image_array.max(axis=2)
        min_channel = image_array.min(axis=2)
        saturation = float(np.mean(max_channel - min_channel))

        warm_ratio = float(np.mean((red > green) & (green > blue * 0.7)))
        cool_ratio = float(np.mean((blue > red) & (blue > green * 0.8)))
        green_ratio = float(np.mean((green > red) & (green > blue)))

        horizontal_edges = np.abs(np.diff(brightness_map, axis=1))
        vertical_edges = np.abs(np.diff(brightness_map, axis=0))
        edge_density = float((horizontal_edges.mean() + vertical_edges.mean()) / 2.0)

        portrait_bias = float(min(height / max(width, 1), 2.0) / 2.0)

        grayscale = brightness_map
        document_bias = float(
            np.mean(grayscale > 0.82) * 0.65
            + np.mean((max_channel - min_channel) < 0.08) * 0.35
        )

        center_y1 = max(int(height * 0.25), 0)
        center_y2 = max(int(height * 0.75), 1)
        center_x1 = max(int(width * 0.25), 0)
        center_x2 = max(int(width * 0.75), 1)
        center_region = brightness_map[center_y1:center_y2, center_x1:center_x2]
        center_focus = float(center_region.mean() if center_region.size else brightness)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "warm_ratio": warm_ratio,
            "cool_ratio": cool_ratio,
            "green_ratio": green_ratio,
            "edge_density": edge_density,
            "portrait_bias": portrait_bias,
            "document_bias": document_bias,
            "center_focus": center_focus,
        }
