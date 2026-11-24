"""Set 8.2: classify the image with the InceptionV3 transfer learning model."""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from skimage.transform import resize

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult
from .dialogs import ask_details_action

ACTION_SELECT = "Select image to classify…"

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

MODEL_DIR = Path(__file__).resolve().parents[2] / "Set8" / "output"
WEIGHTS_PATH = MODEL_DIR / "transfer_learning_inception_weights.weights.h5"
PASSPORT_PATH = MODEL_DIR / "transfer_learning_inception_passport.json"

# Same input size used to fine-tune the model in the Set 8 notebook
INPUT_SIZE = (128, 128)
TOP_K = 3

DEFAULT_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


def load_passport(passport_path: Path = PASSPORT_PATH) -> dict[str, Any]:
    with open(passport_path, encoding="utf-8") as file:
        return json.load(file)


def load_class_names(passport_path: Path = PASSPORT_PATH) -> tuple[str, ...]:
    try:
        names = load_passport(passport_path).get("class_names")
    except (OSError, ValueError):
        return DEFAULT_CLASS_NAMES
    return tuple(names) if names else DEFAULT_CLASS_NAMES


def format_passport_value(value: Any) -> str:
    """Turn a passport value into one readable line for the details panel."""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {format_passport_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def build_model(num_classes: int) -> Any:
    """Rebuild the notebook architecture: InceptionV3 base + new dense head.

    The architecture JSON saved by the notebook comes from a newer Keras than
    the one installed locally, so the layers are rebuilt here and only the
    weights are loaded from disk.
    """
    import keras  # imported lazily: TensorFlow takes seconds to load

    base = keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=(None, None, 3),
        name="inception_v3",
    )
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = keras.layers.Dense(1024, activation="relu", name="custom_dense")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(
        x
    )
    return keras.Model(base.input, outputs)


def preprocess(image: Image.Image) -> np.ndarray:
    """Scale to [0, 1] and resize to 128x128x3, as in the notebook."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    resized = resize(array, (*INPUT_SIZE, 3), preserve_range=True, anti_aliasing=True)
    return resized[np.newaxis].astype(np.float32)


def top_predictions(
    probabilities: np.ndarray, class_names: tuple[str, ...], k: int = TOP_K
) -> list[Prediction]:
    order = np.argsort(probabilities)[::-1][:k]
    return [Prediction(class_names[i], float(probabilities[i])) for i in order]


def annotate(image: Image.Image, best: Prediction) -> Image.Image:
    """Draw the predicted label in a banner on top of the image."""
    result = image.convert("RGB").copy()
    text = f"{best.label} ({best.probability:.1%})"
    font_size = max(14, result.width // 20)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(result)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    padding = font_size // 3
    draw.rectangle(
        [0, 0, right - left + 2 * padding, bottom - top + 2 * padding],
        fill=(0, 0, 0),
    )
    draw.text((padding - left, padding - top), text, fill=(255, 255, 0), font=font)
    return result


class CnnClassificationTool(ForensicsTool):
    tool_id = "cnn_classification"
    title = "CNN classification (InceptionV3)"
    category = "Set 8 duties"
    description = (
        "Show the passport of the Set 8.2 transfer learning InceptionV3 model, "
        "then pick an image and classify it into a CIFAR-10 class."
    )
    requires_image = False  # the image is chosen from the passport window

    def __init__(
        self,
        weights_path: Path = WEIGHTS_PATH,
        passport_path: Path = PASSPORT_PATH,
    ) -> None:
        self.weights_path = weights_path
        self.passport_path = passport_path
        self.class_names = load_class_names(passport_path)
        self._model: Any = None

    def _get_model(self) -> Any:
        # Load once and keep it: building InceptionV3 is slow
        if self._model is None:
            model = build_model(len(self.class_names))
            model.load_weights(str(self.weights_path))
            self._model = model
        return self._model

    def classify(self, image: Image.Image) -> list[Prediction]:
        probabilities = self._get_model().predict(preprocess(image), verbose=0)[0]
        return top_predictions(probabilities, self.class_names)

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        try:
            details = self.passport_details()
        except (OSError, ValueError) as error:
            return ToolResult(message=f"Could not read the model passport: {error}")

        # Show the passport first; its button leads to picking the image
        if not ask_details_action(
            parent, self.title, "Model passport", details, ACTION_SELECT
        ):
            return None

        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose an image to classify",
            message="Select the image to classify with the Set 8.2 model "
            "(one of the CIFAR-10 classes: "
            f"{', '.join(self.class_names)}).",
            filetypes=OPEN_TYPES,
        )
        if not filename:
            return None
        try:
            # The picked image becomes the open image, so Undo shows it clean
            document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            return ToolResult(message=f"Could not open image: {error}")
        return self.classify_document(document)

    def passport_details(self) -> dict[str, str]:
        """Passport fields as readable ``{label: text}`` pairs."""
        passport = load_passport(self.passport_path)
        return {
            key.replace("_", " ").capitalize(): format_passport_value(value)
            for key, value in passport.items()
        }

    def classify_document(self, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        if not self.weights_path.exists():
            return ToolResult(message=f"Model weights not found: {self.weights_path}")
        try:
            predictions = self.classify(document.current)
        except ImportError:
            return ToolResult(
                message="TensorFlow/Keras is not installed. "
                "Run: pip install tensorflow"
            )

        best = predictions[0]
        details: dict[str, Any] = {
            "Operation": "CNN classification (transfer learning)",
            "Model": "InceptionV3 (ImageNet) + Dense(1024) + Dense(10)",
            "Dataset": "CIFAR-10",
            "Input size": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}x3",
            "Prediction": f"{best.label} ({best.probability:.1%})",
        }
        for rank, prediction in enumerate(predictions, start=1):
            details[f"Top {rank}"] = f"{prediction.label}: {prediction.probability:.1%}"

        return ToolResult(
            image=annotate(document.current, best),
            message=f"Predicted class: {best.label} ({best.probability:.1%}).",
            details=details,
        )
