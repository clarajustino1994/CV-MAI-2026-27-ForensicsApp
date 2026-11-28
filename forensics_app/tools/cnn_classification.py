"""Set 8: classify an image with one of the CNN models trained in the notebook.

The user first picks the model (shallow CNN, deeper VGG, complete VGG or the
InceptionV3 transfer learning model), then sees its passport, then picks the
image to classify.
"""

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
from .dialogs import ask_choice, ask_details_action

ACTION_SELECT = "Select image to classify…"

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

MODEL_DIR = Path(__file__).resolve().parents[2] / "Set8" / "output"

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
class CnnModel:
    """One model saved by the Set 8 notebook in ``MODEL_DIR``."""

    title: str
    file_tag: str  # <file_tag>_weights.weights.h5, _architecture.json, _passport.json
    features_tag: str  # test_features_<features_tag>.npz (+ _passport.json)
    input_size: tuple[int, int]  # size the image is resized to before predicting
    architecture: str
    rebuild_inception: bool = False  # its JSON does not load with the local Keras

    def weights_path(self, folder: Path) -> Path:
        return folder / f"{self.file_tag}_weights.weights.h5"

    def architecture_path(self, folder: Path) -> Path:
        return folder / f"{self.file_tag}_architecture.json"

    def passport_path(self, folder: Path) -> Path:
        return folder / f"{self.file_tag}_passport.json"

    def features_path(self, folder: Path) -> Path:
        return folder / f"test_features_{self.features_tag}.npz"

    def features_passport_path(self, folder: Path) -> Path:
        return folder / f"test_features_{self.features_tag}_passport.json"


SHALLOW_CNN = CnnModel(
    title="Shallow CNN",
    file_tag="shallow_cnn",
    features_tag="shallow_cnn",
    input_size=(32, 32),
    architecture="Conv(32) + MaxPool + Dense(100) + Dense(10)",
)
DEEPER_VGG = CnnModel(
    title="Deeper VGG",
    file_tag="deeper_vgg",
    features_tag="deeper_vgg",
    input_size=(32, 32),
    architecture="2 x Conv(32) + MaxPool + Dense(128) + Dense(10)",
)
COMPLETE_VGG = CnnModel(
    title="Complete VGG",
    file_tag="vgg_complete",
    features_tag="vgg_complete",
    # 32x32 in: the model resizes to 64x64 itself with its Resizing layer
    input_size=(32, 32),
    architecture="Resizing(64x64) + 3 VGG blocks (64, 128, 256 filters) "
    "+ Dense(128) + Dense(10)",
)
TRANSFER_INCEPTION = CnnModel(
    title="Transfer learning (InceptionV3)",
    file_tag="transfer_learning_inception",
    features_tag="inception_v3_transfer",
    input_size=(128, 128),  # same input size used to fine-tune it
    architecture="InceptionV3 (ImageNet) + Dense(1024) + Dense(10)",
    rebuild_inception=True,
)

MODELS = (SHALLOW_CNN, DEEPER_VGG, COMPLETE_VGG, TRANSFER_INCEPTION)


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


def choose_model(
    parent: tk.Misc, title: str, prompt: str, models: tuple[CnnModel, ...] = MODELS
) -> CnnModel | None:
    """One button per model; return the picked one, or ``None`` on cancel."""
    choice = ask_choice(parent, title, prompt, [model.title for model in models])
    return None if choice is None else models[choice]


def load_passport(passport_path: Path) -> dict[str, Any]:
    with open(passport_path, encoding="utf-8") as file:
        return json.load(file)


def load_class_names(passport_path: Path) -> tuple[str, ...]:
    try:
        names = load_passport(passport_path).get("class_names")
    except (OSError, ValueError):
        return DEFAULT_CLASS_NAMES
    return tuple(names) if names else DEFAULT_CLASS_NAMES


def summarize_history(history: dict[str, list[float]]) -> str:
    """``{metric: [one value per epoch]}`` -> epoch count and final values."""
    epochs = max(len(values) for values in history.values())
    finals = ", ".join(
        f"{metric} {values[-1]:.3f}" for metric, values in history.items() if values
    )
    return f"{epochs} epochs; final {finals}"


def format_passport_value(value: Any) -> str:
    """Turn a passport value into one readable line for the details panel."""
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(v, list) and v for v in value.values())
        and all(isinstance(x, (int, float)) for v in value.values() for x in v)
    ):
        return summarize_history(value)  # training history: too long to list
    if isinstance(value, dict):
        return ", ".join(f"{k}: {format_passport_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def passport_details(passport_path: Path) -> dict[str, str]:
    """Passport fields as readable ``{label: text}`` pairs."""
    return {
        key.replace("_", " ").capitalize(): format_passport_value(value)
        for key, value in load_passport(passport_path).items()
    }


def build_inception(num_classes: int) -> Any:
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


def _drop_keys(obj: Any, keys: set[str]) -> Any:
    """Recursively remove config keys a newer Keras wrote into the JSON."""
    if isinstance(obj, dict):
        return {k: _drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop_keys(v, keys) for v in obj]
    return obj


def load_keras_model(model: CnnModel, folder: Path, num_classes: int) -> Any:
    """Load the architecture and weights saved by the notebook."""
    import keras  # imported lazily: TensorFlow takes seconds to load

    if model.rebuild_inception:
        loaded = build_inception(num_classes)
    else:
        architecture = model.architecture_path(folder).read_text(encoding="utf-8")
        try:
            loaded = keras.models.model_from_json(architecture)
        except (TypeError, ValueError):
            # e.g. 'quantization_config', added by newer Keras versions
            cleaned = _drop_keys(json.loads(architecture), {"quantization_config"})
            loaded = keras.models.model_from_json(json.dumps(cleaned))
    loaded.load_weights(str(model.weights_path(folder)))
    return loaded


def preprocess(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Scale to [0, 1] and resize to ``size`` x 3, as in the notebook."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    resized = resize(array, (*size, 3), preserve_range=True, anti_aliasing=True)
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
    title = "CNN classification"
    category = "Set 8 duties"
    description = (
        "Choose one of the Set 8 models (shallow CNN, deeper VGG, complete VGG "
        "or InceptionV3 transfer learning), see its passport, then pick an "
        "image and classify it into a CIFAR-10 class."
    )
    requires_image = False  # the image is chosen from the passport window

    def __init__(
        self, model_dir: Path = MODEL_DIR, models: tuple[CnnModel, ...] = MODELS
    ) -> None:
        self.model_dir = model_dir
        self.models = models
        self._loaded: dict[str, Any] = {}  # file_tag -> Keras model

    def _get_model(self, model: CnnModel) -> Any:
        # Load once and keep it: building the networks is slow
        if model.file_tag not in self._loaded:
            self._loaded[model.file_tag] = load_keras_model(
                model, self.model_dir, len(self.class_names(model))
            )
        return self._loaded[model.file_tag]

    def class_names(self, model: CnnModel) -> tuple[str, ...]:
        return load_class_names(model.passport_path(self.model_dir))

    def classify(self, image: Image.Image, model: CnnModel) -> list[Prediction]:
        batch = preprocess(image, model.input_size)
        probabilities = self._get_model(model).predict(batch, verbose=0)[0]
        return top_predictions(probabilities, self.class_names(model))

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        # 1. Model, 2. its passport, 3. the image
        model = choose_model(
            parent, self.title, "Choose the model to classify with:", self.models
        )
        if model is None:
            return None

        try:
            details = passport_details(model.passport_path(self.model_dir))
        except (OSError, ValueError) as error:
            return ToolResult(
                message=f"Could not read the {model.title} passport: {error}"
            )
        if not ask_details_action(
            parent, self.title, f"{model.title} passport", details, ACTION_SELECT
        ):
            return None

        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose an image to classify",
            message=f"Select the image to classify with the {model.title} model "
            "(one of the CIFAR-10 classes: "
            f"{', '.join(self.class_names(model))}).",
            filetypes=OPEN_TYPES,
        )
        if not filename:
            return None
        try:
            # The picked image becomes the open image, so Undo shows it clean
            document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            return ToolResult(message=f"Could not open image: {error}")
        return self.classify_document(document, model)

    def classify_document(self, document: ImageDocument, model: CnnModel) -> ToolResult:
        assert document.current is not None

        weights = model.weights_path(self.model_dir)
        if not weights.exists():
            return ToolResult(message=f"Model weights not found: {weights}")
        try:
            predictions = self.classify(document.current, model)
        except ImportError:
            return ToolResult(
                message="TensorFlow/Keras is not installed. "
                "Run: pip install tensorflow"
            )

        best = predictions[0]
        details: dict[str, Any] = {
            "Operation": f"CNN classification ({model.title})",
            "Model": model.architecture,
            "Dataset": "CIFAR-10",
            "Input size": f"{model.input_size[0]}x{model.input_size[1]}x3",
            "Prediction": f"{best.label} ({best.probability:.1%})",
        }
        for rank, prediction in enumerate(predictions, start=1):
            details[f"Top {rank}"] = f"{prediction.label}: {prediction.probability:.1%}"

        return ToolResult(
            image=annotate(document.current, best),
            message=f"{model.title} predicted class: {best.label} "
            f"({best.probability:.1%}).",
            details=details,
        )
