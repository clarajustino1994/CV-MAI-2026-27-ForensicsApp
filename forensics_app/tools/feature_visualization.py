"""Set 8: project ResNet50 features of the CIFAR-10 test set with t-SNE and UMAP.

The features are extracted once, saved to an ``.npz`` file and loaded from it on
later runs, so the projections can be recomputed without running ResNet50 again.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from forensics_app.core import ImageDocument  # noqa: E402

from .base import ForensicsTool, ToolResult  # noqa: E402
from .cnn_classification import MODEL_DIR, load_class_names  # noqa: E402
from .dialogs import ProgressDialog, ask_choice  # noqa: E402

FEATURES_PATH = MODEL_DIR / "test_features_resnet50.npz"
PLOT_PATH = MODEL_DIR / "test_features_tsne_umap.png"

SAMPLES_PER_CLASS = 100
BATCH_SIZE = 32
RESNET_INPUT = (224, 224)
RANDOM_STATE = 42

ACTION_LOAD = "Load saved features"
ACTION_EXTRACT = "Extract features again"

Progress = Callable[[int, int, str], None]


def stratified_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    """First ``per_class`` indices of every class, grouped by class."""
    return np.concatenate(
        [np.flatnonzero(labels == cls)[:per_class] for cls in np.unique(labels)]
    )


def extract_test_features(
    per_class: int = SAMPLES_PER_CLASS,
    progress: Progress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ResNet50 ``avg_pool`` features of a stratified CIFAR-10 test subset.

    Unlike the notebook, the images go through ``preprocess_input`` (0-255,
    BGR, ImageNet mean subtracted), which is what the ImageNet weights expect.
    """
    import keras  # imported lazily: TensorFlow takes seconds to load
    import tensorflow as tf
    from keras.applications.resnet50 import ResNet50, preprocess_input

    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_test = y_test.ravel()
    indices = stratified_indices(y_test, per_class)
    images, labels = x_test[indices], y_test[indices]

    extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    features = []
    total = len(images)
    for start in range(0, total, BATCH_SIZE):
        if progress is not None:
            progress(start, total, f"Extracting features: {start}/{total} images")
        batch = tf.image.resize(images[start : start + BATCH_SIZE], RESNET_INPUT)
        batch = preprocess_input(batch.numpy())
        features.append(extractor.predict(batch, verbose=0))
    if progress is not None:
        progress(total, total, "Features extracted")

    return np.concatenate(features).astype(np.float32), labels.astype(np.int64)


def save_features(
    path: Path, features: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, features=features, labels=labels, class_names=np.array(class_names)
    )


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path) as data:
        return (
            data["features"],
            data["labels"],
            tuple(str(name) for name in data["class_names"]),
        )


def project_tsne(features: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(30.0, (len(features) - 1) / 3)
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=RANDOM_STATE
    ).fit_transform(features)


def project_umap(features: np.ndarray) -> np.ndarray:
    import umap  # imported lazily: numba compiles on first import

    return umap.UMAP(
        n_neighbors=min(15, len(features) - 1),
        min_dist=0.1,
        n_components=2,
        random_state=RANDOM_STATE,
    ).fit_transform(features)


def plot_projections(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: tuple[str, ...],
) -> Image.Image:
    """Scatter each 2D projection side by side, one colour per class."""
    figure, axes = plt.subplots(
        1, len(projections), figsize=(7 * len(projections), 6.5), squeeze=False
    )
    colours = plt.get_cmap("tab10")
    for ax, (name, points) in zip(axes[0], projections.items()):
        for cls, class_name in enumerate(class_names):
            mask = labels == cls
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=12,
                alpha=0.75,
                color=colours(cls % 10),
                label=class_name,
            )
        ax.set_title(f"{name} of ResNet50 test features")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class FeatureVisualizationTool(ForensicsTool):
    tool_id = "feature_visualization"
    title = "Feature visualization (t-SNE / UMAP)"
    category = "Set 8 duties"
    description = (
        "Extract ResNet50 features of the CIFAR-10 test set, save them, load them "
        "and project them to 2D with t-SNE and UMAP."
    )
    requires_image = False

    def __init__(
        self, features_path: Path = FEATURES_PATH, plot_path: Path = PLOT_PATH
    ) -> None:
        self.features_path = features_path
        self.plot_path = plot_path

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = [ACTION_EXTRACT]
        if self.features_path.exists():
            options.insert(0, ACTION_LOAD)
            prompt = f"Saved features found in {self.features_path.name}."
        else:
            prompt = "No saved features yet: they will be extracted (about 1 minute)."
        choice = ask_choice(parent, self.title, prompt, options)
        if choice is None:
            return None

        try:
            with ProgressDialog(parent, self.title) as progress:
                if options[choice] == ACTION_EXTRACT:
                    features, labels = extract_test_features(progress=progress)
                    save_features(
                        self.features_path, features, labels, load_class_names()
                    )
                return self.visualize(progress)
        except ImportError as error:
            return ToolResult(
                message=f"Missing dependency ({error.name}). "
                "Run: pip install tensorflow umap-learn"
            )

    def visualize(self, progress: Progress | None = None) -> ToolResult:
        """Load the saved features, project them and plot both projections."""

        def step(n: int, text: str) -> None:
            if progress is not None:
                progress(n, 3, text)

        step(0, "Loading saved features…")
        features, labels, class_names = load_features(self.features_path)
        step(1, "Computing t-SNE…")
        projections = {"t-SNE": project_tsne(features)}
        step(2, "Computing UMAP…")
        projections["UMAP"] = project_umap(features)
        step(3, "Plotting…")

        image = plot_projections(projections, labels, class_names)
        image.save(self.plot_path)

        details: dict[str, Any] = {
            "Operation": "Feature visualization (t-SNE / UMAP)",
            "Model": "ResNet50 (ImageNet), avg_pool layer",
            "Dataset": "CIFAR-10 test set",
            "Images": f"{len(labels)} ({len(labels) // len(class_names)} per class)",
            "Feature size": features.shape[1],
            "Features file": str(self.features_path),
            "Plot file": str(self.plot_path),
        }
        return ToolResult(
            image=image,
            message=f"Projected {len(labels)} test feature vectors with t-SNE "
            "and UMAP.",
            details=details,
        )
