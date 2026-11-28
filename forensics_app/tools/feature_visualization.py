"""Set 8: project the test-set features of a Set 8 model with t-SNE and UMAP.

The user picks the model (shallow CNN, deeper VGG, complete VGG or InceptionV3
transfer learning), sees the passport of its saved features, and then the
features are projected to 2D. The features come from the ``.npz`` files written
by the feature extraction cells of the Set 8 notebook.
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
from .cnn_classification import (  # noqa: E402
    MODEL_DIR,
    MODELS,
    CnnModel,
    choose_model,
    passport_details,
)
from .dialogs import ProgressDialog, ask_details_action  # noqa: E402

RANDOM_STATE = 42

ACTION_PROJECT = "Generate t-SNE / UMAP"

Progress = Callable[[int, int, str], None]


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path) as data:
        return (
            data["features"],
            data["labels"],
            tuple(str(name) for name in data["class_names"]),
        )


def plot_path(model: CnnModel, folder: Path) -> Path:
    return folder / f"test_features_{model.features_tag}_tsne_umap.png"


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
    model_title: str,
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
        ax.set_title(f"{name} of {model_title} test features")
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
        "Choose one of the Set 8 models, see the passport of its saved CIFAR-10 "
        "test features, and project them to 2D with t-SNE and UMAP."
    )
    requires_image = False

    def __init__(
        self, model_dir: Path = MODEL_DIR, models: tuple[CnnModel, ...] = MODELS
    ) -> None:
        self.model_dir = model_dir
        self.models = models

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        # 1. Model, 2. passport of its features, 3. the projections
        model = choose_model(
            parent,
            self.title,
            "Choose the model whose features to project:",
            self.models,
        )
        if model is None:
            return None

        features_path = model.features_path(self.model_dir)
        if not features_path.exists():
            return ToolResult(
                message=f"No saved features for {model.title}: "
                f"{features_path.name} not found. Run its feature extraction "
                "cell in the Set 8 notebook first."
            )
        details = self.features_details(model)
        if not ask_details_action(
            parent,
            self.title,
            f"{model.title} features passport",
            details,
            ACTION_PROJECT,
        ):
            return None

        try:
            with ProgressDialog(parent, self.title) as progress:
                return self.visualize(model, progress)
        except ImportError as error:
            return ToolResult(
                message=f"Missing dependency ({error.name}). "
                "Run: pip install scikit-learn umap-learn"
            )

    def features_details(self, model: CnnModel) -> dict[str, str]:
        """Passport of the saved features, or of the model if it has none."""
        for path in (
            model.features_passport_path(self.model_dir),
            model.passport_path(self.model_dir),
        ):
            try:
                details = passport_details(path)
            except (OSError, ValueError):
                continue
            details.pop("Arrays", None)  # array dtypes: noise for the user
            details["Features file"] = model.features_path(self.model_dir).name
            return details
        return {"Features file": model.features_path(self.model_dir).name}

    def visualize(
        self, model: CnnModel, progress: Progress | None = None
    ) -> ToolResult:
        """Load the saved features, project them and plot both projections."""

        def step(n: int, text: str) -> None:
            if progress is not None:
                progress(n, 3, text)

        features_path = model.features_path(self.model_dir)
        step(0, "Loading saved features…")
        features, labels, class_names = load_features(features_path)
        step(1, "Computing t-SNE…")
        projections = {"t-SNE": project_tsne(features)}
        step(2, "Computing UMAP…")
        projections["UMAP"] = project_umap(features)
        step(3, "Plotting…")

        image = plot_projections(projections, labels, class_names, model.title)
        output = plot_path(model, self.model_dir)
        image.save(output)

        details: dict[str, Any] = {
            "Operation": "Feature visualization (t-SNE / UMAP)",
            "Model": f"{model.title}: {model.architecture}",
            "Dataset": "CIFAR-10 test set",
            "Images": f"{len(labels)} ({len(labels) // len(class_names)} per class)",
            "Feature size": features.shape[1],
            "Features file": str(features_path),
            "Plot file": str(output),
        }
        return ToolResult(
            image=image,
            message=f"Projected {len(labels)} {model.title} test feature vectors "
            "with t-SNE and UMAP.",
            details=details,
        )
