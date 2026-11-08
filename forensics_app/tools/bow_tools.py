"""Set 7 tools: codebook builder (7.1), word histogram (7.2), classifier (7.3)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from PIL import Image

from forensics_app.core import ImageDocument

from . import filedialog
from .bag_of_words import (
    CODEBOOK_FILE,
    MODEL_FILE,
    SPLIT_FILE,
    WORDS_PER_CLASS,
    BowModel,
    ImageWords,
    build_model,
    classify,
    codebook_rows,
    compute_histogram,
    draw_codebook,
    draw_histogram,
    draw_prediction,
    load_dataset,
    load_model,
    save_model,
    true_class,
)
from .base import ForensicsTool, ToolResult, mark_scrollable
from .dialogs import (
    ProgressDialog,
    ask_choice,
    ask_image_action,
)
from .feature_matching import ask_descriptor_params

CATEGORY = "Set 7 duties"
USE_LOADED = 0

# Codebooks taller than this many rows of words scroll in the image view
MAX_CODEBOOK_ROWS = 6

# Most frequent words listed in the details panel
TOP_WORDS = 5


class BowSession:
    """The Bag of Words model shared by the three Set 7 tools."""

    def __init__(self) -> None:
        self.model: BowModel | None = None

    def get_model(self, parent: tk.Misc, title: str) -> BowModel | None:
        """Reuse the loaded model or ask for a saved one; None on cancel."""
        if self.model is not None:
            choice = ask_choice(
                parent,
                title,
                f"Model loaded from:\n{self.model.folder}",
                ["Use this model", "Load another model"],
            )
            if choice is None:
                return None
            if choice == USE_LOADED:
                return self.model

        folder = filedialog.askdirectory(
            parent=parent,
            title=title,
            message=(
                "Next, select the Bag of Words MODEL folder: the one you chose "
                "in the Codebook builder (it contains bow_model.joblib)."
            ),
        )
        if not folder:
            return None
        self.model = load_model(Path(folder))
        return self.model


def model_details(model: BowModel) -> dict[str, object]:
    return {
        "Model folder": str(model.folder),
        "Descriptor": model.params.kind,
        "Visual words (K)": model.n_words,
        "Classes": ", ".join(model.class_names),
        "Test accuracy": f"{model.test_accuracy * 100:.1f}%",
    }


def query_image(document: ImageDocument) -> Image.Image:
    # The original, so running 7.2 (which replaces the view with a plot)
    # before 7.3 does not make 7.3 describe the plot instead of the photo
    assert document.original is not None
    return document.original


def split_label(model: BowModel, path: Path | None) -> str:
    split = model.split_of(path)
    if split == "train":
        return "train (seen while training: not a fair test)"
    return split


def histogram_details(words: ImageWords) -> dict[str, object]:
    order = words.histogram.argsort()[::-1][:TOP_WORDS]
    return {
        "Keypoints": len(words.words),
        "Words used": int((words.histogram > 0).sum()),
        "Top words": ", ".join(
            f"vw {w} ({words.histogram[w]:.3f})" for w in order if words.histogram[w]
        ),
    }


class CodebookBuilderTool(ForensicsTool):
    tool_id = "bow_codebook"
    title = "Codebook builder"
    category = CATEGORY
    description = (
        "Cluster the descriptors of a dataset into K visual words, show them, "
        "and train the classifier."
    )
    requires_image = False

    def __init__(self, session: BowSession | None = None) -> None:
        self.session = session or BowSession()

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        title = self.title
        source = filedialog.askdirectory(
            parent=parent,
            title=title,
            message=(
                "Step 1 of 2: select the DATASET folder.\n\n"
                "It must contain one sub-folder of images per class "
                "(e.g. Caltech_101_subset with butterfly/, dalmatian/, kangaroo/)."
            ),
        )
        if not source:
            return None
        dataset = load_dataset(Path(source))
        n_classes = len(dataset.class_names)

        params = ask_descriptor_params(parent, title)
        if params is None:
            return None

        n_words = simpledialog.askinteger(
            title,
            f"Number of visual words K ({WORDS_PER_CLASS} × {n_classes} classes):",
            parent=parent,
            initialvalue=WORDS_PER_CLASS * n_classes,
            minvalue=2,
            maxvalue=1000,
        )
        if n_words is None:
            return None

        output = filedialog.askdirectory(
            parent=parent,
            title=title,
            message=(
                "Step 2 of 2: select the folder where the MODEL will be saved.\n\n"
                "bow_model.joblib, codebook.png and split.csv (the list of "
                "test images) will be written there. Use an empty folder."
            ),
        )
        if not output:
            return None
        output_dir = Path(output)
        if (output_dir / MODEL_FILE).exists() and not messagebox.askyesno(
            title,
            f"{output_dir} already contains a model.\nDo you want to replace it?",
            parent=parent,
        ):
            return None

        with ProgressDialog(parent, title) as progress:
            model, train_features = build_model(dataset, params, n_words, progress)

        save_model(model, output_dir, dataset)
        image = draw_codebook(model, train_features)
        image.save(output_dir / CODEBOOK_FILE)
        if codebook_rows(n_words) > MAX_CODEBOOK_ROWS:
            # Shrunk to fit the view the words become unreadable, so the
            # viewer fits the width instead and scrolls down
            mark_scrollable(image)
        self.session.model = model

        details: dict[str, object] = {
            "Operation": "Codebook construction (K-Means)",
            "Dataset": source,
            "Training images": len(model.train_paths),
            "Test images": len(model.test_paths),
            "Training descriptors": model.n_descriptors,
            **model_details(model),
            "Classifier": "SVM (RBF kernel) on word histograms",
            "Test image list": str(output_dir / SPLIT_FILE),
        }
        return ToolResult(
            image=image,
            message=(
                f"Codebook with {n_words} visual words built; classifier test "
                f"accuracy {model.test_accuracy * 100:.1f}%."
            ),
            details=details,
        )


class BowHistogramTool(ForensicsTool):
    tool_id = "bow_histogram"
    title = "Visual word histogram"
    category = CATEGORY
    description = (
        "Assign every keypoint of the image to its closest visual word and "
        "plot the word frequencies."
    )

    def __init__(self, session: BowSession | None = None) -> None:
        self.session = session or BowSession()

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        model = self.session.get_model(parent, self.title)
        if model is None:
            return None

        words = compute_histogram(query_image(document), model)
        if len(words.words) == 0:
            return ToolResult(message="No keypoints found: the histogram is empty.")

        name = document.path.name if document.path else "query image"
        return ToolResult(
            image=draw_histogram(words, model, title=name),
            message=f"Histogram of {model.n_words} visual words for {name}.",
            details={
                "Operation": "Histogram of visual words",
                "Image split": split_label(model, document.path),
                **histogram_details(words),
                **model_details(model),
            },
        )


class BowClassifierTool(ForensicsTool):
    tool_id = "bow_classifier"
    title = "Bag of Words classifier"
    category = CATEGORY
    description = (
        "Show the visual word histogram of the image and, after clicking "
        "Classify, the most probable class."
    )

    def __init__(self, session: BowSession | None = None) -> None:
        self.session = session or BowSession()

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        model = self.session.get_model(parent, self.title)
        if model is None:
            return None

        image = query_image(document)
        words = compute_histogram(image, model)
        if len(words.words) == 0:
            return ToolResult(
                message="No keypoints found: the image cannot be classified."
            )

        name = document.path.name if document.path else "query image"
        histogram_image = draw_histogram(words, model, title=name)
        if not ask_image_action(parent, self.title, histogram_image, "Classify"):
            return None

        ranked = classify(words.histogram, model)
        label, probability = ranked[0]
        truth = true_class(document.path, model)

        message = f"p({label}) = {probability:.2f}"
        if truth is not None:
            message += (
                f" · true class: {truth} ({'correct' if truth == label else 'wrong'})"
            )

        details: dict[str, object] = {
            "Operation": "Bag of Words classification",
            "Predicted class": label,
            "Top probability": f"{probability:.2f}",
            "True class": truth or "Unknown",
            "Image split": split_label(model, document.path),
        }
        for class_name, p in ranked:
            details[f"p({class_name})"] = f"{p:.3f}"
        details.update(histogram_details(words))
        details.update(model_details(model))

        return ToolResult(
            image=draw_prediction(image, label, probability),
            message=message,
            details=details,
        )
