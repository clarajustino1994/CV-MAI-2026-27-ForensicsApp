from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .eigenface_builder import (
    build_eigenface_model,
    choose_output_folder,
    find_existing_model_root,
)

N_MATCHES = 3

MODEL_SAVED = 0
MODEL_NEW = 1

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

REQUIRED_MODEL_FILES = (
    "mean_face.npy",
    "eigenfaces.npy",
    "train_coefficients.npy",
    "train_labels.npy",
    "target_names.npy",
    "explained_variance.npy",
)


def get_surname(name: str) -> str:
    """Return only the surname of a person."""
    return str(name).rsplit(" ", 1)[-1]


def load_model(output: Path) -> dict:
    """Load mean, eigenfaces and training coefficients w_i = U_K^T (x_i - mu)."""
    model_folder = output / "model"

    missing = [
        name for name in REQUIRED_MODEL_FILES if not (model_folder / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The Eigenface model is incomplete or was built with an older "
            "version of the app. Missing files:\n"
            + "\n".join(missing)
            + "\n\nBuild a new model and try again."
        )

    mean_face = np.load(model_folder / "mean_face.npy")
    eigenfaces = np.load(model_folder / "eigenfaces.npy")
    explained_variance = np.load(model_folder / "explained_variance.npy")

    # The builder saved whitened coefficients (PCA with whiten=True).
    # Undo the whitening so they match w_i = U_K^T (x_i - mu).
    train_coefficients = np.load(model_folder / "train_coefficients.npy") * np.sqrt(
        explained_variance
    )

    return {
        "output": output,
        "mean_face": mean_face,
        "eigenfaces": eigenfaces,
        "train_coefficients": train_coefficients,
        "train_labels": np.load(model_folder / "train_labels.npy"),
        "target_names": np.load(
            model_folder / "target_names.npy",
            allow_pickle=True,
        ),
    }


def project_face(image: Image.Image, model: dict) -> np.ndarray:
    """Compute w_test = U_K^T (x_test - mu) for one face image."""
    h, w = model["mean_face"].shape

    face = image.convert("L").resize((w, h), Image.Resampling.LANCZOS)
    x = np.asarray(face, dtype=np.float64).ravel() / 255.0

    components = model["eigenfaces"].reshape(len(model["eigenfaces"]), -1)
    return components @ (x - model["mean_face"].ravel())


def closest_matches(
    w_test: np.ndarray,
    train_coefficients: np.ndarray,
    n: int = N_MATCHES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and distances d_i = ||w_test - w_i||_2 of the n closest."""
    distances = np.linalg.norm(train_coefficients - w_test, axis=1)
    order = np.argsort(distances)[:n]
    return order, distances[order]


def lookup_true_name(output: Path, image_path: Path) -> str | None:
    """Return the person's name if the image belongs to the saved dataset."""
    labels_file = output / "image_labels.csv"
    if not labels_file.exists():
        return None

    split = image_path.parent.name
    if image_path.parent.resolve() != (output / split).resolve():
        return None

    with open(labels_file, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == split and row["filename"] == image_path.name:
                return row["person"]

    return None


def create_gallery(
    images: list[Image.Image],
    titles: list[str],
    columns: int = 4,
) -> Image.Image:
    """Display face images in a grid, each with a title above it."""

    cell_width = 190
    cell_height = 235

    image_width = 140
    image_height = 189

    rows = (len(images) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            16,
        )
    except OSError:
        font = ImageFont.load_default()

    for index, (image, title) in enumerate(zip(images, titles)):

        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        text_box = draw.multiline_textbbox(
            (0, 0),
            title,
            font=font,
            align="center",
            spacing=2,
        )

        text_width = text_box[2] - text_box[0]
        text_x = x + (cell_width - text_width) // 2 - text_box[0]

        draw.multiline_text(
            (text_x, y + 5),
            title,
            font=font,
            fill="black",
            align="center",
            spacing=2,
        )

        # Resize all faces to the same dimensions.
        image = (
            image.convert("L")
            .convert("RGB")
            .resize(
                (image_width, image_height),
                Image.Resampling.LANCZOS,
            )
        )

        image_x = x + (cell_width - image_width) // 2
        image_y = y + 40

        canvas.paste(image, (image_x, image_y))

        # Add a thin black border.
        draw.rectangle(
            (
                image_x,
                image_y,
                image_x + image_width - 1,
                image_y + image_height - 1,
            ),
            outline="black",
            width=1,
        )

    return canvas


class FaceRecognitionTool(ForensicsTool):

    tool_id = "face_recognition"
    title = "Face recognition"
    category = "Set 6 duties"
    description = (
        "Recognize a face with k-NN: find the closest training faces "
        "in Eigenface (PCA) space."
    )
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        try:
            output = self._choose_model(parent)
            if output is None:
                return None

            model = load_model(output)

            image_path = self._choose_image(parent, output)
            if image_path is None:
                return None

            with Image.open(image_path) as image_file:
                query = image_file.convert("L").copy()

            return self._recognize(query, image_path, model)

        except Exception as error:  # show any failure (download, files, PCA)
            messagebox.showerror(
                "Face recognition error",
                str(error),
                parent=parent,
            )
            return None

    @staticmethod
    def _choose_model(parent: tk.Misc) -> Path | None:
        """Ask whether to reuse a saved model or build a new one."""
        choice = ask_choice(
            parent,
            "Face recognition",
            "Which Eigenface model do you want to use?",
            ["Use saved model", "Build new model"],
        )

        if choice == MODEL_SAVED:
            folder = filedialog.askdirectory(
                parent=parent,
                title="Face recognition",
                message=(
                    "Next, select the Eigenface model folder: the folder you "
                    "chose in the Eigenface builder (it contains model/, "
                    "train/ and test/)."
                ),
            )
            if not folder:
                return None

            root = find_existing_model_root(Path(folder))
            if root is None:
                raise FileNotFoundError(
                    f"No Eigenface model found in:\n{folder}\n\n"
                    "Select the folder created by the Eigenface builder."
                )
            return root

        if choice == MODEL_NEW:
            output = choose_output_folder(parent)
            if output is None:
                return None

            built = build_eigenface_model(parent, output)
            return None if built is None else output

        return None

    @staticmethod
    def _choose_image(parent: tk.Misc, output: Path) -> Path | None:
        """Ask for the face image to recognize, starting in the test folder."""
        test_folder = output / "test"

        filename = filedialog.askopenfilename(
            parent=parent,
            title="Face recognition",
            message=(
                "Next, select the face image to recognize "
                "(for example one from the test/ folder)."
            ),
            initialdir=str(test_folder if test_folder.is_dir() else output),
            filetypes=OPEN_TYPES,
        )

        return Path(filename) if filename else None

    @staticmethod
    def _recognize(
        query: Image.Image,
        image_path: Path,
        model: dict,
    ) -> ToolResult:
        output = model["output"]
        target_names = model["target_names"]

        w_test = project_face(query, model)
        indices, distances = closest_matches(
            w_test,
            model["train_coefficients"],
        )

        match_names = [
            str(target_names[int(model["train_labels"][i])]) for i in indices
        ]
        predicted_name = match_names[0]
        true_name = lookup_true_name(output, image_path)

        query_title = "query"
        if true_name is not None:
            query_title += f"\ntrue: {get_surname(true_name)}"

        images = [query]
        titles = [query_title]

        for rank, (index, name, distance) in enumerate(
            zip(indices, match_names, distances),
            start=1,
        ):
            train_path = output / "train" / f"train_{index:04d}.jpg"
            with Image.open(train_path) as image_file:
                images.append(image_file.convert("L").copy())
            titles.append(f"#{rank} {get_surname(name)}\nd = {distance:.2f}")

        result_image = create_gallery(images, titles)

        message = f"Predicted identity: {predicted_name}."
        if true_name is not None:
            verdict = "correct" if true_name == predicted_name else "wrong"
            message += f" True identity: {true_name} ({verdict})."

        details = {
            "Operation": "Eigenfaces face recognition",
            "Method": "Eigenfaces (PCA) + k-NN (k = 1, L2 distance)",
            "Query image": str(image_path),
            "Predicted identity": predicted_name,
            "True identity": true_name or "Unknown",
            "Eigenfaces": len(model["eigenfaces"]),
            "Training images": len(model["train_labels"]),
            "Model folder": str(output),
        }
        for rank, (name, distance) in enumerate(
            zip(match_names, distances),
            start=1,
        ):
            details[f"Match {rank}"] = f"{name} (d = {distance:.2f})"

        return ToolResult(image=result_image, message=message, details=details)
