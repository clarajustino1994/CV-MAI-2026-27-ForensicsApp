from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def get_surname(name: str) -> str:
    """Return only the surname of a person."""
    return str(name).split()[-1]


def create_gallery(
    images: list[Image.Image],
    titles: list[str],
    columns: int = 4,
) -> Image.Image:
    """Display the test image and its three closest training images."""

    cell_width = 180
    cell_height = 220
    rows = 1

    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, image in enumerate(images[:columns]):

        x = index * cell_width
        y = 0

        image = image.convert("L").convert("RGB").resize((120, 150))

        # Display face image
        canvas.paste(image, (x + 30, y + 55))

        # Add a thin black border
        draw.rectangle(
            (x + 30, y + 55, x + 149, y + 204),
            outline="black",
            width=1,
        )

        # Center the title
        text_box = draw.multiline_textbbox(
            (0, 0),
            titles[index],
            align="center",
        )

        text_width = text_box[2] - text_box[0]

        text_x = (
            x + (cell_width - text_width) // 2 - text_box[0]
        )

        draw.multiline_text(
            (text_x, y + 5),
            titles[index],
            fill="black",
            align="center",
        )

    return canvas


class FaceRecognitionTool(ForensicsTool):

    tool_id = "face_recognition"
    title = "Face recognition"
    category = "Set 6 duties"
    description = "Recognize faces using Eigenfaces, PCA and Euclidean distance."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        messagebox.showinfo(
            "Face recognition",
            "Select one test image from the test folder.",
            parent=parent,
        )

        image_path = filedialog.askopenfilename(
            parent=parent,
            title="Select test image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png"),
            ],
        )

        if not image_path:
            return None

        try:
            image_path = Path(image_path)
            output = image_path.parent.parent

            model_folder = output / "model"
            train_folder = output / "train"

            # Get the test image index from its filename
            match = re.search(
                r"test_(\d+)",
                image_path.stem,
            )

            if not match:
                raise ValueError(
                    "The selected image must have a name such as test_0000.jpg."
                )

            test_index = int(match.group(1))

            # Load saved PCA coefficients and labels
            train_coefficients = np.load(
                model_folder / "train_coefficients.npy"
            )

            test_coefficients = np.load(
                model_folder / "test_coefficients.npy"
            )

            train_labels = np.load(
                model_folder / "train_labels.npy"
            )

            test_labels = np.load(
                model_folder / "test_labels.npy"
            )

            target_names = np.load(
                model_folder / "target_names.npy",
                allow_pickle=True,
            )

            # Check the selected test image index
            if test_index >= len(test_coefficients):
                raise ValueError(
                    "The selected test image does not exist in the saved PCA coefficients."
                )

            # Get the PCA coefficients of the test image
            test_vector = test_coefficients[test_index]

            # Calculate Euclidean distances to all training images
            distances = np.linalg.norm(
                train_coefficients - test_vector,
                axis=1,
            )

            # Find the three closest training images
            nearest_indices = np.argsort(distances)[:3]

            # Get the real identity of the test image
            true_name = get_surname(
                target_names[int(test_labels[test_index])]
            )

            # The closest neighbour gives the predicted identity
            predicted_name = get_surname(
                target_names[int(train_labels[nearest_indices[0]])]
            )

            # Load the test image
            images = [
                Image.open(image_path).convert("L")
            ]

            titles = [
                f"TEST\npredicted: {predicted_name}\ntrue: {true_name}"
            ]

            # Load the three closest training images
            for position, train_index in enumerate(nearest_indices):

                train_image_path = (
                    train_folder / f"train_{train_index:04d}.jpg"
                )

                train_image = Image.open(
                    train_image_path
                ).convert("L")

                images.append(train_image)

                train_name = get_surname(
                    target_names[int(train_labels[train_index])]
                )

                distance = distances[train_index]

                titles.append(
                    f"Nearest {position + 1}\n"
                    f"name: {train_name}\n"
                    f"distance: {distance:.2f}"
                )

            # Create the gallery
            result_image = create_gallery(
                images,
                titles,
            )

            return ToolResult(
                image=result_image,
                message=(
                    f"Face recognition completed. "
                    f"Predicted: {predicted_name}. "
                    f"True: {true_name}."
                ),
                details={
                    "Operation": "Eigenfaces face recognition",
                    "Method": "PCA + Euclidean distance",
                    "Test image": image_path.name,
                    "Predicted identity": predicted_name,
                    "True identity": true_name,
                    "Closest training images": 3,
                    "Distances": [
                        f"{distances[i]:.2f}"
                        for i in nearest_indices
                    ],
                },
            )

        except Exception as error:

            messagebox.showerror(
                "Face recognition error",
                str(error),
                parent=parent,
            )

            return None
