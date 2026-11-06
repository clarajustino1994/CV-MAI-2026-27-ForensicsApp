from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def get_surname(name: str) -> str:
    """Return only the surname of a person."""
    return str(name).rsplit(" ", 1)[-1]


def create_gallery(
    images: list[Image.Image],
    predicted_names: list[str],
    true_names: list[str],
    columns: int = 4,
) -> Image.Image:
    """Display test images with their predicted and true identities."""

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

    for index, image in enumerate(images):

        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        # Display the prediction and the real identity.
        title = (
            f"predicted: {predicted_names[index]}\n"
            f"true:      {true_names[index]}"
        )

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
        image = image.convert("L").convert("RGB").resize(
            (image_width, image_height),
            Image.Resampling.LANCZOS,
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
    description = "Recognize faces using Eigenfaces, PCA and SVM."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        try:
            # Select the Eigenface results folder.
            folder = filedialog.askdirectory(
                parent=parent,
                title="Select Eigenface results folder",
            )

            if not folder:
                return None

            output = Path(folder)

            model_folder = output / "model"
            test_folder = output / "test"

            # Load the saved PCA coefficients and labels.
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

            # Search for the best SVM parameters,
            # following the professor's notebook.
            param_grid = {
                "C": [1e3, 5e3, 1e4, 5e4, 1e5],
                "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
            }

            classifier = GridSearchCV(
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                ),
                param_grid,
                cv=5,
                verbose=1,
            )

            # Train the classifier using the training set.
            classifier.fit(
                train_coefficients,
                train_labels,
            )

            # Predict the identities of the test images.
            predictions = classifier.predict(test_coefficients)

            # Calculate accuracy using the complete test set.
            accuracy = accuracy_score(
                test_labels,
                predictions,
            )

            correct = int(np.sum(predictions == test_labels))
            incorrect = len(test_labels) - correct

            # Generate the classification report.
            report = classification_report(
                test_labels,
                predictions,
                labels=np.arange(len(target_names)),
                target_names=target_names,
                zero_division=0,
            )

            # Display the first 12 test images.
            number_of_images = min(12, len(test_labels))

            images = []
            predicted_names = []
            true_names = []

            for index in range(number_of_images):

                # Prediction and true label belong to the same test image.
                predicted_name = get_surname(
                    target_names[int(predictions[index])]
                )

                true_name = get_surname(
                    target_names[int(test_labels[index])]
                )

                # Load the corresponding test image.
                image_path = test_folder / f"test_{index:04d}.jpg"

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Test image not found: {image_path}"
                    )

                with Image.open(image_path) as image_file:
                    image = image_file.convert("L").copy()

                images.append(image)
                predicted_names.append(predicted_name)
                true_names.append(true_name)

            # Create the gallery with 12 test images.
            result_image = create_gallery(
                images,
                predicted_names,
                true_names,
            )

            # Save the results as a PNG image.
            result_image.save(
                output / "face_recognition_results.png"
            )

            return ToolResult(
                image=result_image,
                message=(
                    f"Face recognition completed. "
                    f"{correct}/{len(test_labels)} correct predictions "
                    f"({accuracy * 100:.2f}% accuracy)."
                ),
                details={
                    "Operation": "Eigenfaces face recognition",
                    "Method": "PCA + SVM (RBF)",
                    "Test images displayed": number_of_images,
                    "Total test images": len(test_labels),
                    "Training images": len(train_labels),
                    "Correct predictions": correct,
                    "Incorrect predictions": incorrect,
                    "Accuracy": f"{accuracy * 100:.2f}%",
                    "Best parameters": str(classifier.best_params_),
                    "Classification report": report,
                    "Saved image": str(
                        output / "face_recognition_results.png"
                    ),
                },
            )

        except Exception as error:

            messagebox.showerror(
                "Face recognition error",
                str(error),
                parent=parent,
            )

            return None
