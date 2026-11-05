from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def create_gallery(
    test_image: Image.Image,
    train_images: list[Image.Image],
    names: list[str],
    distances: list[float],
) -> Image.Image:
    #Display the test image and its three closest training images.

    images = [test_image] + train_images
    titles = ["Test image"] + [
        f"{name}\nDistance: {distance:.3f}"
        for name, distance in zip(names, distances)
    ]

    cell_width = 160
    cell_height = 210

    canvas = Image.new("RGB", (cell_width * len(images), cell_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (image, title) in enumerate(zip(images, titles)):
        image = image.convert("L").convert("RGB").resize((120, 150))

        x = index * cell_width

        canvas.paste(image, (x + 20, 35))
        draw.text((x + 10, 10), title, fill="black")

    return canvas


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition"
    category = "Set 6 duties"
    description = "Find the closest training faces using Eigenfaces."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        messagebox.showinfo(
            "Face recognition",
            "Select the output folder created by the Eigenface builder.",
            parent=parent,
        )

        folder = filedialog.askdirectory(
            parent=parent,
            title="Select Eigenface dataset folder",
        )

        if not folder:
            return None

        output = Path(folder)
        model_folder = output / "model"
        test_folder = output / "test"

        try:
            # Load the saved PCA model and training data
            mean_face = np.load(model_folder / "mean_face.npy")
            eigenfaces = np.load(model_folder / "eigenfaces.npy")
            train_coefficients = np.load(
                model_folder / "train_coefficients.npy"
            )
            train_labels = np.load(model_folder / "train_labels.npy")
            target_names = np.load(
                model_folder / "target_names.npy",
                allow_pickle=True,
            )

            # Select one image from the test folder
            filename = filedialog.askopenfilename(
                parent=parent,
                title="Select a test face",
                initialdir=str(test_folder),
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png"),
                ],
            )

            if not filename:
                return None

            test_image = Image.open(filename).convert("L")

            # Resize the test image to the model dimensions
            height, width = mean_face.shape
            test_image = test_image.resize((width, height))

            # Convert pixels to a normalized vector
            test_array = np.asarray(test_image, dtype=np.float64) / 255.0
            test_vector = test_array.reshape(-1)

            # Project the test image into the Eigenfaces space
            components = eigenfaces.reshape(len(eigenfaces), -1)
            test_coefficients = components @ (
                test_vector - mean_face.reshape(-1)
            )

            # Calculate Euclidean distances to training images
            distances = np.linalg.norm(
                train_coefficients - test_coefficients,
                axis=1,
            )

            # Find the three closest training images
            nearest_indices = np.argsort(distances)[:3]

            train_images = []
            names = []
            nearest_distances = []

            for index in nearest_indices:
                image_path = (
                    output / "train" / f"train_{index:04d}.jpg"
                )

                train_images.append(Image.open(image_path).convert("L"))

                label = int(train_labels[index])
                names.append(str(target_names[label]))
                nearest_distances.append(float(distances[index]))

            # Create the result gallery
            result_image = create_gallery(
                test_image,
                train_images,
                names,
                nearest_distances,
            )

            return ToolResult(
                image=result_image,
                message="Face recognition completed successfully.",
                details={
                    "Operation": "Face recognition with Eigenfaces",
                    "Test image": Path(filename).name,
                    "Closest person": names[0],
                    "Closest distance": f"{nearest_distances[0]:.4f}",
                    "Images compared": len(train_coefficients),
                    "Neighbors displayed": len(nearest_indices),
                },
            )

        except Exception as error:
            messagebox.showerror(
                "Face recognition error",
                str(error),
                parent=parent,
            )
            return None
