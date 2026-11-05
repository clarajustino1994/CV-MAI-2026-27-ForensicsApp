from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def save_images(images, labels, target_names, folder, split, csv_writer):
    #Save face images and their corresponding labels.
    folder.mkdir(parents=True, exist_ok=True)

    for index, (image, label) in enumerate(zip(images, labels)):
        filename = f"{split}_{index:04d}.jpg"
        image_path = folder / filename

        pixels = (image * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(image_path)

        csv_writer.writerow([
            split,
            filename,
            int(label),
            target_names[label],
        ])


def visualize_eigenfaces(eigenfaces: np.ndarray, k: int) -> Image.Image:
    #Create a grid showing the first K eigenfaces.
    rows = (k + 3) // 4
    cell_width = 150
    cell_height = 190

    canvas = Image.new(
        "RGB",
        (4 * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index in range(k):
        face = eigenfaces[index]

        # Normalize each eigenface for visualization
        face = face - face.min()
        if face.max() > 0:
            face = face / face.max()

        face_image = Image.fromarray(
            (face * 255).astype(np.uint8)
        ).resize((120, 150))

        x = (index % 4) * cell_width
        y = (index // 4) * cell_height

        canvas.paste(face_image, (x + 15, y + 25))
        draw.text((x + 15, y + 5), f"Eigenface {index}", fill="black")

    return canvas


class EigenfaceBuilderTool(ForensicsTool):
    tool_id = "eigenface_builder"
    title = "Eigenface builder"
    category = "Set 6 duties"
    description = "Build a PCA face model and visualize its eigenfaces."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        messagebox.showinfo(
            "Eigenface builder",
            "The LFW face dataset will be downloaded.\n\n"
            "Choose a folder where the training images, test images "
            "and PCA model will be saved.",
            parent=parent,
        )

        folder = filedialog.askdirectory(
            parent=parent,
            title="Choose output folder",
        )

        if not folder:
            return None

        try:
            # Download the LFW dataset
            dataset = fetch_lfw_people(
                min_faces_per_person=20,
                resize=0.4,
                color=False,
            )

            images = dataset.images
            labels = dataset.target
            target_names = dataset.target_names

            # Split images into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                images,
                labels,
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )

            output = Path(folder)
            train_folder = output / "train"
            test_folder = output / "test"
            model_folder = output / "model"

            model_folder.mkdir(parents=True, exist_ok=True)

            # Save training and test images with their labels
            with open(
                output / "image_labels.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["split", "filename", "label", "person"])

                save_images(
                    X_train, y_train, target_names,
                    train_folder, "train", writer,
                )
                save_images(
                    X_test, y_test, target_names,
                    test_folder, "test", writer,
                )

            # Ask how many eigenfaces to calculate
            max_k = min(len(X_train), X_train.shape[1] * X_train.shape[2])

            k = simpledialog.askinteger(
                "Eigenface builder",
                f"Number of eigenfaces (1 to {max_k}):",
                parent=parent,
                initialvalue=min(12, max_k),
                minvalue=1,
                maxvalue=max_k,
            )

            if k is None:
                return None

            # Flatten images before applying PCA
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)

            # Calculate mean face and PCA components
            pca = PCA(
                n_components=k,
                svd_solver="randomized",
                random_state=42,
            )

            train_coefficients = pca.fit_transform(X_train_flat)
            test_coefficients = pca.transform(X_test_flat)

            eigenfaces = pca.components_.reshape(
                k, X_train.shape[1], X_train.shape[2]
            )

            mean_face = pca.mean_.reshape(
                X_train.shape[1], X_train.shape[2]
            )

            # Save the model and coefficients
            np.save(model_folder / "mean_face.npy", mean_face)
            np.save(model_folder / "eigenfaces.npy", eigenfaces)
            np.save(model_folder / "train_coefficients.npy", train_coefficients)
            np.save(model_folder / "test_coefficients.npy", test_coefficients)
            np.save(model_folder / "train_labels.npy", y_train)
            np.save(model_folder / "test_labels.npy", y_test)
            np.save(model_folder / "target_names.npy", target_names)
            np.save(
                model_folder / "explained_variance_ratio.npy",
                pca.explained_variance_ratio_,
            )

            # Visualize the first K eigenfaces
            result_image = visualize_eigenfaces(eigenfaces, k)

            return ToolResult(
                image=result_image,
                message=f"Eigenface model created successfully with {k} components.",
                details={
                    "Operation": "Eigenface builder",
                    "Dataset": "LFW People",
                    "Training images": len(X_train),
                    "Test images": len(X_test),
                    "People": len(target_names),
                    "Eigenfaces": k,
                    "Image size": f"{X_train.shape[2]}x{X_train.shape[1]}",
                    "Explained variance": (
                        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
                    ),
                    "Model folder": str(model_folder),
                },
            )

        except Exception as error:
            messagebox.showerror(
                "Eigenface builder error",
                str(error),
                parent=parent,
            )
            return None
