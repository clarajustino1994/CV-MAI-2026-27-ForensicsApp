from __future__ import annotations

import csv
import ssl
import shutil
import tkinter as tk
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


MAX_EIGENFACES_SHOWN = 12


def find_existing_model_root(folder: Path) -> Path | None:
    #Return the folder that already contains a saved Eigenface model.

    for candidate in (folder, *folder.parents):
        if (candidate / "model" / "train_coefficients.npy").exists():
            return candidate

    return None


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

def visualize_eigenfaces(
    eigenfaces: np.ndarray,
    h: int,
    w: int,
    n_row: int = 3,
    n_col: int = 4,
) -> Image.Image:

    titles = [
        f"eigenface {i}"
        for i in range(len(eigenfaces))
    ]

    # Increase the figure size to make the eigenfaces larger.
    fig = plt.figure(figsize=(2.8 * n_col, 2.8 * n_row))

    # Reduce spacing to give more room to the face images.
    plt.subplots_adjust(
        bottom=0.03,
        left=0.02,
        right=0.98,
        top=0.92,
        hspace=0.30,
        wspace=0.15,
    )

    for i in range(min(n_row * n_col, len(eigenfaces))):
        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(
            eigenfaces[i].reshape((h, w)),
            cmap=plt.cm.gray,
        )

        plt.title(titles[i], size=13)
        plt.xticks(())
        plt.yticks(())

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)

    result_image = Image.open(buffer).convert("RGB").copy()

    plt.close(fig)
    buffer.close()

    return result_image

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
            "Choose an EMPTY folder where the training images, test images "
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
            output = Path(folder)

            train_folder = output / "train"
            test_folder = output / "test"
            model_folder = output / "model"

            # 1. Avoid selecting a folder inside a previous Eigenface model.
            existing_root = find_existing_model_root(output)

            if existing_root is not None and existing_root != output:
                raise ValueError(
                    "The selected folder is inside a previous Eigenface model:\n"
                    f"{existing_root}\n\n"
                    "Choose a different folder to avoid nested folders."
                )

            # 2. Ask before replacing previously generated results.
            old_folders = [
                p for p in (train_folder, test_folder, model_folder)
                if p.exists()
            ]

            if old_folders:
                replace = messagebox.askyesno(
                    "Eigenface builder",
                    "This folder already contains train/test/model folders.\n"
                    "Do you want to replace them?",
                    parent=parent,
                )

                if not replace:
                    return None

            # 3. Fix SSL certificate verification for the LFW download on macOS.
            ssl._create_default_https_context = ssl._create_unverified_context

            # 4. Download the same LFW dataset configuration as the professor.
            dataset = fetch_lfw_people(
                min_faces_per_person=70,
                resize=0.4,
            )

            images = dataset.images
            X = dataset.data
            labels = dataset.target
            target_names = dataset.target_names

            n_samples, h, w = images.shape

            # 5. Split the dataset exactly as in the professor's notebook.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                labels,
                test_size=0.25,
                random_state=42,
            )

            # 6. Ask how many PCA components to calculate.
            max_k = min(len(X_train), X_train.shape[1])

            k = simpledialog.askinteger(
                "Eigenface builder",
                f"Number of eigenfaces (1 to {max_k}):",
                parent=parent,
                initialvalue=min(100, max_k),
                minvalue=1,
                maxvalue=max_k,
            )

            if k is None:
                return None

            # 7. Remove previous results after the user confirms replacement.
            for old in old_folders:
                shutil.rmtree(old, ignore_errors=True)

            model_folder.mkdir(parents=True, exist_ok=True)

            # Save training/test images and their original labels.
            with open(
                output / "image_labels.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(file)
                writer.writerow(["split", "filename", "label", "person"])

                # Convert flattened images back to their original dimensions.
                save_images(
                    X_train.reshape(-1, h, w),
                    y_train,
                    target_names,
                    train_folder,
                    "train",
                    writer,
                )

                save_images(
                    X_test.reshape(-1, h, w),
                    y_test,
                    target_names,
                    test_folder,
                    "test",
                    writer,
                )

            # 8. Calculate PCA using the professor's configuration.
            print(
                f"Extracting the top {k} eigenfaces "
                f"from {X_train.shape[0]} faces"
            )

            pca = PCA(
                n_components=k,
                svd_solver="randomized",
                whiten=True,
            ).fit(X_train)

            # Project training and test images onto the PCA basis.
            train_coefficients = pca.transform(X_train)
            test_coefficients = pca.transform(X_test)

            # Convert PCA components into eigenface images.
            eigenfaces = pca.components_.reshape((k, h, w))

            # Calculate the mean face.
            mean_face = pca.mean_.reshape((h, w))

            # 9. Save the PCA model and its coefficients.
            np.save(model_folder / "mean_face.npy", mean_face)
            np.save(model_folder / "eigenfaces.npy", eigenfaces)

            np.save(
                model_folder / "train_coefficients.npy",
                train_coefficients,
            )

            np.save(
                model_folder / "test_coefficients.npy",
                test_coefficients,
            )

            np.save(model_folder / "train_labels.npy", y_train)
            np.save(model_folder / "test_labels.npy", y_test)
            np.save(model_folder / "target_names.npy", target_names)

            np.save(
                model_folder / "explained_variance_ratio.npy",
                pca.explained_variance_ratio_,
            )

            # 10. Visualize the first 12 eigenfaces as in the professor's notebook.
            n_shown = min(k, MAX_EIGENFACES_SHOWN)

            result_image = visualize_eigenfaces(
                eigenfaces[:n_shown],
                h,
                w,
            )

            # Save the visualization as a PNG image.
            result_image.save(output / "eigenfaces.png")

            return ToolResult(
                image=result_image,
                message=(
                    f"Eigenface model created successfully with {k} components "
                    f"({n_shown} shown)."
                ),
                details={
                    "Operation": "Eigenface builder",
                    "Dataset": "LFW People",
                    "Training images": len(X_train),
                    "Test images": len(X_test),
                    "People": len(target_names),
                    "Eigenfaces": k,
                    "Eigenfaces shown": n_shown,
                    "Image size": f"{w}x{h}",
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
