from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_recognition import FaceRecognitionTool

MODULE = "forensics_app.tools.face_recognition"


class FaceRecognitionTests(unittest.TestCase):

    def test_recognizes_closest_faces(self) -> None:

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            model_folder = output / "model"
            train_folder = output / "train"
            test_folder = output / "test"

            model_folder.mkdir()
            train_folder.mkdir()
            test_folder.mkdir()

            # Create a small synthetic PCA model
            np.save(model_folder / "mean_face.npy", np.zeros((2, 2)))
            np.save(
                model_folder / "eigenfaces.npy",
                np.array([
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                ]),
            )
            np.save(
                model_folder / "train_coefficients.npy",
                np.array([
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                ]),
            )
            np.save(
                model_folder / "train_labels.npy",
                np.array([0, 1, 0]),
            )
            np.save(
                model_folder / "target_names.npy",
                np.array(["Person A", "Person B"]),
            )

            # Create three training images
            for index in range(3):
                image = Image.new("L", (2, 2), color=100 + index * 20)
                image.save(train_folder / f"train_{index:04d}.jpg")

            # Create one test image
            test_path = test_folder / "test_0001.jpg"
            Image.new("L", (2, 2), color=120).save(test_path)

            with (
                mock.patch(f"{MODULE}.messagebox.showinfo"),
                mock.patch(
                    f"{MODULE}.filedialog.askdirectory",
                    return_value=temp_dir,
                ),
                mock.patch(
                    f"{MODULE}.filedialog.askopenfilename",
                    return_value=str(test_path),
                ),
            ):
                result = FaceRecognitionTool().run(None, ImageDocument())

            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)
            self.assertEqual(
                result.details["Operation"],
                "Face recognition with Eigenfaces",
            )
            self.assertEqual(result.details["Test image"], "test_0001.jpg")
            self.assertEqual(result.details["Neighbors displayed"], 3)
            self.assertEqual(result.image.size, (640, 210))

    def test_cancelled_folder_selection(self) -> None:

        with (
            mock.patch(f"{MODULE}.messagebox.showinfo"),
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                return_value="",
            ),
        ):
            result = FaceRecognitionTool().run(None, ImageDocument())

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
