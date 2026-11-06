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

    def test_recognizes_faces(self) -> None:

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)

            model_folder = output / "model"
            test_folder = output / "test"

            model_folder.mkdir()
            test_folder.mkdir()

            # Create synthetic training coefficients for two people.
            train_coefficients = np.array([
                [-1.0, -1.0],
                [-1.1, -0.9],
                [-0.9, -1.1],
                [-1.2, -1.0],
                [-0.8, -0.9],
                [1.0, 1.0],
                [1.1, 0.9],
                [0.9, 1.1],
                [1.2, 1.0],
                [0.8, 0.9],
            ])

            train_labels = np.array([
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
            ])

            test_coefficients = np.array([
                [-1.0, -0.9],
                [1.0, 0.9],
            ])

            test_labels = np.array([0, 1])

            # Save the synthetic PCA model.
            np.save(
                model_folder / "train_coefficients.npy",
                train_coefficients,
            )

            np.save(
                model_folder / "test_coefficients.npy",
                test_coefficients,
            )

            np.save(
                model_folder / "train_labels.npy",
                train_labels,
            )

            np.save(
                model_folder / "test_labels.npy",
                test_labels,
            )

            np.save(
                model_folder / "target_names.npy",
                np.array(["Person A", "Person B"]),
            )

            # Create the corresponding test images.
            for index in range(2):
                image = Image.new(
                    "L",
                    (20, 20),
                    color=100 + index * 50,
                )

                image.save(
                    test_folder / f"test_{index:04d}.jpg"
                )

            # Run the face recognition tool.
            with (
                mock.patch(
                    f"{MODULE}.filedialog.askdirectory",
                    return_value=temp_dir,
                ),
                mock.patch(f"{MODULE}.messagebox.showerror") as show_error,
            ):
                result = FaceRecognitionTool().run(
                    None,
                    ImageDocument(),
                )

            # Check that the operation was successful.
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)

            show_error.assert_not_called()

            self.assertEqual(
                result.details["Operation"],
                "Eigenfaces face recognition",
            )

            self.assertEqual(
                result.details["Method"],
                "PCA + SVM (RBF)",
            )

            self.assertEqual(
                result.details["Test images displayed"],
                2,
            )

            self.assertEqual(
                result.details["Total test images"],
                2,
            )

            self.assertEqual(
                result.details["Training images"],
                10,
            )

            self.assertEqual(
                result.details["Correct predictions"],
                2,
            )

            self.assertEqual(
                result.details["Incorrect predictions"],
                0,
            )

            # Check the generated gallery.
            self.assertEqual(result.image.size, (760, 235))

            # Check that the PNG result was saved.
            self.assertTrue(
                (output / "face_recognition_results.png").exists()
            )

    def test_cancelled_folder_selection(self) -> None:

        with (
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                return_value="",
            ),
            mock.patch(f"{MODULE}.messagebox.showerror") as show_error,
        ):
            result = FaceRecognitionTool().run(
                None,
                ImageDocument(),
            )

        self.assertIsNone(result)
        show_error.assert_not_called()

if __name__ == "__main__":
    unittest.main()
