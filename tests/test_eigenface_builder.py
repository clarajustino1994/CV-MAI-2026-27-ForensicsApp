from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.tools.eigenface_builder import EigenfaceBuilderTool

MODULE = "forensics_app.tools.eigenface_builder"


def make_dataset():
    """Create a small synthetic face dataset for testing."""
    images = np.random.RandomState(42).rand(8, 8, 8).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    return SimpleNamespace(
        images=images,
        target=labels,
        target_names=np.array(["Person A", "Person B"]),
    )


class EigenfaceBuilderTests(unittest.TestCase):

    def test_builds_eigenface_model(self) -> None:
        document = ImageDocument()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch(
                    f"{MODULE}.messagebox.showinfo"
                ),
                mock.patch(
                    f"{MODULE}.filedialog.askdirectory",
                    return_value=temp_dir,
                ),
                mock.patch(
                    f"{MODULE}.fetch_lfw_people",
                    return_value=make_dataset(),
                ),
                mock.patch(
                    f"{MODULE}.simpledialog.askinteger",
                    return_value=2,
                ),
            ):
                result = EigenfaceBuilderTool().run(None, document)

            self.assertIsNotNone(result)
            self.assertIsNotNone(result.image)
            self.assertEqual(result.details["Operation"], "Eigenface builder")
            self.assertEqual(result.details["Dataset"], "LFW People")
            self.assertEqual(result.details["Eigenfaces"], 2)

            model_folder = Path(temp_dir) / "model"

            self.assertTrue((model_folder / "mean_face.npy").exists())
            self.assertTrue((model_folder / "eigenfaces.npy").exists())
            self.assertTrue((model_folder / "train_coefficients.npy").exists())
            self.assertTrue((model_folder / "test_coefficients.npy").exists())
            self.assertTrue((Path(temp_dir) / "image_labels.csv").exists())

    def test_cancelled_folder_selection(self) -> None:
        document = ImageDocument()

        with (
            mock.patch(f"{MODULE}.messagebox.showinfo"),
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                return_value="",
            ),
            mock.patch(
                f"{MODULE}.fetch_lfw_people"
            ) as fetch_dataset,
        ):
            result = EigenfaceBuilderTool().run(None, document)

        self.assertIsNone(result)
        fetch_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
