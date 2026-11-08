from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_recognition import (
    MODEL_NEW,
    MODEL_SAVED,
    FaceRecognitionTool,
    closest_matches,
    project_face,
)

MODULE = "forensics_app.tools.face_recognition"

H, W = 4, 3


def face(value: int) -> Image.Image:
    """A flat grey face image of the model size."""
    return Image.new("L", (W, H), color=value)


def write_model(output: Path) -> None:
    """Save a tiny synthetic Eigenface model with two people.

    One eigenface (all pixels equal), so w = mean brightness offset.
    Person A is dark (~50), Person B is bright (~200).
    """
    model = output / "model"
    train = output / "train"
    test = output / "test"
    for folder in (model, train, test):
        folder.mkdir()

    eigenface = np.full((1, H, W), 1 / np.sqrt(H * W))
    mean_face = np.full((H, W), 0.5)
    train_values = [40, 50, 60, 190, 200, 210]
    train_labels = np.array([0, 0, 0, 1, 1, 1])

    # Whitened coefficients, as saved by PCA(whiten=True).
    variance = np.array([4.0])
    raw = np.array(
        [
            [eigenface.ravel() @ (np.full(H * W, v / 255) - mean_face.ravel())]
            for v in train_values
        ]
    )

    np.save(model / "mean_face.npy", mean_face)
    np.save(model / "eigenfaces.npy", eigenface)
    np.save(model / "explained_variance.npy", variance)
    np.save(model / "train_coefficients.npy", raw / np.sqrt(variance))
    np.save(model / "train_labels.npy", train_labels)
    np.save(model / "target_names.npy", np.array(["Ann Alpha", "Bob Beta"]))

    with open(output / "image_labels.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "filename", "label", "person"])
        names = ["Ann Alpha", "Bob Beta"]
        for index, (value, label) in enumerate(zip(train_values, train_labels)):
            filename = f"train_{index:04d}.jpg"
            face(value).save(train / filename)
            writer.writerow(["train", filename, label, names[label]])

        face(205).save(test / "test_0000.jpg")
        writer.writerow(["test", "test_0000.jpg", 1, "Bob Beta"])


class FaceRecognitionTests(unittest.TestCase):

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        write_model(self.output)

    def run_tool(self, choice, folder, image):
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=choice),
            mock.patch(f"{MODULE}.filedialog.askdirectory", return_value=folder),
            mock.patch(
                f"{MODULE}.filedialog.askopenfilename", return_value=image
            ) as pick_image,
            mock.patch(f"{MODULE}.messagebox.showerror") as show_error,
        ):
            result = FaceRecognitionTool().run(None, ImageDocument())
        return result, pick_image, show_error

    def test_saved_model_recognizes_test_image(self) -> None:
        query = self.output / "test" / "test_0000.jpg"

        result, pick_image, show_error = self.run_tool(
            MODEL_SAVED, str(self.output), str(query)
        )

        show_error.assert_not_called()
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Predicted identity"], "Bob Beta")
        self.assertEqual(result.details["True identity"], "Bob Beta")
        self.assertIn("correct", result.message)
        self.assertTrue(result.details["Match 1"].startswith("Bob Beta"))
        # Query plus three matches fill one gallery row.
        self.assertEqual(result.image.size, (760, 235))
        # The file picker opens in the test folder.
        self.assertEqual(
            pick_image.call_args.kwargs["initialdir"],
            str(self.output / "test"),
        )

    def test_selecting_model_subfolder_finds_root(self) -> None:
        query = self.output / "test" / "test_0000.jpg"

        result, _, show_error = self.run_tool(
            MODEL_SAVED, str(self.output / "model"), str(query)
        )

        show_error.assert_not_called()
        self.assertEqual(result.details["Model folder"], str(self.output))

    def test_image_outside_dataset_has_unknown_identity(self) -> None:
        query = self.output / "outside.png"
        face(45).save(query)

        result, _, _ = self.run_tool(MODEL_SAVED, str(self.output), str(query))

        self.assertEqual(result.details["Predicted identity"], "Ann Alpha")
        self.assertEqual(result.details["True identity"], "Unknown")

    def test_new_model_is_built_before_asking_for_image(self) -> None:
        query = self.output / "test" / "test_0000.jpg"

        with (
            mock.patch(f"{MODULE}.choose_output_folder", return_value=self.output),
            mock.patch(f"{MODULE}.build_eigenface_model", return_value={}) as build,
        ):
            result, _, show_error = self.run_tool(MODEL_NEW, "", str(query))

        build.assert_called_once_with(None, self.output)
        show_error.assert_not_called()
        self.assertEqual(result.details["Predicted identity"], "Bob Beta")

    def test_cancelled_model_choice(self) -> None:
        result, pick_image, show_error = self.run_tool(None, "", "")

        self.assertIsNone(result)
        pick_image.assert_not_called()
        show_error.assert_not_called()

    def test_cancelled_image_choice(self) -> None:
        result, _, show_error = self.run_tool(MODEL_SAVED, str(self.output), "")

        self.assertIsNone(result)
        show_error.assert_not_called()

    def test_folder_without_model_shows_error(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            result, pick_image, show_error = self.run_tool(MODEL_SAVED, empty, "")

        self.assertIsNone(result)
        pick_image.assert_not_called()
        show_error.assert_called_once()

    def test_old_model_without_variance_shows_error(self) -> None:
        (self.output / "model" / "explained_variance.npy").unlink()

        result, _, show_error = self.run_tool(MODEL_SAVED, str(self.output), "")

        self.assertIsNone(result)
        show_error.assert_called_once()
        self.assertIn("explained_variance.npy", show_error.call_args.args[1])


class ProjectionTests(unittest.TestCase):

    def test_project_face_uses_mean_and_eigenfaces(self) -> None:
        model = {
            "mean_face": np.zeros((2, 2)),
            "eigenfaces": np.array([[[1.0, 0.0], [0.0, 0.0]]]),
        }
        image = Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8))

        w = project_face(image, model)

        np.testing.assert_allclose(w, [1.0])

    def test_closest_matches_sorted_by_distance(self) -> None:
        train = np.array([[0.0], [5.0], [1.0], [3.0]])

        indices, distances = closest_matches(np.array([0.9]), train, n=3)

        self.assertEqual(indices.tolist(), [2, 0, 3])
        np.testing.assert_allclose(distances, [0.1, 0.9, 2.1])


if __name__ == "__main__":
    unittest.main()
