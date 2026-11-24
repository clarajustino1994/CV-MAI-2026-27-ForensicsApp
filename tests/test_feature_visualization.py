import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.tools.feature_visualization import (
    ACTION_EXTRACT,
    ACTION_LOAD,
    FeatureVisualizationTool,
    load_features,
    plot_projections,
    save_features,
    stratified_indices,
)

MODULE = "forensics_app.tools.feature_visualization"
CLASS_NAMES = ("a", "b", "c")


def make_features(per_class: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(len(CLASS_NAMES)), per_class)
    # well separated clusters so both projections have real structure
    features = rng.normal(size=(len(labels), 16)) + labels[:, None] * 10.0
    return features.astype(np.float32), labels


class FeatureVisualizationHelpersTests(unittest.TestCase):
    def test_stratified_indices_takes_first_n_of_each_class(self) -> None:
        labels = np.array([0, 1, 0, 2, 1, 0, 2, 1, 2])
        indices = stratified_indices(labels, 2)
        np.testing.assert_array_equal(indices, [0, 2, 1, 4, 3, 6])

    def test_save_and_load_round_trip(self) -> None:
        features, labels = make_features()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sub" / "features.npz"
            save_features(path, features, labels, CLASS_NAMES)
            loaded, loaded_labels, names = load_features(path)
        np.testing.assert_array_equal(loaded, features)
        np.testing.assert_array_equal(loaded_labels, labels)
        self.assertEqual(names, CLASS_NAMES)

    def test_plot_projections_returns_rgb_image(self) -> None:
        _, labels = make_features(5)
        points = np.random.default_rng(1).normal(size=(len(labels), 2))
        image = plot_projections({"t-SNE": points, "UMAP": points}, labels, CLASS_NAMES)
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(image.width, image.height)


class FeatureVisualizationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.tool = FeatureVisualizationTool(root / "f.npz", root / "plot.png")
        self.document = ImageDocument()

    def tearDown(self) -> None:
        self.folder.cleanup()

    def run_tool(self, choice: int | None, options_seen: list) -> object:
        def fake_choice(_parent, _title, _prompt, options):
            options_seen.extend(options)
            return choice

        with (
            mock.patch(f"{MODULE}.ask_choice", fake_choice),
            mock.patch(f"{MODULE}.ProgressDialog") as dialog,
        ):
            dialog.return_value.__enter__.return_value = None
            return self.tool.run(mock.Mock(), self.document)

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self.run_tool(None, []))

    def test_without_saved_features_only_extract_is_offered(self) -> None:
        features, labels = make_features()
        options: list = []
        with (
            mock.patch(
                f"{MODULE}.extract_test_features", return_value=(features, labels)
            ) as extract,
            mock.patch(f"{MODULE}.load_class_names", return_value=CLASS_NAMES),
        ):
            result = self.run_tool(0, options)
        self.assertEqual(options, [ACTION_EXTRACT])
        extract.assert_called_once()
        self.assertTrue(self.tool.features_path.exists())
        self.assertTrue(self.tool.plot_path.exists())
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Feature size"], 16)

    def test_saved_features_are_loaded_without_extracting(self) -> None:
        features, labels = make_features()
        save_features(self.tool.features_path, features, labels, CLASS_NAMES)
        options: list = []
        with mock.patch(f"{MODULE}.extract_test_features") as extract:
            result = self.run_tool(0, options)
        self.assertEqual(options, [ACTION_LOAD, ACTION_EXTRACT])
        extract.assert_not_called()
        self.assertIn("60 test feature vectors", result.message)


if __name__ == "__main__":
    unittest.main()
