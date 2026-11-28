import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.tools.cnn_classification import DEEPER_VGG, MODELS
from forensics_app.tools.feature_visualization import (
    ACTION_PROJECT,
    FeatureVisualizationTool,
    load_features,
    plot_path,
    plot_projections,
)

MODULE = "forensics_app.tools.feature_visualization"
CLASS_NAMES = ("a", "b", "c")


def make_features(per_class: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(len(CLASS_NAMES)), per_class)
    # well separated clusters so both projections have real structure
    features = rng.normal(size=(len(labels), 16)) + labels[:, None] * 10.0
    return features.astype(np.float32), labels


def write_features(folder: Path, model) -> None:
    """Same arrays as the .npz files written by the Set 8 notebook."""
    features, labels = make_features()
    np.savez_compressed(
        model.features_path(folder),
        features=features,
        labels=labels,
        class_names=np.array(CLASS_NAMES),
    )


class FeatureVisualizationHelpersTests(unittest.TestCase):
    def test_load_features_reads_notebook_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            write_features(Path(folder), DEEPER_VGG)
            features, labels, names = load_features(
                DEEPER_VGG.features_path(Path(folder))
            )
        self.assertEqual(features.shape, (60, 16))
        self.assertEqual(len(labels), 60)
        self.assertEqual(names, CLASS_NAMES)

    def test_plot_projections_returns_rgb_image(self) -> None:
        _, labels = make_features(5)
        points = np.random.default_rng(1).normal(size=(len(labels), 2))
        image = plot_projections(
            {"t-SNE": points, "UMAP": points}, labels, CLASS_NAMES, "Deeper VGG"
        )
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(image.width, image.height)

    def test_plot_file_is_named_after_the_model(self) -> None:
        self.assertEqual(
            plot_path(DEEPER_VGG, Path("out")).name,
            "test_features_deeper_vgg_tsne_umap.png",
        )


class FeatureVisualizationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.tool = FeatureVisualizationTool(model_dir=self.root)

    def tearDown(self) -> None:
        self.folder.cleanup()

    def run_tool(self, model_index, details_clicked=True):
        self.options_seen: list = []

        def fake_choice(_parent, _title, _prompt, options):
            self.options_seen.extend(options)
            return model_index

        with (
            mock.patch(
                "forensics_app.tools.cnn_classification.ask_choice", fake_choice
            ),
            mock.patch(
                f"{MODULE}.ask_details_action", return_value=details_clicked
            ) as self.dialog,
            mock.patch(f"{MODULE}.ProgressDialog") as progress,
        ):
            progress.return_value.__enter__.return_value = None
            return self.tool.run(mock.Mock(), ImageDocument())

    def test_cancel_model_choice_returns_none(self) -> None:
        self.assertIsNone(self.run_tool(None))
        self.assertEqual(self.options_seen, [model.title for model in MODELS])
        self.dialog.assert_not_called()

    def test_missing_features_reports_message(self) -> None:
        result = self.run_tool(MODELS.index(DEEPER_VGG))
        self.dialog.assert_not_called()
        self.assertIn("test_features_deeper_vgg.npz not found", result.message)

    def test_passport_is_shown_and_close_returns_none(self) -> None:
        write_features(self.root, DEEPER_VGG)
        DEEPER_VGG.features_passport_path(self.root).write_text(
            json.dumps({"feature_dim": 16, "arrays": {"features": "float32"}}),
            encoding="utf-8",
        )
        self.assertIsNone(self.run_tool(MODELS.index(DEEPER_VGG), False))
        heading, details, action = self.dialog.call_args.args[2:5]
        self.assertEqual(heading, "Deeper VGG features passport")
        self.assertEqual(details["Feature dim"], "16")
        self.assertNotIn("Arrays", details)
        self.assertEqual(action, ACTION_PROJECT)
        self.assertFalse(plot_path(DEEPER_VGG, self.root).exists())

    def test_generates_projections_of_the_chosen_model(self) -> None:
        write_features(self.root, DEEPER_VGG)
        result = self.run_tool(MODELS.index(DEEPER_VGG))
        self.assertTrue(plot_path(DEEPER_VGG, self.root).exists())
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Feature size"], 16)
        self.assertIn("60 Deeper VGG test feature vectors", result.message)


if __name__ == "__main__":
    unittest.main()
