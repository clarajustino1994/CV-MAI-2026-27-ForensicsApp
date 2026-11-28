import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.cnn_classification import (
    COMPLETE_VGG,
    DEEPER_VGG,
    MODEL_DIR,
    MODELS,
    SHALLOW_CNN,
    TRANSFER_INCEPTION,
    CnnClassificationTool,
    format_passport_value,
    load_class_names,
    preprocess,
    top_predictions,
)

CAT_IMAGE = Path(__file__).resolve().parents[1] / "Set8" / "test_images" / "cat.jpg"
MODULE = "forensics_app.tools.cnn_classification"
CHOICE = f"{MODULE}.ask_choice"
DIALOG = f"{MODULE}.ask_details_action"
PICK_FILE = f"{MODULE}.filedialog.askopenfilename"


class CnnClassificationHelpersTests(unittest.TestCase):
    def test_preprocess_scales_and_resizes(self) -> None:
        image = Image.new("RGB", (40, 20), (255, 0, 0))
        for size in ((128, 128), (32, 32)):
            batch = preprocess(image, size)
            self.assertEqual(batch.shape, (1, *size, 3))
            self.assertAlmostEqual(float(batch.max()), 1.0, places=5)
            self.assertGreaterEqual(float(batch.min()), 0.0)

    def test_top_predictions_sorted_by_probability(self) -> None:
        names = ("a", "b", "c", "d")
        result = top_predictions(np.array([0.1, 0.6, 0.05, 0.25]), names, k=3)
        self.assertEqual([p.label for p in result], ["b", "d", "a"])

    def test_class_names_come_from_passport(self) -> None:
        names = load_class_names(TRANSFER_INCEPTION.passport_path(MODEL_DIR))
        self.assertEqual(len(names), 10)
        self.assertEqual(names[3], "cat")

    def test_missing_passport_falls_back_to_cifar_names(self) -> None:
        self.assertEqual(load_class_names(Path("nope.json"))[0], "airplane")

    def test_format_passport_value(self) -> None:
        self.assertEqual(format_passport_value(["a", "b"]), "a, b")
        self.assertEqual(
            format_passport_value({"train": 1000, "test": 10}), "train: 1000, test: 10"
        )
        self.assertEqual(format_passport_value(3), "3")

    def test_training_history_is_summarised(self) -> None:
        history = {"accuracy": [0.1, 0.5, 0.75], "loss": [2.0, 1.0, 0.5]}
        self.assertEqual(
            format_passport_value(history),
            "3 epochs; final accuracy 0.750, loss 0.500",
        )

    def test_the_four_models_are_offered_in_order(self) -> None:
        self.assertEqual(
            MODELS, (SHALLOW_CNN, DEEPER_VGG, COMPLETE_VGG, TRANSFER_INCEPTION)
        )


class CnnClassificationDialogTests(unittest.TestCase):
    def run_tool(self, model_index, details_clicked=True, picked="", tool=None):
        options_seen: list = []

        def fake_choice(_parent, _title, _prompt, options):
            options_seen.extend(options)
            return model_index

        with (
            mock.patch(CHOICE, fake_choice),
            mock.patch(DIALOG, return_value=details_clicked) as dialog,
            mock.patch(PICK_FILE, return_value=picked) as pick_file,
        ):
            result = (tool or CnnClassificationTool()).run(None, ImageDocument())
        return result, options_seen, dialog, pick_file

    def test_works_without_an_open_image(self) -> None:
        self.assertFalse(CnnClassificationTool.requires_image)

    def test_model_choice_lists_the_four_models(self) -> None:
        result, options, dialog, _ = self.run_tool(None)
        self.assertIsNone(result)
        self.assertEqual(options, [model.title for model in MODELS])
        dialog.assert_not_called()

    def test_passport_of_chosen_model_is_shown_before_the_image(self) -> None:
        for index, model in enumerate(MODELS):
            if not model.passport_path(MODEL_DIR).exists():
                continue
            with self.subTest(model=model.title):
                result, _, dialog, pick_file = self.run_tool(
                    index, details_clicked=False
                )
                self.assertIsNone(result)
                pick_file.assert_not_called()
                heading, details = dialog.call_args.args[2:4]
                self.assertEqual(heading, f"{model.title} passport")
                self.assertIn("cat", details["Class names"])

    def test_cancel_file_picker_returns_none(self) -> None:
        result, _, _, pick_file = self.run_tool(3, picked="")
        self.assertIsNone(result)
        pick_file.assert_called_once()

    def test_missing_passport_reports_message(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tool = CnnClassificationTool(model_dir=Path(folder))
            result, _, dialog, _ = self.run_tool(0, tool=tool)
        dialog.assert_not_called()
        self.assertIn("Shallow CNN passport", result.message)

    def test_unreadable_image_reports_message(self) -> None:
        result, *_ = self.run_tool(3, picked=__file__)  # not an image
        self.assertIsNone(result.image)
        self.assertIn("Could not open image", result.message)

    def test_missing_weights_reports_message(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            passport = TRANSFER_INCEPTION.passport_path(Path(folder))
            passport.write_text('{"model_name": "x"}', encoding="utf-8")
            tool = CnnClassificationTool(model_dir=Path(folder))
            result, *_ = self.run_tool(3, picked=str(CAT_IMAGE), tool=tool)
        self.assertIsNone(result.image)
        self.assertIn("not found", result.message)


class CnnClassificationModelTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import keras  # noqa: F401
        except ImportError:
            self.skipTest("Keras is not installed")
        self.tool = CnnClassificationTool()

    def test_each_model_classifies_the_selected_image(self) -> None:
        for index, model in enumerate(MODELS):
            if not model.weights_path(MODEL_DIR).exists():
                continue
            with self.subTest(model=model.title):
                document = ImageDocument()
                with (
                    mock.patch(CHOICE, return_value=index),
                    mock.patch(DIALOG, return_value=True),
                    mock.patch(PICK_FILE, return_value=str(CAT_IMAGE)),
                ):
                    result = self.tool.run(None, document)
                self.assertEqual(document.path, CAT_IMAGE)
                self.assertEqual(result.image.size, document.current.size)
                self.assertIn(model.title, result.details["Operation"])
                self.assertIn("Top 3", result.details)

    @unittest.skipUnless(
        TRANSFER_INCEPTION.weights_path(MODEL_DIR).exists(),
        "InceptionV3 weights not available",
    )
    def test_inception_recognises_the_cat(self) -> None:
        with (
            mock.patch(CHOICE, return_value=MODELS.index(TRANSFER_INCEPTION)),
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=str(CAT_IMAGE)),
        ):
            result = self.tool.run(None, ImageDocument())
        self.assertEqual(result.details["Top 1"].split(":")[0], "cat")


if __name__ == "__main__":
    unittest.main()
