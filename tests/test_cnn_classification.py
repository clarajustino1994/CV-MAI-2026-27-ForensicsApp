import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.cnn_classification import (
    WEIGHTS_PATH,
    CnnClassificationTool,
    format_passport_value,
    load_class_names,
    preprocess,
    top_predictions,
)

CAT_IMAGE = Path(__file__).resolve().parents[1] / "Set8" / "test_images" / "cat.jpg"
DIALOG = "forensics_app.tools.cnn_classification.ask_details_action"
PICK_FILE = "forensics_app.tools.cnn_classification.filedialog.askopenfilename"


class CnnClassificationHelpersTests(unittest.TestCase):
    def test_preprocess_scales_and_resizes(self) -> None:
        image = Image.new("RGB", (40, 20), (255, 0, 0))
        batch = preprocess(image)
        self.assertEqual(batch.shape, (1, 128, 128, 3))
        self.assertAlmostEqual(float(batch.max()), 1.0, places=5)
        self.assertGreaterEqual(float(batch.min()), 0.0)

    def test_top_predictions_sorted_by_probability(self) -> None:
        names = ("a", "b", "c", "d")
        result = top_predictions(np.array([0.1, 0.6, 0.05, 0.25]), names, k=3)
        self.assertEqual([p.label for p in result], ["b", "d", "a"])

    def test_class_names_come_from_passport(self) -> None:
        self.assertEqual(len(load_class_names()), 10)
        self.assertEqual(load_class_names()[3], "cat")

    def test_format_passport_value(self) -> None:
        self.assertEqual(format_passport_value(["a", "b"]), "a, b")
        self.assertEqual(
            format_passport_value({"train": 1000, "test": 10}), "train: 1000, test: 10"
        )
        self.assertEqual(format_passport_value(3), "3")


class CnnClassificationDialogTests(unittest.TestCase):
    def test_works_without_an_open_image(self) -> None:
        self.assertFalse(CnnClassificationTool.requires_image)

    def test_dialog_shows_passport_and_close_returns_none(self) -> None:
        with (
            mock.patch(DIALOG, return_value=False) as dialog,
            mock.patch(PICK_FILE) as pick_file,
        ):
            self.assertIsNone(CnnClassificationTool().run(None, ImageDocument()))
        pick_file.assert_not_called()
        details = dialog.call_args.args[3]
        self.assertEqual(details["Model name"], "transfer_learning_inceptionv3")
        self.assertIn("cat", details["Class names"])

    def test_cancel_file_picker_returns_none(self) -> None:
        with (
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=""),
        ):
            self.assertIsNone(CnnClassificationTool().run(None, ImageDocument()))

    def test_missing_passport_reports_message(self) -> None:
        tool = CnnClassificationTool(passport_path=Path("does/not/exist.json"))
        with mock.patch(DIALOG) as dialog:
            result = tool.run(None, ImageDocument())
        dialog.assert_not_called()
        self.assertIn("passport", result.message)

    def test_unreadable_image_reports_message(self) -> None:
        with (
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=__file__),  # not an image
        ):
            result = CnnClassificationTool().run(None, ImageDocument())
        self.assertIsNone(result.image)
        self.assertIn("Could not open image", result.message)

    def test_missing_weights_reports_message(self) -> None:
        tool = CnnClassificationTool(weights_path=Path("does/not/exist.h5"))
        with (
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=str(CAT_IMAGE)),
        ):
            result = tool.run(None, ImageDocument())
        self.assertIsNone(result.image)
        self.assertIn("not found", result.message)


@unittest.skipUnless(WEIGHTS_PATH.exists(), "Set 8 weights not available")
class CnnClassificationModelTests(unittest.TestCase):
    def test_classifies_selected_cat_image(self) -> None:
        try:
            import keras  # noqa: F401
        except ImportError:
            self.skipTest("Keras is not installed")
        document = ImageDocument()
        with (
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=str(CAT_IMAGE)),
        ):
            result = CnnClassificationTool().run(None, document)
        self.assertEqual(document.path, CAT_IMAGE)
        self.assertEqual(result.details["Top 1"].split(":")[0], "cat")
        self.assertEqual(result.image.size, document.current.size)


if __name__ == "__main__":
    unittest.main()
