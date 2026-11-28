import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.unet_segmentation import (
    MODEL_PATH,
    SAMPLES_DIR,
    UNetSegmentationTool,
    compose_panels,
    dice,
    find_mask_file,
    iou,
    preprocess,
    read_food_mask,
)

DIALOG = "forensics_app.tools.unet_segmentation.ask_details_action"
PICK_FILE = "forensics_app.tools.unet_segmentation.filedialog.askopenfilename"

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class UNetHelpersTests(unittest.TestCase):
    def test_preprocess_shape_and_range(self) -> None:
        batch = preprocess(Image.new("RGB", (40, 20), (255, 0, 0)), (208, 128))
        self.assertEqual(batch.shape, (1, 3, 128, 208))
        self.assertAlmostEqual(float(batch.max()), 1.0, places=5)
        self.assertGreaterEqual(float(batch.min()), 0.0)

    def test_dice_and_iou_match_notebook_example(self) -> None:
        pred = np.zeros((4, 4), bool)
        gt = np.zeros((4, 4), bool)
        pred[:2, 1:3] = True
        gt[:2, 2:4] = True
        self.assertAlmostEqual(dice(pred, gt), 0.5)
        self.assertAlmostEqual(iou(pred, gt), 2 / 6)
        empty = np.zeros((2, 2), bool)
        self.assertEqual(dice(empty, empty), 1.0)

    def test_find_and_read_ground_truth_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "XTest").mkdir()
            (root / "yTest").mkdir()
            image_path = root / "XTest" / "a.jpg"
            Image.new("RGB", (8, 4)).save(image_path)
            self.assertIsNone(find_mask_file(image_path))

            classes = np.array([[0, 1, 4, 5], [6, 42, 2, 0]], dtype=np.uint8)
            Image.fromarray(classes).save(root / "yTest" / "a.png")
            mask_path = find_mask_file(image_path)
            self.assertEqual(mask_path, root / "yTest" / "a.png")
            food = read_food_mask(mask_path, first_food=5, size=(4, 2))
            np.testing.assert_array_equal(food, classes >= 5)
        self.assertIsNone(find_mask_file(Path("photos/a.jpg")))

    def test_compose_panels_two_per_row(self) -> None:
        panel = Image.new("RGB", (50, 30))
        four = compose_panels([("a", panel)] * 4)
        self.assertGreater(four.width, 2 * 50)
        self.assertLess(four.width, 3 * 50)
        self.assertGreater(four.height, 2 * 30)
        three = compose_panels([("a", panel)] * 3)
        self.assertEqual(three.size, four.size)


class UNetDialogTests(unittest.TestCase):
    def test_works_without_an_open_image(self) -> None:
        self.assertFalse(UNetSegmentationTool.requires_image)

    def test_missing_model_and_cancelled_picker_returns_none(self) -> None:
        tool = UNetSegmentationTool(model_path=Path("does/not/exist.pt"))
        with (
            mock.patch(PICK_FILE, return_value="") as pick,
            mock.patch(DIALOG) as dialog,
        ):
            self.assertIsNone(tool.run(None, ImageDocument()))
        pick.assert_called_once()
        dialog.assert_not_called()

    def test_unreadable_checkpoint_reports_message(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        tool = UNetSegmentationTool(model_path=Path(__file__))  # not a checkpoint
        with mock.patch(DIALOG) as dialog:
            result = tool.run(None, ImageDocument())
        dialog.assert_not_called()
        self.assertIn("Could not load", result.message)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class UNetModelTests(unittest.TestCase):
    """End to end with an untrained checkpoint saved in the notebook format."""

    def setUp(self) -> None:
        from forensics_app.tools.unet_segmentation import build_unet

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model_path = root / "unet.pt"
        cfg = {
            "in_ch": 3,
            "out_ch": 1,
            "base": 4,
            "H": 32,
            "W": 48,
            "first_food": 5,
            "threshold": 0.5,
        }
        torch.manual_seed(0)
        torch.save(
            {
                "state_dict": build_unet(3, 1, 4).state_dict(),
                "config": cfg,
                "val_dice": 0.9,
                "test_dice": 0.8,
            },
            self.model_path,
        )
        (root / "XTest").mkdir()
        (root / "yTest").mkdir()
        self.image_path = root / "XTest" / "tray.jpg"
        Image.new("RGB", (60, 40), (200, 120, 40)).save(self.image_path)
        Image.fromarray(np.full((40, 60), 7, np.uint8)).save(
            root / "yTest" / "tray.png"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dialog_shows_model_details(self) -> None:
        tool = UNetSegmentationTool(model_path=self.model_path)
        with mock.patch(DIALOG, return_value=False) as dialog:
            self.assertIsNone(tool.run(None, ImageDocument()))
        details = dialog.call_args.args[3]
        self.assertEqual(details["Input size"], "48x32")
        self.assertEqual(details["Test Dice"], "0.800")

    def test_segments_selected_image_with_ground_truth(self) -> None:
        tool = UNetSegmentationTool(model_path=self.model_path)
        document = ImageDocument()
        with (
            mock.patch(DIALOG, return_value=True),
            mock.patch(PICK_FILE, return_value=str(self.image_path)),
        ):
            result = tool.run(None, document)
        self.assertEqual(document.path, self.image_path)
        self.assertIn("Dice", result.details)
        self.assertEqual(result.details["Ground truth"], "tray.png")
        # Original, ground truth, mask and overlay in a 2x2 grid
        self.assertGreater(result.image.width, 2 * 60)
        self.assertGreater(result.image.height, 2 * 40)

    def test_image_without_ground_truth(self) -> None:
        other = Path(self.tmp.name) / "photo.png"
        Image.new("RGB", (30, 30)).save(other)
        document = ImageDocument()
        document.load(other)
        result = UNetSegmentationTool(model_path=self.model_path).segment(
            document, other
        )
        self.assertEqual(result.details["Ground truth"], "not found")
        self.assertNotIn("Dice", result.details)
        self.assertGreater(result.image.height, 2 * 30)


@unittest.skipUnless(
    MODEL_PATH.exists() and SAMPLES_DIR.exists(), "trained U-Net not saved"
)
class TrainedUNetTests(unittest.TestCase):
    def test_trained_model_segments_test_tray(self) -> None:
        image_path = sorted(SAMPLES_DIR.glob("*.jpg"))[0]
        document = ImageDocument()
        document.load(image_path)
        result = UNetSegmentationTool().segment(document, image_path)
        self.assertGreater(float(result.details["Dice"]), 0.7)


if __name__ == "__main__":
    unittest.main()
