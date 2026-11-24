import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.yolo_detection import (
    PRETRAINED_WEIGHTS,
    SAMPLES_DIR,
    SET9_DIR,
    TRAINED_WEIGHTS,
    Box,
    YoloBloodCellTool,
    YoloPretrainedTool,
    box_iou,
    draw_gt_and_predictions,
    find_label_file,
    match_boxes,
    read_training_summary,
    read_yolo_labels,
)

MODULE = "forensics_app.tools.yolo_detection"
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
BLOOD_SAMPLE = SAMPLES_DIR / "blood_cell_01.jpg"
BUS_SAMPLE = SAMPLES_DIR / "bus.jpg"


class YoloHelpersTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        iou = box_iou(
            [[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
        )
        np.testing.assert_allclose(iou, [[1.0, 50 / 150, 0.0]])

    def test_read_yolo_labels_to_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("1 0.5 0.5 0.2 0.4\n\n", encoding="utf-8")
            (box,) = read_yolo_labels(path, width=100, height=50)
        self.assertEqual(box.cls, 1)
        self.assertEqual(box.xyxy, (40.0, 15.0, 60.0, 35.0))
        self.assertIsNone(box.conf)

    def test_find_label_file_in_sibling_labels_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = Path(tmp) / "images", Path(tmp) / "labels"
            images.mkdir()
            labels.mkdir()
            (labels / "x.txt").write_text("", encoding="utf-8")
            self.assertEqual(find_label_file(images / "x.jpg"), labels / "x.txt")
            self.assertIsNone(find_label_file(images / "y.jpg"))

    def test_match_boxes_counts_tp_fp_fn(self) -> None:
        gt = [Box(0, 0, 0, 10, 10), Box(1, 20, 20, 30, 30), Box(1, 40, 40, 50, 50)]
        pred = [
            Box(0, 0, 0, 10, 10, conf=0.9),  # TP
            Box(0, 20, 20, 30, 30, conf=0.8),  # right place, wrong class -> FP
            Box(1, 41, 41, 50, 50, conf=0.7),  # TP
            Box(1, 40, 40, 50, 50, conf=0.6),  # duplicate of a matched GT -> FP
        ]
        result = match_boxes(gt, pred)
        self.assertEqual((result.tp, result.fp, result.fn), (2, 2, 1))
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 2 / 3)

    def test_match_boxes_empty(self) -> None:
        result = match_boxes([], [Box(0, 0, 0, 1, 1, conf=0.5)])
        self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 0))
        self.assertEqual(result.recall, 0.0)

    def test_draw_keeps_size(self) -> None:
        image = Image.new("RGB", (64, 48))
        out = draw_gt_and_predictions(
            image, [Box(0, 5, 5, 30, 30)], [Box(0, 6, 6, 31, 31, conf=0.9)], ("PC",)
        )
        self.assertEqual(out.size, image.size)
        self.assertNotEqual(out.tobytes(), image.tobytes())

    def test_training_summary_reads_run_files(self) -> None:
        summary = read_training_summary()
        self.assertEqual(summary["Epochs"], "50")
        self.assertEqual(summary["Architecture"], "yolo11n.yaml")
        self.assertTrue(summary["Best epoch (val)"].startswith("48:"))

    def test_sample_images_have_labels(self) -> None:
        for i in range(1, 6):
            image = SAMPLES_DIR / f"blood_cell_{i:02d}.jpg"
            self.assertTrue(image.is_file(), image)
            self.assertIsNotNone(find_label_file(image))


class YoloDialogTests(unittest.TestCase):
    def test_blood_cell_tool_needs_no_open_image(self) -> None:
        self.assertFalse(YoloBloodCellTool.requires_image)
        self.assertTrue(YoloPretrainedTool.requires_image)

    def test_close_returns_none(self) -> None:
        with (
            mock.patch(f"{MODULE}.ask_details_action", return_value=False) as dialog,
            mock.patch(f"{MODULE}.filedialog.askopenfilename") as pick_file,
        ):
            self.assertIsNone(YoloBloodCellTool().run(None, ImageDocument()))
        pick_file.assert_not_called()
        details = dialog.call_args.args[3]
        self.assertEqual(details["Classes"], "PC, RBC, WBC")

    def test_cancel_confidence_returns_none(self) -> None:
        with mock.patch(f"{MODULE}.simpledialog.askfloat", return_value=None):
            self.assertIsNone(YoloPretrainedTool().run(None, ImageDocument()))

    def test_missing_weights_message(self) -> None:
        document = ImageDocument()
        document.load(BUS_SAMPLE)
        tool = YoloPretrainedTool(weights_path=SET9_DIR / "missing.pt")
        self.assertIn("not found", tool.detect(document).message)


@unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics is not installed")
class YoloInferenceTests(unittest.TestCase):
    def test_pretrained_finds_bus_and_people(self) -> None:
        self.assertTrue(PRETRAINED_WEIGHTS.is_file())
        document = ImageDocument()
        document.load(BUS_SAMPLE)
        result = YoloPretrainedTool().detect(document)
        self.assertIn("person", result.details["Per class"])
        self.assertIn("bus", result.details["Per class"])
        self.assertEqual(result.image.size, document.current.size)

    def test_trained_model_on_unseen_image(self) -> None:
        self.assertTrue(TRAINED_WEIGHTS.is_file())
        with (
            mock.patch(f"{MODULE}.ask_details_action", return_value=True),
            mock.patch(
                f"{MODULE}.filedialog.askopenfilename", return_value=str(BLOOD_SAMPLE)
            ),
        ):
            result = YoloBloodCellTool().run(None, ImageDocument())
        # Most false positives are cells cut by the border that the GT skips
        self.assertGreater(float(result.details["Recall"]), 0.9)
        self.assertGreater(float(result.details["Precision"]), 0.7)
        self.assertIn("TP", result.details["Matched (IoU >= 0.5)"])


if __name__ == "__main__":
    unittest.main()
