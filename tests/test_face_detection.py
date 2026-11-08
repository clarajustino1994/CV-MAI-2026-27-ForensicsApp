from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_detection import (
    SOURCE_IMAGE,
    SOURCE_WEBCAM,
    YUNET_MODEL_PATH,
    FaceDetectionTool,
    draw_faces,
)

MODULE = "forensics_app.tools.face_detection"

# One detected face with five landmarks and a score.
FACE = np.array(
    [
        [
            10,
            10,
            40,
            40,
            20,
            20,
            40,
            20,
            30,
            30,
            22,
            40,
            38,
            40,
            0.99,
        ]
    ],
    dtype=np.float32,
)


def make_document(image: Image.Image | None = None) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def white_image() -> Image.Image:
    return Image.new("RGB", (100, 80), "white")


class FaceDetectionImageTests(unittest.TestCase):

    def setUp(self) -> None:
        patcher = mock.patch(f"{MODULE}.ask_source", return_value=SOURCE_IMAGE)
        self.ask_source = patcher.start()
        self.addCleanup(patcher.stop)

    def test_detects_face(self) -> None:
        document = make_document(white_image())

        detector = mock.Mock()
        detector.detect.return_value = (None, FACE)

        with mock.patch(
            f"{MODULE}.cv2.FaceDetectorYN.create",
            return_value=detector,
        ):
            result = FaceDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (100, 80))
        self.assertEqual(result.details["Method"], "YuNet")
        self.assertEqual(result.details["Source"], "Image")
        self.assertEqual(result.details["Faces detected"], 1)
        self.assertIn("1 face(s) detected", result.message)

        detector.detect.assert_called_once()

    def test_no_faces_detected(self) -> None:
        document = make_document(white_image())

        detector = mock.Mock()
        detector.detect.return_value = (None, None)

        with mock.patch(
            f"{MODULE}.cv2.FaceDetectorYN.create",
            return_value=detector,
        ):
            result = FaceDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Faces detected"], 0)
        self.assertIn("0 face(s) detected", result.message)

    def test_runs_with_bundled_model(self) -> None:
        document = make_document(white_image())

        result = FaceDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertEqual(result.details["Faces detected"], 0)

    def test_loaded_image_skips_file_picker(self) -> None:
        document = make_document(white_image())

        with mock.patch(f"{MODULE}.filedialog.askopenfilename") as picker:
            FaceDetectionTool().run(None, document)

        picker.assert_not_called()

    def test_asks_for_file_when_no_image_loaded(self) -> None:
        document = make_document()

        with (
            mock.patch(
                f"{MODULE}.filedialog.askopenfilename", return_value="face.png"
            ) as picker,
            mock.patch.object(
                document,
                "load",
                side_effect=lambda _path: setattr(document, "current", white_image()),
            ) as load,
        ):
            result = FaceDetectionTool().run(None, document)

        picker.assert_called_once()
        load.assert_called_once_with("face.png")
        self.assertIsNotNone(result)

    def test_cancelled_file_picker_returns_none(self) -> None:
        document = make_document()

        with mock.patch(f"{MODULE}.filedialog.askopenfilename", return_value=""):
            result = FaceDetectionTool().run(None, document)

        self.assertIsNone(result)


class FaceDetectionSourceTests(unittest.TestCase):

    def test_cancelled_selector_returns_none(self) -> None:
        with mock.patch(f"{MODULE}.ask_source", return_value=None):
            result = FaceDetectionTool().run(None, make_document(white_image()))

        self.assertIsNone(result)

    def test_webcam_capture_returns_frame(self) -> None:
        frame = white_image()

        with (
            mock.patch(f"{MODULE}.ask_source", return_value=SOURCE_WEBCAM),
            mock.patch(f"{MODULE}.run_webcam", return_value=(frame, 2)) as webcam,
        ):
            result = FaceDetectionTool().run(None, make_document())

        webcam.assert_called_once()
        self.assertIsNotNone(result)
        self.assertIs(result.image, frame)
        self.assertEqual(result.details["Source"], "Webcam")
        self.assertEqual(result.details["Faces detected"], 2)

    def test_cancelled_webcam_returns_none(self) -> None:
        with (
            mock.patch(f"{MODULE}.ask_source", return_value=SOURCE_WEBCAM),
            mock.patch(f"{MODULE}.run_webcam", return_value=None),
        ):
            result = FaceDetectionTool().run(None, make_document())

        self.assertIsNone(result)

    def test_does_not_require_loaded_image(self) -> None:
        self.assertFalse(FaceDetectionTool.requires_image)


class DrawFacesTests(unittest.TestCase):

    def test_draws_box_without_changing_input(self) -> None:
        image = white_image()

        result, count = draw_faces(image, FACE)

        self.assertEqual(count, 1)
        self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))
        self.assertNotEqual(result.getpixel((10, 10)), (255, 255, 255))

    def test_no_faces(self) -> None:
        _, count = draw_faces(white_image(), None)

        self.assertEqual(count, 0)

    def test_bundled_model_exists(self) -> None:
        self.assertTrue(YUNET_MODEL_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
