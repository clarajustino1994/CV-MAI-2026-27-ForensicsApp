from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_detection import FaceDetectionTool

MODULE = "forensics_app.tools.face_detection"


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


class FaceDetectionTests(unittest.TestCase):

    def test_detects_face(self) -> None:
        document = make_document(
            Image.new("RGB", (100, 80), "white")
        )

        # Simulate one detected face with five landmarks.
        face = np.array([[
            10, 10, 40, 40,
            20, 20, 40, 20, 30, 30,
            22, 40, 38, 40,
            0.99,
        ]], dtype=np.float32)

        detector = mock.Mock()
        detector.detect.return_value = (None, face)

        with (
            mock.patch(
                f"{MODULE}.filedialog.askopenfilename",
                return_value="yunet.onnx",
            ),
            mock.patch(
                f"{MODULE}.cv2.FaceDetectorYN.create",
                return_value=detector,
            ),
        ):
            result = FaceDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (100, 80))
        self.assertEqual(result.details["Method"], "YuNet")
        self.assertEqual(result.details["Faces detected"], 1)
        self.assertIn("1 face(s) detected", result.message)

        detector.detect.assert_called_once()

    def test_no_faces_detected(self) -> None:
        document = make_document(
            Image.new("RGB", (100, 80), "white")
        )

        detector = mock.Mock()
        detector.detect.return_value = (None, None)

        with (
            mock.patch(
                f"{MODULE}.filedialog.askopenfilename",
                return_value="yunet.onnx",
            ),
            mock.patch(
                f"{MODULE}.cv2.FaceDetectorYN.create",
                return_value=detector,
            ),
        ):
            result = FaceDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Faces detected"], 0)
        self.assertIn("0 face(s) detected", result.message)


if __name__ == "__main__":
    unittest.main()