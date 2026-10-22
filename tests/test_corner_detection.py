
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.corner_detection import CornerDetectionTool


MODULE = "forensics_app.tools.corner_detection"


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def textured_image(size: int = 128) -> Image.Image:
    """Create an image with corners for testing."""
    pixels = np.zeros((size, size), dtype=np.uint8)
    pixels[20:60, 20:60] = 255
    pixels[70:110, 70:110] = 180
    return Image.fromarray(pixels).convert("RGB")


class CornerDetectionTests(unittest.TestCase):

    def test_fast_detects_corners(self) -> None:
        document = make_document(textured_image())

        # Simulate a FAST response containing one clear corner.
        fast_response = np.zeros((128, 128), dtype=np.float32)
        fast_response[40, 40] = 1.0

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="FAST",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askfloat",
                return_value=0.1,
            ),
            mock.patch(
                f"{MODULE}.corner_fast",
                return_value=fast_response,
            ),
        ):
            result = CornerDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (128, 256))
        self.assertEqual(result.details["Method"], "FAST")
        self.assertGreater(result.details["Number of peaks"], 0)

    def test_harris_detects_corners(self) -> None:
        document = make_document(textured_image())

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="HARRIS",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askfloat",
                return_value=0.1,
            ),
        ):
            result = CornerDetectionTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Method"], "HARRIS")
        self.assertEqual(result.details["Peak threshold"], "0.10")
        self.assertGreater(result.details["Number of peaks"], 0)


if __name__ == "__main__":
    unittest.main()
