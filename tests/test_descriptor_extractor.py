
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.descriptor_extractor import DescriptorExtractorTool


MODULE = "forensics_app.tools.descriptor_extractor"


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def textured_image(size: int = 128) -> Image.Image:
    """Create an image with texture for testing."""
    rng = np.random.default_rng(42)
    pixels = rng.integers(
        0, 256, size=(size, size), dtype=np.uint8
    )
    return Image.fromarray(pixels).convert("RGB")


class DescriptorExtractorTests(unittest.TestCase):

    def test_orb_extracts_keypoints_and_descriptors(self) -> None:
        document = make_document(textured_image())

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="ORB",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askinteger",
                return_value=200,
            ),
        ):
            result = DescriptorExtractorTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Method"], "ORB")
        self.assertGreater(result.details["Keypoints"], 0)

    def test_sift_extracts_keypoints_and_descriptors(self) -> None:
        document = make_document(textured_image())

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="SIFT",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askinteger",
                side_effect=[8, 3],
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askfloat",
                return_value=1.6,
            ),
        ):
            result = DescriptorExtractorTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Method"], "SIFT")
        self.assertGreater(result.details["Keypoints"], 0)

    def test_returns_none_when_method_is_cancelled(self) -> None:
        document = make_document(textured_image())

        with mock.patch(
            f"{MODULE}.simpledialog.askstring",
            return_value=None,
        ):
            result = DescriptorExtractorTool().run(None, document)

        self.assertIsNone(result)

    def test_returns_none_when_orb_configuration_is_cancelled(self) -> None:
        document = make_document(textured_image())

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="ORB",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askinteger",
                return_value=None,
            ),
        ):
            result = DescriptorExtractorTool().run(None, document)

        self.assertIsNone(result)

    def test_rejects_invalid_method(self) -> None:
        document = make_document(textured_image())

        with mock.patch(
            f"{MODULE}.simpledialog.askstring",
            return_value="INVALID",
        ):
            result = DescriptorExtractorTool().run(None, document)

        self.assertIsNotNone(result)
        self.assertIn("Choose ORB or SIFT", result.message)

    def test_original_image_is_not_modified(self) -> None:
        image = textured_image()
        document = make_document(image.copy())
        original_pixels = np.asarray(document.current).copy()

        with (
            mock.patch(
                f"{MODULE}.simpledialog.askstring",
                return_value="ORB",
            ),
            mock.patch(
                f"{MODULE}.simpledialog.askinteger",
                return_value=200,
            ),
        ):
            DescriptorExtractorTool().run(None, document)

        np.testing.assert_array_equal(
            np.asarray(document.current),
            original_pixels,
        )

if __name__ == "__main__":
    unittest.main()