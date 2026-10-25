from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.constants import DESCRIPTORS, ORB, SIFT
from forensics_app.tools.descriptor_extractor import DescriptorExtractorTool

MODULE = "forensics_app.tools.descriptor_extractor"


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def blocks_image(size: int = 128) -> Image.Image:
    """Random black and white blocks: many corners for ORB and SIFT."""
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 2, size=(size // 8, size // 8), dtype=np.uint8)
    pixels = np.kron(blocks, np.ones((8, 8), dtype=np.uint8)) * 255
    return Image.fromarray(pixels, mode="L").convert("RGB")


def run_tool(
    document: ImageDocument,
    method: str | None,
    integers: list[int | None] | None = None,
    floats: list[float | None] | None = None,
):
    choice = None if method is None else DESCRIPTORS.index(method)
    with (
        mock.patch(f"{MODULE}.ask_choice", return_value=choice),
        mock.patch(f"{MODULE}.simpledialog.askinteger", side_effect=integers or []),
        mock.patch(f"{MODULE}.simpledialog.askfloat", side_effect=floats or []),
    ):
        return DescriptorExtractorTool().run(None, document)


class DescriptorExtractorTests(unittest.TestCase):
    def test_orb_respects_keypoint_budget_and_gives_binary_descriptors(
        self,
    ) -> None:
        image = blocks_image()
        document = make_document(image)
        before = np.asarray(image).copy()

        result = run_tool(document, ORB, integers=[50])

        self.assertEqual(result.details["Method"], ORB)
        self.assertLessEqual(result.details["Keypoints"], 50)
        self.assertGreater(result.details["Keypoints"], 0)
        # ORB describes each keypoint with 256 binary tests
        self.assertEqual(
            result.details["Descriptors shape"],
            f"({result.details['Keypoints']}, 256)",
        )
        self.assertEqual(
            result.message,
            f"{result.details['Keypoints']} " "keypoints extracted using ORB.",
        )
        np.testing.assert_array_equal(np.asarray(document.current), before)

    def test_sift_gives_128_value_descriptors(self) -> None:
        document = make_document(blocks_image())

        result = run_tool(document, SIFT, integers=[4, 3], floats=[1.6])

        self.assertEqual(result.details["Method"], SIFT)
        self.assertGreater(result.details["Keypoints"], 0)
        # SIFT: 4x4 cells with 8 orientation bins = 128 values per keypoint
        self.assertEqual(
            result.details["Descriptors shape"],
            f"({result.details['Keypoints']}, 128)",
        )

    def test_keypoints_are_drawn_in_red_on_a_grey_copy(self) -> None:
        image = blocks_image()
        result = run_tool(make_document(image), ORB, integers=[50])

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, image.size)
        array = np.asarray(result.image)
        self.assertTrue((array == [255, 0, 0]).all(axis=2).any())

    def test_flat_image_reports_no_keypoints_instead_of_crashing(self) -> None:
        flat = Image.new("RGB", (64, 64), (128, 128, 128))
        for method, integers, floats in (
            (ORB, [50], []),
            (SIFT, [4, 3], [1.6]),
        ):
            with self.subTest(method=method):
                result = run_tool(make_document(flat), method, integers, floats)
                self.assertIsNone(result.image)
                self.assertIn("No keypoints", result.message)

    def test_tool_returns_none_when_any_dialog_is_cancelled(self) -> None:
        document = make_document(blocks_image())

        self.assertIsNone(run_tool(document, None))
        self.assertIsNone(run_tool(document, ORB, integers=[None]))
        self.assertIsNone(run_tool(document, SIFT, integers=[None]))
        self.assertIsNone(run_tool(document, SIFT, integers=[4, None]))
        self.assertIsNone(run_tool(document, SIFT, integers=[4, 3], floats=[None]))


if __name__ == "__main__":
    unittest.main()
