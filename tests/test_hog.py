import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import HogParams, HogTool, compute_hog, hog_descriptor
from forensics_app.tools.matching_configs import PANEL_GAP
from forensics_app.tools.person_detection import (
    DOT_PRODUCT_COLOR,
    EUCLIDEAN_COLOR,
    Detection,
    PersonDetectionTool,
    detect_person,
    draw_detections,
    max_step,
)

HOG_MODULE = "forensics_app.tools.hog"
PERSON_MODULE = "forensics_app.tools.person_detection"


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def square_image(size: int = 32) -> Image.Image:
    # White square in the middle of a black image: strong edges for HOG
    pixels = np.zeros((size, size), dtype=np.uint8)
    pixels[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return Image.fromarray(pixels, mode="L").convert("RGB")


def textured_patch(width: int, height: int) -> Image.Image:
    # Random texture so every HOG block of the patch has gradients
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def scene_with_patch(patch: Image.Image, x: int, y: int) -> Image.Image:
    # Flat grey scene (no gradients) with the patch pasted at (x, y)
    scene = Image.new("RGB", (96, 80), (128, 128, 128))
    scene.paste(patch, (x, y))
    return scene


class HogTests(unittest.TestCase):
    def test_descriptor_length_follows_cells_and_blocks(self) -> None:
        # 32x32 with 8 px cells -> 4x4 cells -> 3x3 blocks of 2x2 cells
        params = HogParams(orientations=9, pixels_per_cell=8, cells_per_block=2)
        descriptor = hog_descriptor(square_image(32), params)
        self.assertEqual(descriptor.size, 3 * 3 * 2 * 2 * 9)

    def test_more_orientations_give_longer_descriptor(self) -> None:
        image = square_image(32)
        short = hog_descriptor(image, HogParams(orientations=2))
        long = hog_descriptor(image, HogParams(orientations=9))
        self.assertEqual(long.size, short.size * 9 // 2)

    def test_compute_hog_returns_visible_image_of_same_size(self) -> None:
        image = square_image(32)
        descriptor, hog_image = compute_hog(image, HogParams())
        self.assertEqual(hog_image.mode, "L")
        self.assertEqual(hog_image.size, image.size)
        self.assertEqual(np.asarray(hog_image).max(), 255)  # rescaled intensity
        np.testing.assert_allclose(descriptor, hog_descriptor(image, HogParams()))

    def test_flat_image_has_no_gradients(self) -> None:
        flat = Image.new("RGB", (32, 32), (90, 90, 90))
        descriptor, hog_image = compute_hog(flat, HogParams())
        self.assertFalse(descriptor.any())
        self.assertFalse(np.asarray(hog_image).any())

    def test_hog_tool_returns_image_and_details_without_mutating(self) -> None:
        document = make_document(square_image(32))
        params = HogParams(orientations=4, pixels_per_cell=8, cells_per_block=2)
        with mock.patch(f"{HOG_MODULE}.ask_hog_params", return_value=params):
            result = HogTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, (32, 32))
        self.assertEqual(document.current.mode, "RGB")
        self.assertEqual(result.details["Orientations"], 4)
        self.assertEqual(result.details["Pixels per cell"], "8x8")
        self.assertEqual(result.details["Cells per block"], "2x2")
        self.assertEqual(result.details["Descriptor length"], 3 * 3 * 2 * 2 * 4)

    def test_hog_tool_returns_none_when_cancelled(self) -> None:
        document = make_document(square_image(32))
        with mock.patch(f"{HOG_MODULE}.ask_hog_params", return_value=None):
            self.assertIsNone(HogTool().run(None, document))

    def test_hog_tool_rejects_block_larger_than_image(self) -> None:
        document = make_document(square_image(16))
        params = HogParams(orientations=9, pixels_per_cell=8, cells_per_block=3)
        with mock.patch(f"{HOG_MODULE}.ask_hog_params", return_value=params):
            result = HogTool().run(None, document)
        self.assertIsNone(result.image)
        self.assertIn("does not fit", result.message)


class PersonDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patch = textured_patch(32, 48)
        self.scene = scene_with_patch(self.patch, 24, 16)

    def write_template(self, image: Image.Image) -> str:
        handle, path = tempfile.mkstemp(suffix=".png")
        os.close(handle)
        image.save(path)
        self.addCleanup(os.remove, path)
        return path

    def test_detect_person_finds_planted_template_with_both_metrics(self) -> None:
        euclidean, dot_product = detect_person(self.scene, self.patch, step=8)
        self.assertEqual((euclidean.x, euclidean.y), (24, 16))
        self.assertEqual((dot_product.x, dot_product.y), (24, 16))
        self.assertAlmostEqual(euclidean.score, 0.0, places=5)
        self.assertGreater(dot_product.score, 0.0)

    def test_detect_person_rejects_template_larger_than_image(self) -> None:
        with self.assertRaises(ValueError):
            detect_person(self.patch, self.scene, step=4)

    def test_draw_detections_marks_both_windows_without_mutating(self) -> None:
        image = Image.new("RGB", (60, 60), (0, 0, 0))
        result = draw_detections(
            image, (10, 20), Detection(5, 5, 0.0), Detection(30, 30, 1.0)
        )
        self.assertEqual(result.size, (60, 60))
        self.assertEqual(result.getpixel((5, 5)), EUCLIDEAN_COLOR)
        self.assertEqual(result.getpixel((32, 32)), DOT_PRODUCT_COLOR)  # inset
        self.assertEqual(result.getpixel((10, 15)), (0, 0, 0))  # inside, not filled
        self.assertEqual(image.getpixel((5, 5)), (0, 0, 0))

    def test_draw_detections_keeps_both_colours_when_windows_match(self) -> None:
        image = Image.new("RGB", (60, 60), (0, 0, 0))
        same = Detection(10, 10, 0.0)
        result = draw_detections(image, (20, 30), same, same)
        self.assertEqual(result.getpixel((10, 10)), EUCLIDEAN_COLOR)  # outer
        self.assertEqual(result.getpixel((12, 12)), DOT_PRODUCT_COLOR)  # inner

    def test_person_tool_draws_detection_and_reports_window(self) -> None:
        document = make_document(self.scene)
        path = self.write_template(self.patch)
        with (
            mock.patch(f"{PERSON_MODULE}.ask_template_path", return_value=path),
            mock.patch(f"{PERSON_MODULE}.ask_step", return_value=8),
        ):
            result = PersonDetectionTool().run(None, document)
        # Detections, then the template and its HOG enlarged to the scene height
        template_width = round(self.patch.width * self.scene.height / self.patch.height)
        self.assertEqual(
            result.image.size,
            (self.scene.width + 2 * (template_width + PANEL_GAP), self.scene.height),
        )
        self.assertEqual(result.image.getpixel((24, 16)), EUCLIDEAN_COLOR)
        self.assertEqual(result.details["Window size"], "32x48")
        self.assertEqual(result.details["Step"], "8 px")
        self.assertTrue(result.details["Euclidean (red)"].startswith("(24, 16)"))
        self.assertEqual(
            document.current.getpixel((24, 16)), self.patch.getpixel((0, 0))
        )

    def test_person_tool_returns_none_when_cancelled(self) -> None:
        document = make_document(self.scene)
        with mock.patch(f"{PERSON_MODULE}.ask_template_path", return_value=""):
            self.assertIsNone(PersonDetectionTool().run(None, document))
        path = self.write_template(self.patch)
        with (
            mock.patch(f"{PERSON_MODULE}.ask_template_path", return_value=path),
            mock.patch(f"{PERSON_MODULE}.ask_step", return_value=None),
        ):
            self.assertIsNone(PersonDetectionTool().run(None, document))

    def test_person_tool_reports_template_too_large(self) -> None:
        document = make_document(self.patch)
        path = self.write_template(self.scene)
        with (
            mock.patch(f"{PERSON_MODULE}.ask_template_path", return_value=path),
            mock.patch(f"{PERSON_MODULE}.ask_step", return_value=4) as ask_step,
        ):
            result = PersonDetectionTool().run(None, document)
        self.assertIsNone(result.image)
        self.assertIn("smaller", result.message)
        ask_step.assert_not_called()  # rejected before asking for the step

    def test_max_step_is_the_largest_window_travel(self) -> None:
        # Pedestrian image 113x150 with the 49x129 person template
        self.assertEqual(max_step((113, 150), (49, 129)), 64)
        self.assertEqual(max_step((60, 200), (49, 129)), 71)  # vertical wins
        self.assertEqual(max_step((49, 129), (49, 129)), 1)  # window cannot move

    def test_person_tool_limits_step_to_image_and_template(self) -> None:
        document = make_document(self.scene)  # 96x80 scene, 32x48 template
        path = self.write_template(self.patch)
        with (
            mock.patch(f"{PERSON_MODULE}.ask_template_path", return_value=path),
            mock.patch(f"{PERSON_MODULE}.ask_step", return_value=None) as ask_step,
        ):
            PersonDetectionTool().run(None, document)
        ask_step.assert_called_once_with(None, 96 - 32)


if __name__ == "__main__":
    unittest.main()
