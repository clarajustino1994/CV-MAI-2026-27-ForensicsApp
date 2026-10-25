import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.cross_correlation import (
    CrossCorrelationTool,
    compute_ncc,
    ncc_image,
)
from forensics_app.tools.matching_configs import (
    PANEL_GAP,
    MatchingConfigsTool,
    enlarge_to_height,
    side_by_side,
    thresholded_image,
)

NCC_MODULE = "forensics_app.tools.cross_correlation"
CONFIGS_MODULE = "forensics_app.tools.matching_configs"

# Where the template is cut from the scene, and its size
TEMPLATE_X, TEMPLATE_Y = 20, 10
TEMPLATE_W, TEMPLATE_H = 12, 10


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def textured_scene(width: int = 64, height: int = 48) -> Image.Image:
    # Random texture so the template only matches where it was cut from
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return Image.fromarray(pixels, mode="L")


def cut_template(scene: Image.Image) -> Image.Image:
    return scene.crop(
        (TEMPLATE_X, TEMPLATE_Y, TEMPLATE_X + TEMPLATE_W, TEMPLATE_Y + TEMPLATE_H)
    )


def template_centre() -> tuple[int, int]:
    # pad_input=True puts the score at the centre of the template
    return TEMPLATE_Y + TEMPLATE_H // 2, TEMPLATE_X + TEMPLATE_W // 2


class TemplateFileMixin:
    """Save the template to a temporary file, as the file dialog expects."""

    def save_template(self, template: Image.Image) -> str:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = os.path.join(folder.name, "template.png")
        template.save(path)
        return path


class NccTests(unittest.TestCase):
    def test_ncc_has_image_size_and_peaks_at_template_centre(self) -> None:
        scene = textured_scene()
        correlation = compute_ncc(scene, cut_template(scene))

        self.assertEqual(correlation.shape, (scene.height, scene.width))
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        self.assertEqual(peak, template_centre())
        self.assertAlmostEqual(float(correlation.max()), 1.0, places=3)

    def test_ncc_ignores_brightness_and_contrast_of_the_template(self) -> None:
        scene = textured_scene()
        template = np.asarray(cut_template(scene), dtype=np.float32)
        # Half the contrast and a brightness offset, without clipping
        changed = Image.fromarray((template * 0.5 + 40).astype(np.uint8))

        correlation = compute_ncc(scene, changed)

        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        self.assertEqual(peak, template_centre())
        self.assertGreater(float(correlation.max()), 0.98)

    def test_ncc_values_stay_between_minus_one_and_one(self) -> None:
        scene = textured_scene()
        correlation = compute_ncc(scene, cut_template(scene))
        self.assertGreaterEqual(float(correlation.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(correlation.max()), 1.0 + 1e-5)

    def test_ncc_image_maps_minus_one_zero_and_one_to_black_grey_white(
        self,
    ) -> None:
        image = ncc_image(np.array([[-1.0, 0.0, 1.0]]))
        self.assertEqual(image.mode, "L")
        np.testing.assert_array_equal(np.asarray(image), [[0, 127, 255]])


class CrossCorrelationToolTests(TemplateFileMixin, unittest.TestCase):
    def run_tool(self, document: ImageDocument, path: str):
        with (
            mock.patch(f"{NCC_MODULE}.messagebox.showinfo"),
            mock.patch(f"{NCC_MODULE}.messagebox.showerror") as showerror,
            mock.patch(f"{NCC_MODULE}.filedialog.askopenfilename", return_value=path),
        ):
            return CrossCorrelationTool().run(None, document), showerror

    def test_tool_returns_ncc_image_without_mutating(self) -> None:
        scene = textured_scene().convert("RGB")
        document = make_document(scene)
        before = np.asarray(scene).copy()

        result, _showerror = self.run_tool(
            document, self.save_template(cut_template(scene))
        )

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, scene.size)
        # The brightest pixel of the NCC image is the matched template centre
        array = np.asarray(result.image)
        peak = np.unravel_index(np.argmax(array), array.shape)
        self.assertEqual(peak, template_centre())
        self.assertEqual(result.details["Template size"], f"{TEMPLATE_W}x{TEMPLATE_H}")
        self.assertEqual(result.details["Maxim correlation"], "1.000")
        np.testing.assert_array_equal(np.asarray(document.current), before)

    def test_tool_rejects_template_larger_than_image(self) -> None:
        document = make_document(textured_scene(16, 16).convert("RGB"))
        path = self.save_template(textured_scene(32, 32))

        result, showerror = self.run_tool(document, path)

        self.assertIsNone(result)
        showerror.assert_called_once()

    def test_tool_returns_none_when_cancelled(self) -> None:
        document = make_document(textured_scene().convert("RGB"))
        result, _showerror = self.run_tool(document, "")
        self.assertIsNone(result)


class MatchingConfigsHelperTests(unittest.TestCase):
    def test_enlarge_to_height_keeps_aspect_ratio(self) -> None:
        template = Image.new("L", (12, 10))
        enlarged = enlarge_to_height(template, 50)
        self.assertEqual(enlarged.size, (60, 50))

    def test_side_by_side_leaves_a_gap_between_images(self) -> None:
        left = Image.new("RGB", (10, 20), (255, 0, 0))
        right = Image.new("L", (30, 20), 0)

        panel = np.asarray(side_by_side([left, right]))

        self.assertEqual(panel.shape, (20, 10 + PANEL_GAP + 30, 3))
        np.testing.assert_array_equal(panel[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(panel[0, 10], [255, 255, 255])  # the gap
        np.testing.assert_array_equal(panel[0, 10 + PANEL_GAP], [0, 0, 0])

    def test_thresholded_image_darkens_and_paints_matches_red(self) -> None:
        original = Image.new("RGB", (9, 9), (100, 100, 100))
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True

        array = np.asarray(thresholded_image(original, mask))

        # The detected pixel grows to 3x3 so it stays visible
        np.testing.assert_array_equal(array[3:6, 3:6], np.full((3, 3, 3), [255, 0, 0]))
        np.testing.assert_array_equal(array[0, 0], [40, 40, 40])  # darkened to 40%
        self.assertEqual(int((array == [255, 0, 0]).all(axis=2).sum()), 9)


class MatchingConfigsToolTests(TemplateFileMixin, unittest.TestCase):
    def run_tool(
        self,
        document: ImageDocument,
        path: str,
        angle: int | None = 0,
        threshold: float | None = 0.9,
    ):
        with (
            mock.patch(f"{CONFIGS_MODULE}.messagebox.showinfo"),
            mock.patch(
                f"{CONFIGS_MODULE}.filedialog.askopenfilename", return_value=path
            ),
            mock.patch(f"{CONFIGS_MODULE}.simpledialog.askinteger", return_value=angle),
            mock.patch(
                f"{CONFIGS_MODULE}.simpledialog.askfloat", return_value=threshold
            ),
        ):
            return MatchingConfigsTool().run(None, document)

    def test_tool_shows_template_ncc_and_thresholded_panels(self) -> None:
        scene = textured_scene().convert("RGB")
        document = make_document(scene)
        before = np.asarray(scene).copy()

        result = self.run_tool(document, self.save_template(cut_template(scene)))

        template_width = enlarge_to_height(cut_template(scene), scene.height).width
        expected_width = template_width + 2 * scene.width + 2 * PANEL_GAP
        self.assertEqual(result.image.size, (expected_width, scene.height))

        # The red mark of the thresholded panel sits on the template centre
        thresholded = np.asarray(result.image)[:, -scene.width :]
        row, col = template_centre()
        np.testing.assert_array_equal(thresholded[row, col], [255, 0, 0])

        self.assertEqual(result.details["Matching regions"], 1)
        self.assertEqual(result.details["Best correlation"], "1.000")
        self.assertEqual(result.details["Rotation angle"], "0 degrees")
        np.testing.assert_array_equal(np.asarray(document.current), before)

    def test_lower_threshold_never_finds_fewer_pixels(self) -> None:
        scene = textured_scene().convert("RGB")
        document = make_document(scene)
        path = self.save_template(cut_template(scene))

        counts = [
            self.run_tool(document, path, threshold=threshold).details[
                "Pixels above threshold"
            ]
            for threshold in (0.9, 0.5, 0.2)
        ]

        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], counts[0])

    def test_threshold_above_best_correlation_finds_nothing(self) -> None:
        scene = textured_scene().convert("RGB")
        # A template from another texture never correlates perfectly
        other = textured_scene(80, 80).crop((50, 50, 62, 60))
        result = self.run_tool(
            make_document(scene), self.save_template(other), threshold=1.0
        )

        self.assertEqual(result.details["Matching regions"], 0)
        self.assertEqual(result.message, "No match above the selected threshold.")
        thresholded = np.asarray(result.image)[:, -scene.width :]
        self.assertFalse((thresholded == [255, 0, 0]).all(axis=2).any())

    def test_rotation_lowers_the_best_correlation(self) -> None:
        scene = textured_scene().convert("RGB")
        document = make_document(scene)
        path = self.save_template(cut_template(scene))

        straight = self.run_tool(document, path, angle=0)
        rotated = self.run_tool(document, path, angle=30)

        self.assertEqual(rotated.details["Rotation angle"], "30 degrees")
        self.assertLess(
            float(rotated.details["Best correlation"]),
            float(straight.details["Best correlation"]),
        )

    def test_tool_rejects_rotated_template_larger_than_image(self) -> None:
        document = make_document(textured_scene(16, 16).convert("RGB"))
        # Fits at 0 degrees, but rotating 45 degrees makes it about 21x21
        path = self.save_template(textured_scene(15, 15))

        result = self.run_tool(document, path, angle=45)

        self.assertIsNone(result.image)
        self.assertIn("must be smaller", result.message)

    def test_tool_returns_none_when_any_dialog_is_cancelled(self) -> None:
        scene = textured_scene().convert("RGB")
        document = make_document(scene)
        path = self.save_template(cut_template(scene))

        self.assertIsNone(self.run_tool(document, ""))
        self.assertIsNone(self.run_tool(document, path, angle=None))
        self.assertIsNone(self.run_tool(document, path, threshold=None))


if __name__ == "__main__":
    unittest.main()
