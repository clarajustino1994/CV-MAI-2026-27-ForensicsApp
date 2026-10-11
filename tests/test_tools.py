import unittest
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.channel_swap import ChannelSwapTool, swap_channels
from forensics_app.tools.contrast_stretching import ContrastStretchingTool
from forensics_app.tools.convolution import (
    Convolution,
    apply_convolution,
    max_kernel_size,
)
from forensics_app.tools.edges import (
    CannyTool,
    EdgeOverlayTool,
    apply_canny,
    overlay_edges,
)
from forensics_app.tools.filters import FiltersTool, apply_filter
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.masking import MaskingTool, apply_mask
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.sharpening import SharpeningTool, apply_sharpening


def make_document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def step_image(mode: str = "L") -> Image.Image:
    # Left half dark, right half bright: a single vertical edge at x=10
    pixels = np.full((20, 20), 100, dtype=np.uint8)
    pixels[:, 10:] = 150
    return Image.fromarray(pixels, mode="L").convert(mode)


def square_image() -> Image.Image:
    # White 10x10 square in the middle of a black 30x30 image
    pixels = np.zeros((30, 30), dtype=np.uint8)
    pixels[10:20, 10:20] = 255
    return Image.fromarray(pixels, mode="L")


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_split_channels_returns_three_planes(self) -> None:
        image = Image.new("RGB", (2, 3), (10, 20, 30))
        R, G, B = split_channels(image)
        self.assertEqual(R.shape, (3, 2))
        self.assertEqual((int(R[0, 0]), int(G[0, 0]), int(B[0, 0])), (10, 20, 30))
        gray = Image.new("L", (2, 2), 7)  # non-RGB input still yields 3 planes
        self.assertEqual(int(split_channels(gray)[2][0, 0]), 7)

    def test_channel_split_shows_chosen_channel_without_mutating(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), (10, 20, 30))
        with mock.patch(
            "forensics_app.tools.channel_split.ask_channel", return_value=1
        ):
            result = ChannelSplitTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, (40, 30))
        self.assertEqual(result.image.getpixel((0, 0)), 20)
        self.assertEqual(document.current.mode, "RGB")
        self.assertEqual(result.details["Channel"], "Green")
        self.assertEqual(result.details["Green mean"], "20.0")

    def test_channel_split_returns_none_when_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        with mock.patch(
            "forensics_app.tools.channel_split.ask_channel", return_value=None
        ):
            self.assertIsNone(ChannelSplitTool().run(None, document))

    def test_swap_channels_exchanges_red_and_blue_by_default(self) -> None:
        image = Image.new("RGB", (2, 3), (10, 20, 30))
        self.assertEqual(swap_channels(image).getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(swap_channels(image, 0, 1).getpixel((0, 0)), (20, 10, 30))
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))  # input untouched
        gray = Image.new("L", (2, 2), 7)  # non-RGB input is promoted to RGB first
        self.assertEqual(swap_channels(gray).getpixel((0, 0)), (7, 7, 7))

    def test_channel_swap_returns_swapped_image_without_mutating(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), (10, 20, 30))
        with mock.patch(
            "forensics_app.tools.channel_swap.ask_pair", return_value=(0, 2)
        ):
            result = ChannelSwapTool().run(None, document)
        self.assertEqual(result.image.size, (40, 30))
        self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(document.current.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result.details["Swapped"], "R <-> B")
        self.assertEqual(result.details["RGB mean after"], "30.0, 20.0, 10.0")

    def test_channel_swap_returns_none_when_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        with mock.patch("forensics_app.tools.channel_swap.ask_pair", return_value=None):
            self.assertIsNone(ChannelSwapTool().run(None, document))

    def test_apply_mask_copies_coat_only_where_non_zero(self) -> None:
        model = Image.new("RGB", (4, 2), (50, 50, 50))
        coat = Image.new("RGB", (4, 2), (0, 0, 0))
        coat.putpixel((1, 0), (200, 180, 40))
        result, msk = apply_mask(model, coat)
        self.assertEqual(result.getpixel((1, 0)), (200, 180, 40))
        self.assertEqual(result.getpixel((0, 0)), (50, 50, 50))
        self.assertEqual(int(msk.any(axis=2).sum()), 1)
        self.assertEqual(model.getpixel((1, 0)), (50, 50, 50))  # input untouched

    def test_apply_mask_uses_texture_inside_coat(self) -> None:
        model = Image.new("RGB", (4, 2), (50, 50, 50))
        coat = Image.new("RGB", (2, 1), (0, 0, 0))  # different size gets resized
        coat.putpixel((0, 0), (200, 180, 40))
        texture = Image.new("RGB", (8, 8), (220, 40, 30))
        result, _ = apply_mask(model, coat, texture)
        self.assertEqual(result.getpixel((0, 0)), (220, 40, 30))
        self.assertEqual(result.getpixel((3, 1)), (50, 50, 50))

    def test_masking_asks_for_coat_first_then_offers_texture(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 2), (50, 50, 50))
        coat = Image.new("RGB", (4, 2), (0, 0, 0))
        coat.putpixel((1, 0), (200, 180, 40))
        texture = Image.new("RGB", (4, 2), (220, 40, 30))
        tool = MaskingTool()
        module = "forensics_app.tools.masking"
        with (
            mock.patch(f"{module}.ask_choice") as ask,
            mock.patch(f"{module}._pick_image", return_value=coat),
        ):
            result = tool.run(None, document)
        ask.assert_not_called()  # no coat yet, so no texture option
        self.assertEqual(result.details["Mode"], "Coat")
        document.current = result.image
        with (
            mock.patch(f"{module}.ask_choice", return_value=0),
            mock.patch(f"{module}._pick_image", return_value=texture),
        ):
            result = tool.run(None, document)
        self.assertEqual(result.details["Mode"], "Textured coat")
        self.assertEqual(result.image.getpixel((1, 0)), (220, 40, 30))
        self.assertEqual(result.image.getpixel((0, 0)), (50, 50, 50))

    def test_masking_reset_forgets_the_coat(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 2), (50, 50, 50))
        coat = Image.new("RGB", (4, 2), (200, 180, 40))
        tool = MaskingTool()
        module = "forensics_app.tools.masking"
        with mock.patch(f"{module}._pick_image", return_value=coat):
            tool.run(None, document)
        tool.reset()
        with (
            mock.patch(f"{module}.ask_choice") as ask,
            mock.patch(f"{module}._pick_image", return_value=coat),
        ):
            tool.run(None, document)
        ask.assert_not_called()  # back to step 1: asks for a coat again

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_max_kernel_size_is_largest_odd_fitting_the_image(self) -> None:
        self.assertEqual(max_kernel_size(Image.new("L", (618, 476))), 475)
        self.assertEqual(max_kernel_size(Image.new("L", (9, 20))), 9)
        self.assertEqual(max_kernel_size(Image.new("L", (1, 1))), 1)

    def test_convolution_matches_mean_kernel_and_rejects_oversized(self) -> None:
        import numpy as np
        from scipy import ndimage

        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, (30, 40), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="L")
        expected = ndimage.convolve(
            pixels.astype(np.float32), np.ones((5, 5), np.float32) / 25, mode="reflect"
        )
        result = np.array(apply_convolution(image, 5), dtype=np.int16)
        self.assertLessEqual(int(np.abs(result - np.rint(expected)).max()), 1)
        self.assertEqual(apply_convolution(image, 1).getpixel((3, 4)), pixels[4, 3])
        apply_convolution(image, 29)  # largest allowed for a 40x30 image
        with self.assertRaises(ValueError):
            apply_convolution(image, 31)

    def test_convolution_tool_limits_dialog_to_image_size(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (618, 476), "red")
        with mock.patch(
            "forensics_app.tools.convolution.ask_kernel_size", return_value=475
        ) as ask:
            result = Convolution().run(None, document)
        ask.assert_called_once_with(None, 475)
        self.assertEqual(result.image.size, (618, 476))
        self.assertEqual(result.details["Kernel size"], "475x475")

    def test_contrast_stretching_maps_percentiles_to_full_range(self) -> None:
        pixels = np.full((10, 10), 50, dtype=np.uint8)
        pixels[:, 5:] = 150
        document = make_document(Image.fromarray(pixels, mode="L").convert("RGB"))
        result = ContrastStretchingTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.getpixel((0, 0)), 0)
        self.assertEqual(result.image.getpixel((9, 0)), 255)
        self.assertEqual(document.current.mode, "RGB")
        self.assertEqual(result.details["Lower percentile"], 5)
        self.assertEqual(result.details["Upper percentile"], 95)

    def test_contrast_stretching_flat_image_is_left_unchanged(self) -> None:
        document = make_document(Image.new("L", (8, 8), 80))
        result = ContrastStretchingTool().run(None, document)
        self.assertEqual(result.message, "Contrast stretching could not be applied.")
        self.assertEqual(result.image.getpixel((0, 0)), 80)

    def test_apply_filter_smoothing_keeps_flat_image(self) -> None:
        image = Image.new("RGB", (12, 10), (90, 90, 90))
        for name in ("Gaussian", "Median"):
            result = apply_filter(image, name, sigma=2.0)
            self.assertEqual(result.mode, "L")
            self.assertEqual(result.size, (12, 10))
            pixels = np.array(result, dtype=np.int16)
            self.assertLessEqual(int(np.abs(pixels - 90).max()), 1, name)

    def test_apply_filter_median_removes_salt_noise(self) -> None:
        pixels = np.full((9, 9), 40, dtype=np.uint8)
        pixels[4, 4] = 255
        result = apply_filter(Image.fromarray(pixels, mode="L"), "Median", 1.0)
        self.assertLessEqual(abs(result.getpixel((4, 4)) - 40), 1)

    def test_apply_filter_gradients_respond_only_at_the_edge(self) -> None:
        image = step_image()
        for name in ("Sobel", "Prewitt", "Roberts", "Laplace"):
            pixels = np.array(apply_filter(image, name, 1.0))
            self.assertEqual(int(pixels[:, :5].max()), 0, name)  # flat region
            self.assertEqual(int(pixels[:, 15:].max()), 0, name)
            self.assertGreater(int(pixels[:, 8:12].max()), 0, name)  # edge

    def test_filters_tool_asks_sigma_only_for_gaussian(self) -> None:
        document = make_document(step_image("RGB"))
        module = "forensics_app.tools.filters"
        with (
            mock.patch(f"{module}.ask_filter", return_value="Gaussian"),
            mock.patch(f"{module}.ask_sigma", return_value=2.5) as ask,
        ):
            result = FiltersTool().run(None, document)
        ask.assert_called_once()
        self.assertEqual(result.details["Filter"], "Gaussian")
        self.assertEqual(result.details["Sigma"], "2.50")
        self.assertEqual(document.current.mode, "RGB")
        with (
            mock.patch(f"{module}.ask_filter", return_value="Sobel"),
            mock.patch(f"{module}.ask_sigma") as ask,
        ):
            result = FiltersTool().run(None, document)
        ask.assert_not_called()
        self.assertNotIn("Sigma", result.details)

    def test_filters_tool_returns_none_when_cancelled(self) -> None:
        document = make_document(step_image())
        module = "forensics_app.tools.filters"
        with mock.patch(f"{module}.ask_filter", return_value=None):
            self.assertIsNone(FiltersTool().run(None, document))
        with (
            mock.patch(f"{module}.ask_filter", return_value="Gaussian"),
            mock.patch(f"{module}.ask_sigma", return_value=None),
        ):
            self.assertIsNone(FiltersTool().run(None, document))

    def test_apply_canny_marks_square_border_only(self) -> None:
        pixels = np.array(apply_canny(square_image(), sigma=1.0))
        self.assertEqual(set(np.unique(pixels)), {0, 255})
        self.assertEqual(int(pixels[:5, :].max()), 0)  # background
        self.assertEqual(int(pixels[14:16, 14:16].max()), 0)  # inside square
        self.assertEqual(int(pixels[8:12, 15].max()), 255)  # top border

    def test_overlay_edges_paints_edges_red_on_original(self) -> None:
        image = square_image()
        for detector in ("Canny", "Sobel", "Prewitt", "Roberts"):
            result = overlay_edges(image, detector, sigma=1.0)
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 0), detector)
            self.assertEqual(result.getpixel((15, 15)), (255, 255, 255), detector)
            red = np.all(np.array(result) == (255, 0, 0), axis=2)
            self.assertGreater(int(red.sum()), 0, detector)
        self.assertEqual(image.mode, "L")  # input untouched

    def test_canny_tool_returns_edges_without_mutating(self) -> None:
        document = make_document(square_image().convert("RGB"))
        with mock.patch("forensics_app.tools.edges.ask_sigma", return_value=1.5):
            result = CannyTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.details["Sigma"], "1.50")
        self.assertEqual(document.current.mode, "RGB")
        with mock.patch("forensics_app.tools.edges.ask_sigma", return_value=None):
            self.assertIsNone(CannyTool().run(None, document))

    def test_edge_overlay_asks_sigma_only_for_canny(self) -> None:
        document = make_document(square_image())
        module = "forensics_app.tools.edges"
        with (
            mock.patch(f"{module}.ask_choice", return_value=0),
            mock.patch(f"{module}.ask_sigma", return_value=2.0) as ask,
        ):
            result = EdgeOverlayTool().run(None, document)
        ask.assert_called_once()
        self.assertEqual(result.details["Edge detector"], "Canny")
        self.assertEqual(result.details["Sigma"], "2.00")
        with (
            mock.patch(f"{module}.ask_choice", return_value=1),
            mock.patch(f"{module}.ask_sigma") as ask,
        ):
            result = EdgeOverlayTool().run(None, document)
        ask.assert_not_called()
        self.assertEqual(result.details["Edge detector"], "Sobel")
        self.assertEqual(result.details["Threshold"], "Otsu")

    def test_edge_overlay_returns_none_when_cancelled(self) -> None:
        document = make_document(square_image())
        module = "forensics_app.tools.edges"
        with mock.patch(f"{module}.ask_choice", return_value=None):
            self.assertIsNone(EdgeOverlayTool().run(None, document))
        with (
            mock.patch(f"{module}.ask_choice", return_value=0),
            mock.patch(f"{module}.ask_sigma", return_value=None),
        ):
            self.assertIsNone(EdgeOverlayTool().run(None, document))

    def test_apply_sharpening_boosts_contrast_at_the_edge(self) -> None:
        result = np.array(apply_sharpening(step_image(), radius=1.0), np.int16)
        self.assertLess(int(result[10, 9]), 100)  # dark side gets darker
        self.assertGreater(int(result[10, 10]), 150)  # bright side gets brighter
        self.assertLessEqual(abs(int(result[10, 0]) - 100), 1)  # flat stays flat

    def test_apply_sharpening_keeps_gray_or_converts_to_rgb(self) -> None:
        self.assertEqual(apply_sharpening(step_image("L"), 1.0).mode, "L")
        self.assertEqual(apply_sharpening(step_image("RGB"), 1.0).mode, "RGB")
        self.assertEqual(apply_sharpening(step_image("RGBA"), 1.0).mode, "RGB")

    def test_sharpening_tool_returns_image_without_mutating(self) -> None:
        image = step_image("RGB")
        document = make_document(image)
        module = "forensics_app.tools.sharpening"
        with mock.patch(f"{module}.ask_radius", return_value=2.0):
            result = SharpeningTool().run(None, document)
        self.assertEqual(result.image.size, (20, 20))
        self.assertEqual(result.details["Radius"], "2.00")
        self.assertEqual(document.current.getpixel((9, 0)), (100, 100, 100))
        with mock.patch(f"{module}.ask_radius", return_value=None):
            self.assertIsNone(SharpeningTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
