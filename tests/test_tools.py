import unittest
from unittest import mock

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.channel_swap import ChannelSwapTool, swap_channels
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.masking import MaskingTool, apply_mask
from forensics_app.tools.registry import ToolRegistry


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


if __name__ == "__main__":
    unittest.main()
