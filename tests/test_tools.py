import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.channel_swap import ChannelSwapTool, swap_channels
from forensics_app.tools.grayscale import GrayscaleTool
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

    def test_channel_split_renders_four_panels_without_mutating(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), (10, 20, 30))
        result = ChannelSplitTool().run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, 2 * 40)  # 2x2 grid, not one row
        self.assertLess(result.image.width, 4 * 40)
        self.assertGreater(result.image.height, 2 * 30)
        self.assertEqual(document.current.size, (40, 30))
        self.assertEqual(result.details["Red mean"], "10.0")

    def test_swap_channels_exchanges_red_and_blue_by_default(self) -> None:
        image = Image.new("RGB", (2, 3), (10, 20, 30))
        self.assertEqual(swap_channels(image).getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(swap_channels(image, 0, 1).getpixel((0, 0)), (20, 10, 30))
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))  # input untouched
        gray = Image.new("L", (2, 2), 7)  # non-RGB input is promoted to RGB first
        self.assertEqual(swap_channels(gray).getpixel((0, 0)), (7, 7, 7))

    def test_channel_swap_renders_two_panels_without_mutating(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), (10, 20, 30))
        result = ChannelSwapTool().run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, 2 * 40)  # one row of two panels
        self.assertLess(result.image.width, 3 * 40)
        split = ChannelSplitTool().run(None, document)  # 2x2 grid, so twice as tall
        self.assertLess(result.image.height, split.image.height)
        self.assertEqual(document.current.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result.details["Swapped"], "R <-> B")
        self.assertEqual(result.details["RGB mean after"], "30.0, 20.0, 10.0")

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
