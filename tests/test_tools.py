import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
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

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])


if __name__ == "__main__":
    unittest.main()
