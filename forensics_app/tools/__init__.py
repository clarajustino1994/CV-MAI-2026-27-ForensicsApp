"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .contrast_stretching import ContrastStretchingTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .histogram import HistogramTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ContrastStretchingTool(),
            HistogramTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
