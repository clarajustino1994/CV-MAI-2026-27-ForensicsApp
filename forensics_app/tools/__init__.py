"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretching import ContrastStretchingTool
from .convolution import Convolution
from .edges import CannyTool, EdgeOverlayTool
from .filters import FiltersTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .sharpening import SharpeningTool
from .cross_correlation import CrossCorrelationTool
from .matching_configs import MatchingConfigsTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            ContrastStretchingTool(),
            HistogramTool(),
            MaskingTool(),
            Convolution(),
            FiltersTool(),
            CannyTool(),
            EdgeOverlayTool(),
            SharpeningTool(),
            CrossCorrelationTool(),
            MatchingConfigsTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
