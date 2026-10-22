"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretching import ContrastStretchingTool
from .convolution import Convolution
from .cross_correlation import CrossCorrelationTool
from .edges import CannyTool, EdgeOverlayTool
from .filters import FiltersTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .hog import HogTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .matching_configs import MatchingConfigsTool
from .person_detection import PersonDetectionTool
from .registry import ToolRegistry
from .sharpening import SharpeningTool
from .corner_detection import CornerDetectionTool
from .descriptor_extractor import DescriptorExtractorTool


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
            HogTool(),
            PersonDetectionTool(),
            CornerDetectionTool(),
            DescriptorExtractorTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
