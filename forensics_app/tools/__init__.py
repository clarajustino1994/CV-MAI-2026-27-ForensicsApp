"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretching import ContrastStretchingTool
from .convolution import Convolution
from .corner_detection import CornerDetectionTool
from .cross_correlation import CrossCorrelationTool
from .descriptor_extractor import DescriptorExtractorTool
from .edges import CannyTool, EdgeOverlayTool
from .feature_matching import FeatureMatchingTool
from .filters import FiltersTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .hog import HogTool
from .image_info import ImageInfoTool
from .image_lookup import ImageLookupTool
from .masking import MaskingTool
from .matching_configs import MatchingConfigsTool
from .person_detection import PersonDetectionTool
from .registry import ToolRegistry
from .sharpening import SharpeningTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            # Set 5
            CornerDetectionTool(),
            DescriptorExtractorTool(),
            FeatureMatchingTool(),
            ImageLookupTool(),
            # Set 4
            CrossCorrelationTool(),
            MatchingConfigsTool(),
            HogTool(),
            PersonDetectionTool(),
            # Set 3
            Convolution(),
            FiltersTool(),
            CannyTool(),
            EdgeOverlayTool(),
            SharpeningTool(),
            # Set 2
            ChannelSplitTool(),
            ChannelSwapTool(),
            ContrastStretchingTool(),
            HistogramTool(),
            MaskingTool(),
            # Starter tools
            ImageInfoTool(),
            GrayscaleTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
