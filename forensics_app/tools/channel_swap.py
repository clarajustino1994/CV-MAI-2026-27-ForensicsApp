"""Exchange two colour channels and show the result beside the original."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .channel_split import render_panels

CHANNEL_NAMES = {0: "R", 1: "G", 2: "B"}


def swap_channels(image: Image.Image, first: int = 0, second: int = 2) -> Image.Image:
    img = np.array(image.convert("RGB"))
    # [;] fancy indexing to avoid tmp
    img[:, :, [first, second]] = img[:, :, [second, first]]
    return Image.fromarray(img)


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Set 2 duties"
    description = "Exchange the red and blue channels of the image."

    # Channel indices follow the array layout: 0 = R, 1 = G, 2 = B. 
    first_channel = 0
    second_channel = 2

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  
        original = document.current.convert("RGB")
        swapped = swap_channels(original, self.first_channel, self.second_channel)
        name = document.path.stem if document.path else "Original"
        output = render_panels(
            [
                (f"{name} image", np.asarray(original)),
                ("Transformed image", np.asarray(swapped)),
            ]
        )
        first, second = (
            CHANNEL_NAMES[self.first_channel],
            CHANNEL_NAMES[self.second_channel],
        )
        pair = f"{first} <-> {second}"
        before = np.asarray(original).reshape(-1, 3).mean(axis=0)
        after = np.asarray(swapped).reshape(-1, 3).mean(axis=0)
        return ToolResult(
            image=output,
            message=f"Swapped the {pair} channels.",
            details={
                "Operation": "Channel swap",
                "Swapped": pair,
                "Panels": "Original, Transformed",
                "RGB mean before": ", ".join(f"{v:.1f}" for v in before),
                "RGB mean after": ", ".join(f"{v:.1f}" for v in after),
            },
        )
