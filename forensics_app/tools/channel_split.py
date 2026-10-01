"""Show one colour channel of the working image, picked by the user."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

CHANNELS = ("Red", "Green", "Blue")


def split_channels(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the R, G and B planes of ``image`` as 2-D uint8 arrays."""
    img = np.asarray(image.convert("RGB"))
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


def ask_channel(parent: tk.Misc) -> int | None:
    return ask_choice(
        parent, "Channel split", "Which channel do you want to show?", CHANNELS
    )


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set 2 duties"
    description = "Show the red, green or blue channel of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        channel = ask_channel(parent)
        if channel is None:
            return None
        plane = split_channels(document.current)[channel]
        channel_name = CHANNELS[channel]
        return ToolResult(
            image=Image.fromarray(plane),
            message=f"Showing the {channel_name.lower()} channel.",
            details={
                "Operation": "Channel split",
                "Channel": channel_name,
                f"{channel_name} mean": f"{plane.mean():.1f}",
            },
        )
