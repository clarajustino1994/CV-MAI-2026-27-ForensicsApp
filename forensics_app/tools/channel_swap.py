"""Exchange two colour channels of the working image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

CHANNEL_NAMES = {0: "R", 1: "G", 2: "B"}
# Channel indices follow the array layout: 0 = R, 1 = G, 2 = B.
PAIRS = ((0, 1), (0, 2), (1, 2))


def swap_channels(image: Image.Image, first: int = 0, second: int = 2) -> Image.Image:
    img = np.array(image.convert("RGB"))
    # [;] fancy indexing to avoid tmp
    img[:, :, [first, second]] = img[:, :, [second, first]]
    return Image.fromarray(img)


def ask_pair(parent: tk.Misc) -> tuple[int, int] | None:
    labels = [f"{CHANNEL_NAMES[a]} <-> {CHANNEL_NAMES[b]}" for a, b in PAIRS]
    index = ask_choice(
        parent, "Channel swap", "Which channels do you want to swap?", labels
    )
    return None if index is None else PAIRS[index]


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap"
    category = "Set 2 duties"
    description = "Exchange two colour channels of the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        pair = ask_pair(parent)
        if pair is None:
            return None
        first_channel, second_channel = pair
        original = document.current.convert("RGB")
        swapped = swap_channels(original, first_channel, second_channel)
        label = f"{CHANNEL_NAMES[first_channel]} <-> {CHANNEL_NAMES[second_channel]}"
        before = np.asarray(original).reshape(-1, 3).mean(axis=0)
        after = np.asarray(swapped).reshape(-1, 3).mean(axis=0)
        return ToolResult(
            image=swapped,
            message=f"Swapped the {label} channels.",
            details={
                "Operation": "Channel swap",
                "Swapped": label,
                "RGB mean before": ", ".join(f"{v:.1f}" for v in before),
                "RGB mean after": ", ".join(f"{v:.1f}" for v in after),
            },
        )
