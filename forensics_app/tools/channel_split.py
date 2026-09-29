"""Show the working image beside its red, green and blue channels."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

BACKGROUND = "white"
TEXT = "black"


def split_channels(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the R, G and B planes of ``image`` as 2-D uint8 arrays."""
    img = np.asarray(image.convert("RGB"))
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


def render_panels(
    panels: list[tuple[str, np.ndarray]], columns: int = 2
) -> Image.Image:
    """Lay ``(title, array)`` pairs in a grid, like a matplotlib subplot figure."""
    height, width = panels[0][1].shape[:2]
    rows = math.ceil(len(panels) / columns)
    gap = max(8, width // 50)
    title_height = max(24, height // 12)
    cell_height = title_height + height + gap
    canvas = Image.new(
        "RGB",
        (columns * (width + gap) + gap, rows * cell_height + gap),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    font = _title_font(int(title_height * 0.6))
    for index, (title, array) in enumerate(panels):
        left = gap + (index % columns) * (width + gap)
        top = gap + (index // columns) * cell_height
        x0, y0, x1, y1 = draw.textbbox((0, 0), title, font=font)
        text_x = left + (width - (x1 - x0)) // 2 - x0
        text_y = top + (title_height - (y1 - y0)) // 2 - y0
        draw.text((text_x, text_y), title, fill=TEXT, font=font)
        canvas.paste(Image.fromarray(array), (left, top + title_height))
    return canvas


def _title_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 only ships a fixed-size bitmap font
        return ImageFont.load_default()


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split"
    category = "Set 2 duties"
    description = "Show the image beside its red, green and blue channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        original = document.current.convert("RGB")
        R, G, B = split_channels(original)
        name = document.path.stem if document.path else "Original"
        output = render_panels(
            [
                (f"{name} image", np.asarray(original)),
                ("Red channel", R),
                ("Green channel", G),
                ("Blue channel", B),
            ]
        )
        return ToolResult(
            image=output,
            message="Split the image into red, green and blue channels.",
            details={
                "Operation": "Channel split",
                "Panels": "Original, R, G, B",
                "Red mean": f"{R.mean():.1f}",
                "Green mean": f"{G.mean():.1f}",
                "Blue mean": f"{B.mean():.1f}",
            },
        )
