from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

# Strength of the sharpening: how much of the detail is added back
AMOUNT = 1.0


def apply_sharpening(image: Image.Image, radius: float) -> Image.Image:

    mode = "L" if image.mode == "L" else "RGB"
    img = np.array(image.convert(mode), dtype=np.float32) / 255.0

    result = filters.unsharp_mask(
        img,
        radius=radius,
        amount=AMOUNT,
        channel_axis=None if mode == "L" else 2,
    )

    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    return Image.fromarray(result, mode=mode)


def ask_radius(parent: tk.Misc) -> float | None:
    return simpledialog.askfloat(
        "Sharpening",
        "Enter radius:",
        parent=parent,
        minvalue=0.1,
        maxvalue=20.0,
        initialvalue=1.0,
    )


class SharpeningTool(ForensicsTool):

    tool_id = "sharpening"
    title = "Sharpening"
    category = "Set 3 duties"
    description = "Sharpen the image with an unsharp mask of configurable radius."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None
        radius = ask_radius(parent)

        # Stop if the user cancels
        if radius is None:
            return None

        output = apply_sharpening(document.current, radius)

        return ToolResult(
            image=output,
            message=f"Sharpened image (radius={radius:.2f}).",
            details={
                "Operation": "Sharpening",
                "Method": "Unsharp mask",
                "Radius": f"{radius:.2f}",
                "Amount": f"{AMOUNT:.2f}",
            },
        )
