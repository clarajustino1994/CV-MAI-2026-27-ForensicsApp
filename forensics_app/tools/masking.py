"""Paste a coat (optionally textured) onto the working image using a mask."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

NEXT_STEPS = ("Apply texture", "Change coat")
OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def apply_mask(
    model: Image.Image, coat: Image.Image, texture: Image.Image | None = None
) -> tuple[Image.Image, np.ndarray]:
    size = model.size
    base = np.array(model.convert("RGB"))
    # Inputs come from separate files, so align them to the model before indexing.
    coat_img = np.array(coat.convert("RGB").resize(size))
    msk = coat_img > 0
    source = coat_img
    if texture is not None:
        source = np.array(texture.convert("RGB").resize(size))
    base[msk] = source[msk]
    return Image.fromarray(base), msk


def _pick_image(parent: tk.Misc, kind: str) -> Image.Image | None:
    prompt = f"Choose the {kind} image."
    if ask_choice(parent, "Masking", prompt, (f"Choose {kind}",)) is None:
        return None
    filename = filedialog.askopenfilename(
        parent=parent, title=f"Open {kind} image", filetypes=OPEN_TYPES
    )
    return Image.open(filename) if filename else None


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Set 2 duties"
    description = "Dress the image with a coat, plain or textured, using a mask."

    def __init__(self) -> None:
        self._coat: Image.Image | None = None

    def reset(self) -> None:
        self._coat = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        step = 1
        if self._coat is not None:
            step = ask_choice(
                parent, "Masking", "Coat loaded. What do you want to do?", NEXT_STEPS
            )
            if step is None:
                return None
        if step == 1:
            coat = _pick_image(parent, "coat")
            if coat is None:
                return None
            self._coat = coat
            return self._result(*apply_mask(document.current, coat), "Coat")
        texture = _pick_image(parent, "texture")
        if texture is None:
            return None
        return self._result(
            *apply_mask(document.current, self._coat, texture), "Textured coat"
        )

    @staticmethod
    def _result(image: Image.Image, msk: np.ndarray, mode: str) -> ToolResult:
        return ToolResult(
            image=image,
            message=f"Applied the {mode.lower()} with a mask.",
            details={
                "Operation": "Masking",
                "Mode": mode,
                "Masked pixels": f"{int(msk.any(axis=2).sum())}",
            },
        )
