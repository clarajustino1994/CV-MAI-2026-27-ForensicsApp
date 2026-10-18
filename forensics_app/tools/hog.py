"""Set 4.3: extract a HOG descriptor and visualise it."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import exposure, feature

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


@dataclass(frozen=True)
class HogParams:
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2


def to_array(image: Image.Image) -> np.ndarray:
    # RGB values between 0 and 1; HOG keeps the strongest gradient per pixel
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def hog_descriptor(image: Image.Image | np.ndarray, params: HogParams) -> np.ndarray:
    array = to_array(image) if isinstance(image, Image.Image) else image
    return feature.hog(
        array,
        orientations=params.orientations,
        pixels_per_cell=(params.pixels_per_cell, params.pixels_per_cell),
        cells_per_block=(params.cells_per_block, params.cells_per_block),
        channel_axis=-1,
    )


def compute_hog(
    image: Image.Image, params: HogParams
) -> tuple[np.ndarray, Image.Image]:
    descriptor, hog_image = feature.hog(
        to_array(image),
        orientations=params.orientations,
        pixels_per_cell=(params.pixels_per_cell, params.pixels_per_cell),
        cells_per_block=(params.cells_per_block, params.cells_per_block),
        visualize=True,
        channel_axis=-1,
    )

    # The HOG drawing is very dark, so stretch it to the full 0..255 range
    rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255))
    return descriptor, Image.fromarray(rescaled.astype(np.uint8), mode="L")


def ask_hog_params(
    parent: tk.Misc, title: str, default: HogParams = HogParams()
) -> HogParams | None:
    orientations = simpledialog.askinteger(
        title,
        "Number of orientations:",
        parent=parent,
        initialvalue=default.orientations,
        minvalue=1,
        maxvalue=36,
    )
    if orientations is None:
        return None

    pixels_per_cell = simpledialog.askinteger(
        title,
        "Pixels per cell (cell size in pixels):",
        parent=parent,
        initialvalue=default.pixels_per_cell,
        minvalue=2,
        maxvalue=64,
    )
    if pixels_per_cell is None:
        return None

    cells_per_block = simpledialog.askinteger(
        title,
        "Cells per block:",
        parent=parent,
        initialvalue=default.cells_per_block,
        minvalue=1,
        maxvalue=8,
    )
    if cells_per_block is None:
        return None

    return HogParams(orientations, pixels_per_cell, cells_per_block)


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG extraction"
    category = "Set 4"
    description = "Extract and visualise the Histogram of Oriented Gradients."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        params = ask_hog_params(parent, "HOG extraction")
        if params is None:
            return None

        # One block must fit inside the image
        block_size = params.pixels_per_cell * params.cells_per_block
        width, height = document.current.size
        if block_size > min(width, height):
            return ToolResult(
                message=f"One block ({block_size}x{block_size} px) does not fit "
                "in the image. Use smaller cells or fewer cells per block."
            )

        descriptor, hog_image = compute_hog(document.current, params)

        return ToolResult(
            image=hog_image,
            message=f"HOG computed with {params.orientations} orientations.",
            details={
                "Operation": "HOG extraction",
                "Orientations": params.orientations,
                "Pixels per cell": (
                    f"{params.pixels_per_cell}x{params.pixels_per_cell}"
                ),
                "Cells per block": (
                    f"{params.cells_per_block}x{params.cells_per_block}"
                ),
                "Descriptor length": descriptor.size,
            },
        )
