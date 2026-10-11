from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

FILTERS = ("Gaussian", "Median", "Sobel", "Prewitt", "Roberts", "Laplace")


def apply_filter(
    image: Image.Image,
    filter_name: str,
    sigma: float,
) -> Image.Image:

    gray = image.convert("L")
    img = np.array(gray, dtype=np.float32) / 255.0

    if filter_name == "Gaussian":
        result = filters.gaussian(img, sigma=sigma)

    elif filter_name == "Median":

        footprint = np.ones(
            (3, 3),
            dtype=bool,
        )

        result = filters.median(
            img,
            footprint=footprint,
        )

    elif filter_name == "Laplace":
        result = filters.laplace(img)

    elif filter_name == "Prewitt":
        result = filters.prewitt(img)

    elif filter_name == "Roberts":
        result = filters.roberts(img)

    else:
        result = filters.sobel(img)

    # Keep values between 0 and 1
    result = np.clip(result, 0, 1)

    # Convert values from 0-1 back to 0-255
    # and convert them to 8-bit unsigned integers
    result = (result * 255).astype(np.uint8)

    return Image.fromarray(result, mode="L")


def ask_filter(parent: tk.Misc) -> str | None:

    index = ask_choice(
        parent,
        "Filters",
        "Choose a filter",
        FILTERS,
    )
    return None if index is None else FILTERS[index]


def ask_sigma(parent: tk.Misc) -> float | None:
    return simpledialog.askfloat(
        "Gaussian filter",
        "Enter sigma:",
        parent=parent,
        minvalue=0.1,
        maxvalue=20.0,
        initialvalue=1.0,
    )


class FiltersTool(ForensicsTool):

    tool_id = "filters"
    title = "Filters"
    category = "Set 3 duties"
    description = "Apply different filters from skimage.filters."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None
        filter_name = ask_filter(parent)

        if filter_name is None:
            return None

        sigma = 1.0

        if filter_name == "Gaussian":
            selected_sigma = ask_sigma(parent)

            if selected_sigma is None:
                return None

            sigma = selected_sigma

        output = apply_filter(
            document.current,
            filter_name,
            sigma,
        )

        details = {
            "Operation": "Filter",
            "Filter": filter_name,
        }

        if filter_name == "Gaussian":
            details["Sigma"] = f"{sigma:.2f}"

        return ToolResult(
            image=output,
            message=f"Applied {filter_name} filter.",
            details=details,
        )
