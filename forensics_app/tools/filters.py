from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

SIGMA_FILTERS = ("Gaussian", "Laplace")
FILTERS = ("Gaussian", "Median", "Sobel", "Prewitt", "Roberts", "Laplace")


def apply_filter(
    image: Image.Image,
    filter_name: str,
    sigma: float,
    kernel_size: int = 3,
) -> Image.Image:

    gray = image.convert("L")
    img = np.array(gray, dtype=np.float32) / 255.0

    if filter_name == "Gaussian":
        result = filters.gaussian(img, sigma=sigma)

    elif filter_name == "Median":

        footprint = np.ones(
            (kernel_size, kernel_size),
            dtype=bool,
        )

        result = filters.median(
            img,
            footprint=footprint,
        )

    elif filter_name == "Laplace":
        # Smooth first so the Laplacian does not amplify noise
        smoothed = filters.gaussian(img, sigma=sigma)
        result = np.abs(filters.laplace(smoothed))

        # Normalizing
        peak = result.max()
        if peak > 0:
            result = result / peak

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


def ask_sigma(parent: tk.Misc, title: str) -> float | None:
    return simpledialog.askfloat(
        title,
        "Enter sigma:",
        parent=parent,
        minvalue=0.1,
        maxvalue=20.0,
        initialvalue=1.0,
    )


def ask_kernel_size(parent: tk.Misc) -> int | None:
    return simpledialog.askinteger(
        "Median filter",
        "Enter kernel size:",
        parent=parent,
        minvalue=1,
        maxvalue=31,
        initialvalue=3,
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

        if filter_name in SIGMA_FILTERS:
            selected_sigma = ask_sigma(parent, f"{filter_name} filter")

            if selected_sigma is None:
                return None

            sigma = selected_sigma

        kernel_size = 3

        if filter_name == "Median":
            selected_kernel_size = ask_kernel_size(parent)

            if selected_kernel_size is None:
                return None

            kernel_size = selected_kernel_size

        output = apply_filter(
            document.current,
            filter_name,
            sigma,
            kernel_size,
        )

        details = {
            "Operation": "Filter",
            "Filter": filter_name,
        }

        if filter_name in SIGMA_FILTERS:
            details["Sigma"] = f"{sigma:.2f}"

        if filter_name == "Median":
            details["Kernel size"] = f"{kernel_size}x{kernel_size}"

        return ToolResult(
            image=output,
            message=f"Applied {filter_name} filter.",
            details=details,
        )
