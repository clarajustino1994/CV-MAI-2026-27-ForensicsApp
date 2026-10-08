from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

FILTERS = (
    "Gaussian",
    "Median",
    "Sobel",
)

def apply_filter(
        image: Image.Image,
        filter_name: str,
        sigma: float,
) -> Image.Image:

    # Convert the image to grayscale
    gray = image.convert("L")

    # Convert the image to a Numpy array
    # and normalize pixel values from 0.255 to 0-1
    img = np.array(gray, dtype=np.float32) / 255.0

    # Apply the selected filter
    if filter_name == "Gaussian":
        result = filters.gaussian(
            img,
            sigma=sigma
        )

    elif filter_name == "Median":

        # Create a 3x3 neighborhood
        footprint = np.ones(
            (3,3),
            dtype=bool,
        )

        result = filters.median(
            img,
            footprint=footprint,
        )

    else:

        # Apply the Sobel edge detection filter
        result = filters.sobel(img)

    # Keep values between 0 and 1
    result = np.clip(result, 0, 1)

    #Convert values from 0-1 back to 0-255
    # and convert them to 8-bit unsigned integers
    result = (result * 255).astype(np.uint8)

    # Convert the NumPy array back to a PIL grayscale image
    return Image.fromarray(
        result,
        mode="L"
    )

def ask_filter(parent: tk.Misc) -> str | None:

    # Ask the user to choose a filter
    index = ask_choice(
        parent,
        "Filters",
        "Choose a filter",
        FILTERS,
    )
    # Return the selected filter or None if cancelled
    return None if index is None else FILTERS[index]

def ask_sigma(parent: tk.Misc) -> float | None:
    # Ask the user to enter Gaussian sigma value
    return tk.simpledialog.askfloat(
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

        # Check that an image is loaded
        assert document.current is not None

        # Ask the user to choose a filter
        filter_name = ask_filter(parent)

        # Stop if the user cancels
        if filter_name is None:
            return None

        # Default sigma value
        sigma = 1.0

        # Ask for sigma only when Gaussian is selected
        if filter_name == "Gaussian":
            selected_sigma = ask_sigma(parent)

            if selected_sigma is None:
                return None

            sigma = selected_sigma
        # Apply the selected filter
        output = apply_filter(
            document.current,
            filter_name,
            sigma,
        )

        # Store information about the operation
        details = {
            "Operation": "Filter",
            "Filter": filter_name,
        }

        # Store sigma when Gaussian is used
        if filter_name == "Gaussian":
            details["Sigma"] = f"{sigma:.2f}"

        # Return the filtered image and its information
        return ToolResult(
            image=output,
            message=f"Applied {filter_name} filter.",
            details=details,
        )