from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

KERNEL_SIZES = ("1", "3", "5", "7", "9")

def apply_convolution(
        image: Image.Image,
        kernel_size: int
) -> Image.Image:

    # Convert the image to grayscale
    gray = image.convert("L")

    # Convert the image to a Numpy array
    img = np.array(gray, dtype=np.float32)

    # Create a square kernel filled with ones
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float32
    )
    # Normalize the kernel
    kernel /= kernel.size

    #Apply the convolution
    result = ndimage.convolve(
        img,
        kernel,
        mode="reflect"
    )

    # Keep pixel values between 0 and 255
    result = np.clip(result, 0, 255)

    # Convert the result to 8-bit unsigned integers
    result = result.astype(np.uint8)

    # Convert the NumPy array back to a grayscale image
    return Image.fromarray(
        result,
        mode="L"
    )

def ask_kernel_size(parent: tk.Misc) -> int | None:

    # Ask the user to choose a kernel size
    index = ask_choice(
        parent,
        "Convolution",
        "Choose a kernel size",
        KERNEL_SIZES,
    )

    #  Return the selected size or None if cancelled
    return None if index is None else int(KERNEL_SIZES[index])

class Convolution(ForensicsTool):

    tool_id = "convolution"
    title = "Convolution"
    category = "Set 3 duties"
    description = "Apply convolution using a configurable square kernel."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    )-> ToolResult | None:

        # Check that an image is loaded
        assert document.current is not None

        # Ask the user to choose the kernel size
        kernel_size = ask_kernel_size(parent)

        if kernel_size is None:
            return None

        # Apply the convolution
        output = apply_convolution(
            document.current,
            kernel_size,
        )

        return ToolResult(
            image=output,
            message=f"Applied{kernel_size}x{kernel_size} convolution.",
            details={
                "Operation": "Convolution",
                "Kernel size": f"{kernel_size}x{kernel_size}",
            },
        )
