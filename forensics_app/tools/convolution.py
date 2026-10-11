from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_odd_size


def max_kernel_size(image: Image.Image) -> int:

    width, height = image.size
    largest = min(width, height)

    # Kernels need a center pixel, so keep the size odd
    return largest if largest % 2 == 1 else largest - 1


def apply_convolution(image: Image.Image, kernel_size: int) -> Image.Image:

    # Make sure the kernel fits inside the image
    if not 1 <= kernel_size <= max_kernel_size(image):
        raise ValueError(f"Kernel size must be between 1 and {max_kernel_size(image)}.")

    # Convert the image to grayscale
    gray = image.convert("L")

    # Convert the image to a Numpy array
    img = np.array(gray, dtype=np.float32)

    # Convolve with a normalized square kernel of ones (mean filter).
    # uniform_filter gives the same result as ndimage.convolve with
    # np.ones((k, k)) / k**2, but stays fast for very large kernels.
    result = ndimage.uniform_filter(img, size=kernel_size, mode="reflect")

    # Keep pixel values between 0 and 255
    result = np.clip(np.rint(result), 0, 255)

    # Convert the result to 8-bit unsigned integers
    result = result.astype(np.uint8)

    # Convert the NumPy array back to a grayscale image
    return Image.fromarray(result, mode="L")


def ask_kernel_size(parent: tk.Misc, maximum: int) -> int | None:

    # Ask the user to choose a kernel size between 1x1 and the image size
    return ask_odd_size(
        parent,
        "Convolution",
        f"Choose a kernel size (1 to {maximum} pixels)",
        maximum,
    )


class Convolution(ForensicsTool):

    tool_id = "convolution"
    title = "Convolution"
    category = "Set 3 duties"
    description = "Apply convolution using a configurable square kernel."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        # Check that an image is loaded
        assert document.current is not None

        # Ask the user to choose the kernel size, limited by the image size
        maximum = max_kernel_size(document.current)
        kernel_size = ask_kernel_size(parent, maximum)

        if kernel_size is None:
            return None

        # Apply the convolution
        output = apply_convolution(
            document.current,
            kernel_size,
        )

        return ToolResult(
            image=output,
            message=f"Applied {kernel_size}x{kernel_size} convolution.",
            details={
                "Operation": "Convolution",
                "Kernel size": f"{kernel_size}x{kernel_size}",
                "Maximum kernel size": f"{maximum}x{maximum}",
            },
        )
