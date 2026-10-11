from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import feature, filters

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

# Colour used to draw the edges on top of the original image
EDGE_COLOR = (255, 0, 0)

DETECTORS = ("Canny", "Sobel", "Prewitt", "Roberts")

GRADIENT_FILTERS = {
    "Sobel": filters.sobel,
    "Prewitt": filters.prewitt,
    "Roberts": filters.roberts,
}


def detect_edges(image: Image.Image, sigma: float) -> np.ndarray:

    gray = image.convert("L")

    img = np.array(gray, dtype=np.float32) / 255.0

    return feature.canny(img, sigma=sigma)


def gradient_edges(image: Image.Image, detector: str) -> np.ndarray:

    gray = image.convert("L")
    img = np.array(gray, dtype=np.float32) / 255.0

    # Sobel, Prewitt and Roberts give the gradient magnitude, not a yes/no edge
    magnitude = GRADIENT_FILTERS[detector](img)

    # Otsu picks the threshold that separates edges from background
    return magnitude > filters.threshold_otsu(magnitude)


def apply_canny(image: Image.Image, sigma: float) -> Image.Image:

    edges = detect_edges(image, sigma)

    # White edges on a black background
    result = edges.astype(np.uint8) * 255
    return Image.fromarray(result, mode="L")


def overlay_edges(
    image: Image.Image, detector: str = "Canny", sigma: float = 1.0
) -> Image.Image:

    # Canny uses sigma; the gradient filters do not have parameters
    if detector == "Canny":
        edges = detect_edges(image, sigma)
    else:
        edges = gradient_edges(image, detector)

    # Paint the edge pixels of the original image in the edge colour
    base = np.array(image.convert("RGB"))
    base[edges] = EDGE_COLOR

    return Image.fromarray(base)


def ask_sigma(parent: tk.Misc, title: str) -> float | None:
    return simpledialog.askfloat(
        title,
        "Enter sigma:",
        parent=parent,
        minvalue=0.1,
        maxvalue=20.0,
        initialvalue=1.0,
    )


class CannyTool(ForensicsTool):

    tool_id = "canny"
    title = "Canny edges"
    category = "Set 3 duties"
    description = "Detect edges with skimage.feature.canny."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None
        sigma = ask_sigma(parent, "Canny edge detection")

        # Stop if the user cancels
        if sigma is None:
            return None

        output = apply_canny(document.current, sigma)

        return ToolResult(
            image=output,
            message=f"Applied Canny edge detection (sigma={sigma:.2f}).",
            details={
                "Operation": "Canny edge detection",
                "Sigma": f"{sigma:.2f}",
            },
        )


class EdgeOverlayTool(ForensicsTool):

    tool_id = "edge_overlay"
    title = "Edge overlay"
    category = "Set 3 duties"
    description = "Draw the edges in red on top of the original image."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None
        index = ask_choice(
            parent,
            "Edge overlay",
            "Choose an edge detector",
            DETECTORS,
        )

        if index is None:
            return None

        detector = DETECTORS[index]
        details = {
            "Operation": "Edge overlay",
            "Edge detector": detector,
        }

        sigma = 1.0
        if detector == "Canny":
            selected_sigma = ask_sigma(parent, "Edge overlay")

            if selected_sigma is None:
                return None

            sigma = selected_sigma
            details["Sigma"] = f"{sigma:.2f}"
        else:
            details["Threshold"] = "Otsu"

        output = overlay_edges(document.current, detector, sigma)

        return ToolResult(
            image=output,
            message=f"Superimposed {detector} edges.",
            details=details,
        )
