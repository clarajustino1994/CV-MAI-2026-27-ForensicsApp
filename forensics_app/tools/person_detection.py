"""Set 4.4: find a person by comparing HOG descriptors in a sliding window."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .hog import HogParams, hog_descriptor, to_array

# HOG settings used in the course material for person detection
PERSON_HOG = HogParams(orientations=16, pixels_per_cell=8, cells_per_block=2)

EUCLIDEAN_COLOR = (255, 0, 0)
DOT_PRODUCT_COLOR = (0, 120, 255)
BOX_WIDTH = 2


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    score: float


def detect_person(
    image: Image.Image,
    template: Image.Image,
    step: int,
    params: HogParams = PERSON_HOG,
) -> tuple[Detection, Detection]:
    """Return the best window by Euclidean distance and by dot product."""
    window_w, window_h = template.size
    width, height = image.size
    if window_w > width or window_h > height:
        raise ValueError("The template must be smaller than the image.")

    template_fd = hog_descriptor(template, params)
    array = to_array(image)

    coords: list[tuple[int, int]] = []
    distances: list[float] = []
    dot_products: list[float] = []

    # Slide a window of the template size over the image
    for y in range(0, height - window_h + 1, step):
        for x in range(0, width - window_w + 1, step):
            window = array[y : y + window_h, x : x + window_w]
            window_fd = hog_descriptor(window, params)
            coords.append((x, y))
            distances.append(float(np.linalg.norm(template_fd - window_fd)))
            dot_products.append(float(np.dot(window_fd, template_fd)))

    # Smallest distance and largest dot product mean most similar
    best_euclid = int(np.argmin(distances))
    best_dot = int(np.argmax(dot_products))
    return (
        Detection(*coords[best_euclid], distances[best_euclid]),
        Detection(*coords[best_dot], dot_products[best_dot]),
    )


def draw_detections(
    image: Image.Image,
    window_size: tuple[int, int],
    euclidean: Detection,
    dot_product: Detection,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    window_w, window_h = window_size
    # Draw the blue box just inside the red one so both stay visible
    # when the two metrics pick the same window
    for detection, color, inset in (
        (euclidean, EUCLIDEAN_COLOR, 0),
        (dot_product, DOT_PRODUCT_COLOR, BOX_WIDTH),
    ):
        draw.rectangle(
            [
                detection.x + inset,
                detection.y + inset,
                detection.x + window_w - 1 - inset,
                detection.y + window_h - 1 - inset,
            ],
            outline=color,
            width=BOX_WIDTH,
        )
    return result


def ask_template_path(parent: tk.Misc) -> str:
    return filedialog.askopenfilename(
        parent=parent,
        title="Choose person template",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )


def max_step(image_size: tuple[int, int], window_size: tuple[int, int]) -> int:
    """Largest step that still moves the window in at least one direction."""
    width, height = image_size
    window_w, window_h = window_size
    return max(width - window_w, height - window_h, 1)


def ask_step(parent: tk.Misc, maximum: int) -> int | None:
    return simpledialog.askinteger(
        "Person detection",
        f"Sliding window step (1 to {maximum} pixels):",
        parent=parent,
        initialvalue=min(4, maximum),
        minvalue=1,
        maxvalue=maximum,
    )


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person detection (HOG)"
    category = "Set 4"
    description = "Find a person by comparing HOG descriptors of sliding windows."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        filename = ask_template_path(parent)
        if not filename:
            return None

        # Load the template first: its size limits how far the window can move
        template = Image.open(filename).convert("RGB")
        width, height = document.current.size
        if template.width > width or template.height > height:
            return ToolResult(message="The template must be smaller than the image.")

        step = ask_step(parent, max_step(document.current.size, template.size))
        if step is None:
            return None

        euclidean, dot_product = detect_person(document.current, template, step)

        result_image = draw_detections(
            document.current, template.size, euclidean, dot_product
        )
        return ToolResult(
            image=result_image,
            message="Person detection finished (red: Euclidean, blue: dot product).",
            details={
                "Operation": "Person detection with HOG",
                "Window size": f"{template.width}x{template.height}",
                "Step": f"{step} px",
                "Orientations": PERSON_HOG.orientations,
                "Euclidean (red)": (
                    f"({euclidean.x}, {euclidean.y}), distance {euclidean.score:.3f}"
                ),
                "Dot product (blue)": (
                    f"({dot_product.x}, {dot_product.y}), "
                    f"score {dot_product.score:.3f}"
                ),
            },
        )
