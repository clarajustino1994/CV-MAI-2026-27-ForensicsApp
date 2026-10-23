from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw
from skimage.feature import corner_fast, corner_harris, corner_peaks

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .constants import CORNER_METHODS, FAST
from .dialogs import ask_choice


class CornerDetectionTool(ForensicsTool):

    tool_id = "corner_detection"
    title = "Corner Detection"
    category = "Set 5 duties"
    description = "Detect corners using FAST or Harris."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        index = ask_choice(
            parent,
            "Corner detection",
            "Which detection method do you want to use?",
            CORNER_METHODS,
        )

        if index is None:
            return None

        method = CORNER_METHODS[index]

        peak_threshold = simpledialog.askfloat(
            "Corner detection",
            "Peak threshold (0 to 1):",
            parent=parent,
            initialvalue=0.1,
            minvalue=0.0,
            maxvalue=1.0,
        )

        if peak_threshold is None:
            return None

        # Convert the image to grayscale.
        image = document.current.convert("L")
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        # Calculate the corner response.
        if method == FAST:
            response = corner_fast(image_array)
        else:
            response = corner_harris(image_array)
        # Find the peaks using the selected threshold.
        points = corner_peaks(
            response,
            threshold_rel=peak_threshold,
        )

        # Colour the FAST/Harris response with viridis, stretched to its own
        # min..max like imshow does, so it has exactly the image size.
        low, high = float(response.min()), float(response.max())
        scaled = (response - low) / (high - low) if high > low else response * 0
        colours = colormaps["viridis"](scaled)[..., :3]
        response_image = Image.fromarray((colours * 255).astype(np.uint8))

        # Draw the detected corners in red.
        result_image = document.current.convert("L").convert("RGB")
        draw = ImageDraw.Draw(result_image)

        radius = 3
        for y, x in points:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=1,
            )
        # Combine the response map and the image with detected corners.

        combined_image = Image.new(
            "RGB",
            (image.width, image.height * 2),
        )

        combined_image.paste(response_image, (0, 0))
        combined_image.paste(result_image, (0, image.height))

        return ToolResult(
            image=combined_image,
            message=f"{len(points)} corners detected using {method}.",
            details={
                "Operation": "Corner detection",
                "Method": method,
                "Peak threshold": f"{peak_threshold:.2f}",
                "Number of peaks": len(points),
            },
        )
