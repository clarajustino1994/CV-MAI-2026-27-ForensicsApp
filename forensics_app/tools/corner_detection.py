from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import corner_harris, corner_peaks, corner_fast

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CornerDetectionTool(ForensicsTool):

    tool_id = "corner_detection"
    title = "Corner Detection"
    category = "Set 5"
    description = "Detect corners using FAST or Harris."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        method = simpledialog.askstring(
            "Corner Detection",
            "Detection method: Fast or Harris.",
            parent=parent,
            initialvalue="FAST",
        )

        if method is None:
            return None

        method = method.strip().lower()

        if method not in ("fast", "harris"):
            return ToolResult(
                message="Choose FAST or HARRIS",
            )
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

        #Calculate the corner response.
        if method == "fast":
            response = corner_fast(image_array)
        else:
            response = corner_harris(image_array)
        # Find the peaks using the selected threshold.
        points = corner_peaks(
            response,
            threshold_rel=peak_threshold,
        )

        # Create the FAST/Harris response map with the same image size.
        fig, ax = plt.subplots(
            figsize=(image.width / 100, image.height / 100),
            dpi=100
        )
        ax.imshow(response, cmap="viridis")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=100,
            bbox_inches=None,
            pad_inches=0
        )
        plt.close(fig)
        buffer.seek(0)
        response_image = Image.open(buffer).convert("RGB")
        buffer.close()

        #Make both images exactly the same size
        response_image = response_image.resize(
            (image.width, image.height),
            Image.Resampling.LANCZOS,
        )

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
        combined_image.paste(result_image,(0, image.height))

        return ToolResult(
            image=combined_image,
            message=f"{len(points)} corners detected using {method.upper()}.",
            details={
                "Operation": "Corner detection",
                "Method": method.upper(),
                "Peak threshold": f"{peak_threshold:.2f}",
                "Number of peaks": len(points),
            }
        )

