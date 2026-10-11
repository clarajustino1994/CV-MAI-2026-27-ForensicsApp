from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_name = "histogram"
    title = "Histogram visualization"
    category = "Set 2 duties"
    description = "Visualize the intensity histogram of the current image"

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult:
        assert document.current is not None
        image = document.current
        gray = image.convert("L")
        img = np.array(gray)

        histogram, _ = np.histogram(
            img.flatten(),
            bins=256,
            range=(0, 256),
        )
        plt.figure(figsize=(8, 4.5))
        plt.plot(histogram)
        plt.title("Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.xlim(0, 255)
        plt.tight_layout()
        plt.show(block=False)

        return ToolResult(
            message="Histogram visualization displayed",
            details={
                "Operation": "Histogram visualization",
                "Image mode": gray.mode,
                "Bins": 256,
            },
        )
