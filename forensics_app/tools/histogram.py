from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

class HistogramTool(ForensicsTool):
    #Basic information about the tool
    tool_name = "histogram"
    title = "Histogram visualization"
    category = "Set 2 duties"
    description = "Visualize the intensity histogram of the current image"

    def run(
            self,
            parent: tk.Misc,
            document: ImageDocument,
    ) -> ToolResult:
        #Check that an image has been loaded
        assert document.current is not None
        #Get the current image
        image = document.current
        #Convert the image to grayscale
        gray = image.convert("L")
        #Convert the image into a Numpy array
        img = np.array(gray)

        histogram, _ = np.histogram(
            img.flatten(),
            bins=256,
            range=(0, 256),
        )
        #Create the histogram figure
        plt.figure(figsize=(8, 4.5))
        #Draw the histogram
        plt.plot(histogram)
        #Add a title and labels to the axes
        plt.title("Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        #Limit the intensity axis from 0 to 255
        plt.xlim(0, 255)
        #Adjust tje layout
        plt.tight_layout()
        #Display the histogram without blocking the main application
        plt.show(block=False)
        #Return information about the operation
        return ToolResult(
            message="Histogram visualization displayed",
            details={
                "Operation": "Histogram visualization",
                "Image mode": gray.mode,
                "Bins": 256,
            },
        )