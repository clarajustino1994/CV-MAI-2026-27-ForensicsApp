from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ContrastStretchingTool(ForensicsTool):
    #Basic information about the tool
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Set 2 duties"
    description = "Enhance the contrast of the working image using linear contrast stretching."

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
        #Convert the image to a NumPy array
        img = np.asarray(gray, dtype=np.float32)
        #Find the lower and upper intensity limits
        #using the 5th and 95th percentiles
        min_value = np.percentile(img, 5)
        max_value = np.percentile(img, 95)
        #Check if there is enough intensity range
        if max_value <= min_value:
            return ToolResult(
                image=gray,
                message="Contrast stretching could not be applied.",
                details={"Operation": "Contrast stretching"}
            )
        #Stretch the intensity values to the range 0-255
        stretched = (img - min_value) * (
            255.0 / (max_value - min_value)
        )
        #Make sure all values stay between 0 and 255
        stretched = np.clip(stretched, 0, 255)
        #Convert the NumPy array back to a PIL image
        output = Image.fromarray(
            stretched.astype(np.uint8),
            mode="L"
        )
        #Return the new image and information about the operation
        return ToolResult(
            image=output,
            message="Contrast stretching has been applied.",
            details={
                "Operation": "Contrast stretching",
                "Lower percentile": 5,
                "Upper percentile": 95,
                "Output mode": output.mode,
            },
        )