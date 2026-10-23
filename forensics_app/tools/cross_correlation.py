"Calculate the normalized cross_correlation between two images."

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image
from skimage.feature import match_template

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class CrossCorrelationTool(ForensicsTool):
    tool_id = "cross_correlation"
    title = "Normalized cross correlation"
    category = "Set 4 duties"
    description = "Compare a template with the current image."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        # Check that an image is already open in the application
        assert document.current is not None
        messagebox.showinfo(
            "Choose a template image",
            "Select a small image or part of an image"
            "that you want to find in the current image.\n\n"
            "Supported formats: PNG, JPG, JPEG, TIF an TIFF.",
            parent=parent,
        )

        # Open the file dialog to select the template image
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose template image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        # Stip if the user cancels the selection
        if not filename:
            return None

        # Convert the current image and the template to grayscale
        image = document.current.convert("L")
        template = Image.open(filename).convert("L")

        # Check that the template fits inside the current image
        if (template.width > image.width) or (template.height > image.height):
            messagebox.showerror(
                "Invalid template",
                "The template must be smaller than the current image.",
            )
            return None
        # Convert both images to Numpy arrays with values between 0 and 1
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        template_array = np.asarray(template, dtype=np.float32) / 255.0

        # Calculate normalized cross-correlation
        correlation = match_template(image_array, template_array, pad_input=True)

        # Convert the correlation values into a visible grayscale image
        result_array = np.clip((correlation + 1) * 127.5, 0, 255).astype(np.uint8)
        # Create a PIL image from the result array
        result_image = Image.fromarray(result_array)
        # Return the result and its information to the application
        return ToolResult(
            image=result_image,
            message="Normalized cross_correlation calculated.",
            details={
                "Operation": "Normalized cross_correlation",
                "Template size": f"{template.width}x{template.height}",
                "Maxim correlation": f"{correlation.max():.3f}",
            },
        )
