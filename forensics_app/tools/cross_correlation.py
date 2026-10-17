
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from skimage.feature import match_template

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

class CrossCorrelationTool(ForensicsTool):
    tool_id = "cross_correlation"
    title = "Normalized cross_correlation"
    category = "Set 4"
    description = "Compare a template with the current image."

    def run(
            self,
            parent: tk.Misc,
            document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        # Select the template image
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose template image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff" ),
                       ("All files", "*.*"),
            ],
        )

        if not filename:
            return None

        #Convert both images to grayscale
        image = document.current.convert("L")
        template = Image.open(filename).convert("L")

        if (template.width > image.width) or (template.height > image.height):
            return ToolResult(
                message="The template must be smaller than the image.",
            )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        template_array = np.asarray(template, dtype=np.float32) / 255.0


        #Calculate normalized cross-correlation
        correlation = match_template(image_array, template_array, pad_input=True)

        #Convert correlation values to a grayscale image
        result_array = np.clip (
            (correlation + 1) * 127.5, 0 , 255
        ).astype(np.uint8)

        result_image = Image.fromarray(result_array)

        return ToolResult(
            image=result_image,
            message="Normalized cross_correlation calculated.",
        details={
            "Operation": "Normalized cross_correlation",
            "Template size": f"{template.width}x{template.height}",
            "Maxim correlation": f"{correlation.max():.3f}",
        },
    )