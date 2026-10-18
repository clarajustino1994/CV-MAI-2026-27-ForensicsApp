from __future__ import annotations

import tkinter as tk

from tkinter import filedialog, simpledialog, messagebox

import numpy as np
from PIL import Image
from skimage.feature import match_template

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

class MatchingConfigsTool(ForensicsTool):
    tool_id = "matching_configs"
    title = "Normalized matching configs"
    category = "Set 4"
    description = "Compare the template angle and matching threshold."

    def run(
            self,
            parent: tk.Misc,
            document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None
        messagebox.showinfo(
            "Template matching",
            "Select a small image or a cropped part of the current image "
            "that you want to find.\n\n"
            "For example, if the current image shows a person, "
            "select a small image containing only the person's face.\n\n"
            "Supported formats: PNG, JPG, JPEG, TIF and TIFF.",
            parent=parent,
        )
        # Select the template image
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose template image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
           ],
        )

        if not filename:
            return None

        # Ask for the rotation angle
        angle = simpledialog.askinteger(
            "Template matching",
            "Rotation angle in degrees:",
            parent=parent,
            initialvalue=0,
            minvalue=-180,
            maxvalue=180,
        )

        if angle is None:
            return None

        #Ask fot the matching threshold
        threshold = simpledialog.askfloat(
            "Template matching",
            "Matching threshold (0 to 1):",
            parent=parent,
            initialvalue=0.5,
            minvalue=0.0,
            maxvalue=1.0,
        )
        if threshold is None:
            return None
        #Convert images to grayscale and rotate the template
        image = document.current.convert("L")
        template = Image.open(filename).convert("L")
        template = template.rotate(angle, expand=True)

        if (template.width > image.width) or (template.height > image.height):
            return ToolResult(
                message="The rotated template must be smaller than the image."
            )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        template_array = np.asarray(template, dtype=np.float32) / 255.0

        # Calculate normalized cross-correlation
        correlation = match_template(image_array, template_array, pad_input=True,)

        # Find the highest correlation value
        best_match = float(correlation.max())

        #Find the position with the highest correlation
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape,)

        # Find all pixels where correlation exceeds the threshold

        # Create a copy of the original image
        original_array = np.array(document.current.convert("RGB"))

        # Darken the original image
        result_array = (
            original_array * 0.4
        ).astype(np.uint8)

        if best_match >= threshold:
            radius = 1
            result_array[
                max(0, y - radius):y + radius + 1,
                max(0, x - radius):x + radius + 1,
            ] = [255, 0 , 0]

        # Convert the result back to a PIL image
        result_image = Image.fromarray(result_array)

        return ToolResult(
            image = result_image,
            message = (
                 "Template match found." if best_match >= threshold else "No match above the selected threshold."
            ),
            details = {
                "Operation": "Template matching",
                "Rotation angle": f"{angle} degrees",
                "Threshold": f"{threshold:.2f}",
                "Best correlation": f"{best_match:.3f}",
                "Matching positions": 1 if best_match >= threshold else 0,
            },
        )
