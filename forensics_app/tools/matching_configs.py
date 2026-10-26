from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .cross_correlation import compute_ncc, ncc_image

# Space between the panels of the result image
PANEL_GAP = 10


def enlarge_to_height(image: Image.Image, height: int) -> Image.Image:
    # The template is tiny, so keep its pixels sharp like the slide
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.NEAREST)


def side_by_side(images: list[Image.Image]) -> Image.Image:
    """Paste images of the same height next to each other on a white canvas."""
    width = sum(image.width for image in images) + PANEL_GAP * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        canvas.paste(image.convert("RGB"), (x, 0))
        x += image.width + PANEL_GAP
    return canvas


def thresholded_image(original: Image.Image, mask: np.ndarray) -> Image.Image:
    # Grow each detected pixel to 3x3 so single pixels stay visible
    visible_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))

    # Superimposition from Set 3: darken the image, paint the matches red
    array = (np.array(original.convert("RGB")) * 0.4).astype(np.uint8)
    array[visible_mask] = [255, 0, 0]
    return Image.fromarray(array)


class MatchingConfigsTool(ForensicsTool):
    tool_id = "matching_configs"
    title = "Normalized matching configs"
    category = "Set 4 duties"
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
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff"),
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

        # Ask fot the matching threshold
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
        # Rotate the template; the new corners are filled with black
        template = Image.open(filename).convert("L")
        template = template.rotate(angle, expand=True)

        image = document.current
        if (template.width > image.width) or (template.height > image.height):
            return ToolResult(
                message="The rotated template must be smaller than the image."
            )

        # Calculate normalized cross-correlation with the rotated template
        correlation = compute_ncc(image, template)

        # Thresholded image: every pixel whose correlation exceeds the threshold
        mask = correlation >= threshold
        # Each connected group of pixels above the threshold is one match
        _, n_matches = ndimage.label(mask)

        # Rotated template | normalized cross-correlation | thresholded image
        panel = side_by_side(
            [
                enlarge_to_height(template, image.height),
                ncc_image(correlation),
                thresholded_image(image, mask),
            ]
        )
        return ToolResult(
            image=panel,
            message=(
                f"{n_matches} match(es) above the threshold."
                if n_matches
                else "No match above the selected threshold."
            ),
            details={
                "Operation": "Template matching",
                "Rotation angle": f"{angle} degrees",
                "Threshold": f"{threshold:.2f}",
                "Best correlation": f"{correlation.max():.3f}",
                "Matching regions": n_matches,
                "Pixels above threshold": int(mask.sum()),
            },
        )
