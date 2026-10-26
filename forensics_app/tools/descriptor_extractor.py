from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .constants import DESCRIPTORS, ORB, SIFT, SIFT_MIN_SIGMA
from .dialogs import ask_choice


class DescriptorExtractorTool(ForensicsTool):

    tool_id = "descriptor_extractor"
    title = "Descriptor extractor"
    category = "Set 5 duties"
    description = "Extract ORB or SIFT keypoints."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        index = ask_choice(
            parent,
            "Descriptor extractor",
            "Which descriptor do you want to use?",
            DESCRIPTORS,
        )

        if index is None:
            return None
        method = DESCRIPTORS[index]

        image = document.current.convert("L")
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        # Configure the selected descriptor extractor.
        if method == ORB:
            n_keypoints = simpledialog.askinteger(
                ORB,
                "Maximum number of keypoints.",
                parent=parent,
                initialvalue=200,
                minvalue=1,
            )
            if n_keypoints is None:
                return None

            extractor = feature.ORB(n_keypoints=n_keypoints)

        else:
            n_octaves = simpledialog.askinteger(
                SIFT,
                "Number of octaves:",
                parent=parent,
                initialvalue=8,
                minvalue=1,
            )

            if n_octaves is None:
                return None

            n_scales = simpledialog.askinteger(
                SIFT,
                "Number of scales:",
                parent=parent,
                initialvalue=3,
                minvalue=1,
            )

            if n_scales is None:
                return None

            sigma_min = simpledialog.askfloat(
                SIFT,
                "Minimum sigma:",
                parent=parent,
                initialvalue=1.6,
                minvalue=SIFT_MIN_SIGMA,
            )

            if sigma_min is None:
                return None

            extractor = feature.SIFT(
                n_octaves=n_octaves,
                n_scales=n_scales,
                sigma_min=sigma_min,
            )
        # Detect keypoints and calculate descriptors
        try:
            extractor.detect_and_extract(image_array)
        except (RuntimeError, ValueError):
            return ToolResult(
                message="No keypoints could be detected. "
                "Try another image or configuration.",
            )
        keypoints = extractor.keypoints
        descriptors = extractor.descriptors

        # Draw keypoints in red on the original image.
        result_image = document.current.convert("L").convert("RGB")
        draw = ImageDraw.Draw(result_image)

        radius = 3
        for y, x in keypoints:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=1,
            )
        return ToolResult(
            image=result_image,
            message=f"{len(keypoints)} keypoints extracted using {method}.",
            details={
                "Operation": "Descriptor extraction",
                "Method": method,
                "Keypoints": len(keypoints),
                "Descriptors shape": str(descriptors.shape),
            },
        )
