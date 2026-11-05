from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class FaceDetectionTool(ForensicsTool):

    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Set 6 duties"
    description = "Detect faces using the YuNet model."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        assert document.current is not None

        # Select the YuNet ONNX model
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose YuNet model",
            filetypes=[("ONNX model", "*.onnx")],
        )

        if not filename:
            return None

        # Convert the image to RGB
        image = document.current.convert("RGB")
        image_array = np.array(image)

        height, width = image_array.shape[:2]

        # Create the YuNet face detector
        detector = cv2.FaceDetectorYN.create(
            filename,
            "",
            (width, height),
            0.9,
            0.3,
            5000,
        )

        # Detect faces
        _, faces = detector.detect(image_array)

        # Draw the results
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        number_of_faces = 0

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)

                # Draw the face rectangle
                draw.rectangle(
                    (x, y, x + w, y + h),
                    outline="lime",
                    width=3,
                )

                # Draw the five facial landmarks
                landmarks = face[4:14].reshape(5, 2)

                for px, py in landmarks.astype(int):
                    radius = 3
                    draw.ellipse(
                        (
                            px - radius,
                            py - radius,
                            px + radius,
                            py + radius,
                        ),
                        fill="blue",
                    )

                number_of_faces += 1

        return ToolResult(
            image=result_image,
            message=f"{number_of_faces} face(s) detected using YuNet.",
            details={
                "Operation": "Face detection",
                "Method": "YuNet",
                "Faces detected": number_of_faces,
                "Confidence threshold": 0.90,
            },
        )
