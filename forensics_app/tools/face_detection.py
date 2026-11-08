from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk, UnidentifiedImageError

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult

YUNET_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

CONFIDENCE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000

# YuNet only finds faces up to a limited pixel size, so a face close to the
# camera in a full-resolution frame is missed. Detect on a downscaled copy
# and map the results back to the original coordinates.
DETECTION_MAX_SIDE = 640

SOURCE_IMAGE = "image"
SOURCE_WEBCAM = "webcam"

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

PREVIEW_SIZE = (800, 600)
PREVIEW_DELAY_MS = 30


def create_detector(width: int, height: int):
    """Create a YuNet face detector for frames of the given size."""
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (width, height),
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        TOP_K,
    )


def detection_size(width: int, height: int) -> tuple[int, int, float]:
    """Return the detection width, height and the scale from the original."""
    scale = min(1.0, DETECTION_MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale)), scale


def detect_faces(detector, image_bgr: np.ndarray, scale: float):
    """Run YuNet on a scaled copy; return faces in original coordinates."""
    if scale != 1.0:
        height, width = image_bgr.shape[:2]
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)

    _, faces = detector.detect(image_bgr)
    if faces is None or scale == 1.0:
        return faces

    faces = faces.copy()
    faces[:, :14] /= scale  # box and landmarks; leave the score untouched
    return faces


def center_on_parent(window: tk.Toplevel, parent: tk.Misc | None) -> None:
    """Place the window in the middle of its parent (or of the screen)."""
    window.update_idletasks()
    width = window.winfo_reqwidth()
    height = window.winfo_reqheight()

    if parent is not None and parent.winfo_ismapped():
        top = parent.winfo_toplevel()
        x = top.winfo_rootx() + (top.winfo_width() - width) // 2
        y = top.winfo_rooty() + (top.winfo_height() - height) // 2
    else:
        x = (window.winfo_screenwidth() - width) // 2
        y = (window.winfo_screenheight() - height) // 2

    window.geometry(f"+{max(x, 0)}+{max(y, 0)}")


def draw_faces(image: Image.Image, faces) -> tuple[Image.Image, int]:
    """Draw face boxes and the five landmarks; return the image and face count."""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    if faces is None:
        return result_image, 0

    for face in faces:
        x, y, w, h = face[:4].astype(int)

        # Draw the face rectangle
        draw.rectangle((x, y, x + w, y + h), outline="lime", width=3)

        # Draw the five facial landmarks
        landmarks = face[4:14].reshape(5, 2)
        radius = 3
        for px, py in landmarks.astype(int):
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill="blue",
            )

    return result_image, len(faces)


def ask_source(parent: tk.Misc) -> str | None:
    """Ask whether to use an image or the webcam; return None on cancel."""
    dialog = tk.Toplevel(parent)
    dialog.title("Face detection")
    dialog.resizable(False, False)
    dialog.transient(parent)

    choice: dict[str, str | None] = {"value": None}

    def choose(value: str | None) -> None:
        choice["value"] = value
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Do you want to load an image or use your webcam?",
    ).pack(pady=(0, 12))

    buttons = ttk.Frame(frame)
    buttons.pack()
    ttk.Button(buttons, text="Load image", command=lambda: choose(SOURCE_IMAGE)).pack(
        side="left", padx=4
    )
    ttk.Button(buttons, text="Use webcam", command=lambda: choose(SOURCE_WEBCAM)).pack(
        side="left", padx=4
    )

    dialog.protocol("WM_DELETE_WINDOW", lambda: choose(None))
    dialog.bind("<Escape>", lambda _event: choose(None))
    center_on_parent(dialog, parent)
    dialog.grab_set()
    dialog.wait_window()

    return choice["value"]


def load_image(parent: tk.Misc, document: ImageDocument) -> bool:
    """Use the loaded image, or ask for one; return False if none is chosen."""
    if document.is_loaded:
        return True

    filename = filedialog.askopenfilename(
        parent=parent,
        title="Face detection",
        message="Next, select the image in which to detect faces.",
        filetypes=OPEN_TYPES,
    )
    if not filename:
        return False

    try:
        document.load(filename)
    except (OSError, UnidentifiedImageError) as error:
        messagebox.showerror("Could not open image", str(error), parent=parent)
        return False

    return True


class WebcamDialog:
    """Live webcam preview with face detection; Capture returns the frame."""

    def __init__(self, parent: tk.Misc, camera_index: int = 0) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "Could not open the webcam. Check that a camera is connected "
                "and that this app has camera permission "
                "(System Settings > Privacy & Security > Camera)."
            )

        self.result: tuple[Image.Image, int] | None = None
        self._latest: tuple[Image.Image, int] | None = None
        self._detector = None
        self._detector_size: tuple[int, int] | None = None
        self._scale = 1.0
        self._parent = parent
        self._photo: ImageTk.PhotoImage | None = None
        self._job: str | None = None

        self.window = tk.Toplevel(parent)
        self.window.title("Face detection - webcam")
        self.window.transient(parent)

        self.preview = ttk.Label(self.window)
        self.preview.pack(padx=8, pady=8)

        self.status = tk.StringVar(value="Starting webcam…")
        ttk.Label(self.window, textvariable=self.status).pack()

        buttons = ttk.Frame(self.window, padding=8)
        buttons.pack()
        ttk.Button(buttons, text="Capture", command=self._capture).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Cancel", command=self._close).pack(
            side="left", padx=4
        )

        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.bind("<Escape>", lambda _event: self._close())
        self.window.bind("<space>", lambda _event: self._capture())

    def show(self) -> tuple[Image.Image, int] | None:
        """Run the preview until the user captures or cancels."""
        self._update()
        center_on_parent(self.window, self._parent)
        self.window.grab_set()
        self.window.wait_window()
        return self.result

    def _update(self) -> None:
        ok, frame = self.capture.read()
        if ok:
            height, width = frame.shape[:2]
            if self._detector_size != (width, height):
                det_width, det_height, self._scale = detection_size(width, height)
                if self._detector is None:
                    self._detector = create_detector(det_width, det_height)
                else:
                    self._detector.setInputSize((det_width, det_height))
                self._detector_size = (width, height)

            faces = detect_faces(self._detector, frame, self._scale)
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._latest = draw_faces(rgb, faces)

            preview = self._latest[0].copy()
            preview.thumbnail(PREVIEW_SIZE)
            self._photo = ImageTk.PhotoImage(preview)
            self.preview.configure(image=self._photo)
            self.status.set(
                f"{self._latest[1]} face(s) detected. "
                "Press Capture (or Space) to keep this frame."
            )
        else:
            self.status.set("Waiting for webcam frames…")

        self._job = self.window.after(PREVIEW_DELAY_MS, self._update)

    def _capture(self) -> None:
        if self._latest is None:
            return
        self.result = self._latest
        self._close()

    def _close(self) -> None:
        if self._job is not None:
            self.window.after_cancel(self._job)
            self._job = None
        self.capture.release()
        self.window.grab_release()
        self.window.destroy()


def run_webcam(parent: tk.Misc) -> tuple[Image.Image, int] | None:
    """Show the live webcam preview; return the captured annotated frame."""
    return WebcamDialog(parent).show()


class FaceDetectionTool(ForensicsTool):

    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Set 6 duties"
    description = "Detect faces in an image or live from the webcam using YuNet."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        if not YUNET_MODEL_PATH.is_file():
            messagebox.showerror(
                "Face detection",
                f"YuNet model not found:\n{YUNET_MODEL_PATH}",
                parent=parent,
            )
            return None

        source = ask_source(parent)

        if source == SOURCE_IMAGE:
            if not load_image(parent, document):
                return None
            return self._detect_in_document(document)

        if source == SOURCE_WEBCAM:
            captured = run_webcam(parent)
            if captured is None:
                return None
            result_image, number_of_faces = captured
            return self._result(result_image, number_of_faces, "Webcam")

        return None

    def _detect_in_document(self, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        # Convert the image to RGB, then to BGR as OpenCV models expect
        image = document.current.convert("RGB")
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        height, width = image_array.shape[:2]
        det_width, det_height, scale = detection_size(width, height)
        detector = create_detector(det_width, det_height)
        faces = detect_faces(detector, image_array, scale)

        result_image, number_of_faces = draw_faces(image, faces)
        return self._result(result_image, number_of_faces, "Image")

    @staticmethod
    def _result(image: Image.Image, number_of_faces: int, source: str) -> ToolResult:
        return ToolResult(
            image=image,
            message=f"{number_of_faces} face(s) detected using YuNet.",
            details={
                "Operation": "Face detection",
                "Method": "YuNet",
                "Source": source,
                "Faces detected": number_of_faces,
                "Confidence threshold": CONFIDENCE_THRESHOLD,
            },
        )
