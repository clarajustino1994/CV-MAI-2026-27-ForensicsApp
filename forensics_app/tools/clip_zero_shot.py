from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult
from .dialogs import ask_texts

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_CLASSES = "no cats, one cat, two cats, three cats"
DEFAULT_TEMPLATE = "a photo of {}."
BAR_COLOR = "#3b6ea8"

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def parse_classes(text: str) -> list[str]:
    """Comma-separated class texts, trimmed, without empty or repeated ones."""
    classes: list[str] = []
    for part in text.split(","):
        label = " ".join(part.split())
        if label and label not in classes:
            classes.append(label)
    return classes


def build_prompts(classes: list[str], template: str) -> list[str]:
    """Put each class into the template: ``"a photo of {}."`` -> one sentence."""
    if "{}" not in template:
        raise ValueError(
            'The template must contain {} where the class goes, e.g. "a photo of {}."'
        )
    return [template.replace("{}", label) for label in classes]


class ClipClassifier:
    """Pre-trained CLIP, loaded once (downloaded the first time, ~600 MB)."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _as_tensor(out: Any) -> Any:
        # newer transformers versions return an output object instead of a tensor
        import torch

        return out if torch.is_tensor(out) else out.pooler_output

    def probabilities(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            img_in = self.processor(images=[image.convert("RGB")], return_tensors="pt")
            txt_in = self.processor(text=prompts, return_tensors="pt", padding=True)
            img = F.normalize(
                self._as_tensor(self.model.get_image_features(**img_in)), dim=-1
            )
            txt = F.normalize(
                self._as_tensor(self.model.get_text_features(**txt_in)), dim=-1
            )
            # scaled cosine similarity -> softmax over the classes the user typed
            logits = self.model.logit_scale.exp() * img @ txt.T
            return logits.softmax(dim=-1)[0].numpy()


def render_chart(
    image: Image.Image, classes: list[str], probs: np.ndarray, template: str
) -> Image.Image:
    """Image on the left, probability bars on the right (as in the notebook)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    order = np.argsort(probs)
    height = max(3.2, 0.45 * len(classes) + 1.2)
    fig = Figure(figsize=(10, height), dpi=110)
    FigureCanvasAgg(fig)
    a0, a1 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    a0.imshow(image.convert("RGB"))
    a0.axis("off")
    a1.barh(np.array(classes)[order], probs[order], color=BAR_COLOR)
    a1.set_xlim(0, 1)
    a1.set_xlabel("probability")
    a1.set_title(f'template: "{template}"')
    for i, value in enumerate(probs[order]):
        a1.text(min(value + 0.02, 0.9), i, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


class ClipZeroShotTool(ForensicsTool):
    tool_id = "clip_zero_shot"
    title = "CLIP zero-shot classification"
    category = "Set 10 duties"
    description = (
        "Set 10.1: type any number of classes with any text; the pre-trained "
        "CLIP model scores the image against each one, with no training."
    )
    requires_image = False  # any image is chosen from disk

    def __init__(self) -> None:
        self._classifier: ClipClassifier | None = None
        self.last_dir: Path | None = None  # the picker reopens where you were
        self.last_classes = DEFAULT_CLASSES
        self.last_template = DEFAULT_TEMPLATE

    def _get_classifier(self) -> ClipClassifier:
        if self._classifier is None:
            self._classifier = ClipClassifier()
        return self._classifier

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        # 1. Any image from disk
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose an image to classify",
            message="Select any image to classify with CLIP. Next you will "
            "write the classes.",
            initialdir=str(self.last_dir) if self.last_dir else None,
            filetypes=OPEN_TYPES,
        )
        if not filename:
            return None
        self.last_dir = Path(filename).parent

        # 2. The classes (any number, any text) and the template
        answer = ask_texts(
            parent,
            f"{self.title} — {Path(filename).name}",
            "Write the classes separated by commas: as many as you want, with "
            "any text. Nothing is trained: CLIP compares the image with the "
            "sentence of each class.\n\nThe template turns each class into a "
            'sentence ({} is replaced by the class). Leave "a photo of {}." if unsure.',
            [("Classes", self.last_classes), ("Template", self.last_template)],
            action="Classify",
        )
        if answer is None:
            return None
        classes_text, template = answer
        self.last_classes, self.last_template = classes_text, template.strip()

        # Loaded only now, so cancelling above leaves the open image untouched
        try:
            document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            return ToolResult(message=f"Could not open image: {error}")
        return self.classify(document, classes_text, template)

    def classify(
        self, document: ImageDocument, classes_text: str, template: str
    ) -> ToolResult:
        assert document.current is not None
        template = template.strip() or DEFAULT_TEMPLATE
        classes = parse_classes(classes_text)
        if len(classes) < 2:
            return ToolResult(message="Write at least 2 classes separated by commas.")
        try:
            prompts = build_prompts(classes, template)
        except ValueError as error:
            return ToolResult(message=str(error))

        try:
            start = time.perf_counter()
            classifier = self._get_classifier()
            probs = classifier.probabilities(document.current, prompts)
            elapsed = time.perf_counter() - start
        except ImportError:
            return ToolResult(
                message="transformers is not installed. Run: pip install transformers"
            )

        best = int(np.argmax(probs))
        details: dict[str, Any] = {
            "Operation": "CLIP zero-shot classification",
            "Model": f"{MODEL_NAME} (pre-trained, not retrained)",
            "Number of classes": len(classes),
            "Template": template,
            "Prediction": f"{classes[best]} ({probs[best]:.1%})",
            "Time": f"{elapsed:.2f} s",
        }
        for rank, i in enumerate(np.argsort(probs)[::-1], start=1):
            details[f"{rank}. {prompts[i]}"] = f"{probs[i]:.1%}"

        return ToolResult(
            image=render_chart(document.current, classes, probs, template),
            message=f"CLIP says: {classes[best]} ({probs[best]:.1%}).",
            details=details,
        )
