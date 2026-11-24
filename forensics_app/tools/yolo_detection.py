"""Set 9: object detection with Ultralytics YOLO11.

* 9.1 runs the COCO-pretrained ``yolo11n.pt`` on the open image.
* 9.3 runs the blood cell model trained in the Set 9 notebook on an unseen
  test image and draws the ground truth boxes next to the predicted ones.
"""

from __future__ import annotations

import csv
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import simpledialog
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult
from .dialogs import ask_details_action

SET9_DIR = Path(__file__).resolve().parents[2] / "Set9"
PRETRAINED_WEIGHTS = SET9_DIR / "yolo11n.pt"
TRAINED_RUN_DIR = SET9_DIR / "runs" / "detect" / "from_scratch_50e_v1"
TRAINED_WEIGHTS = TRAINED_RUN_DIR / "weights" / "best.pt"
# Unseen test images (and their YOLO labels) copied from the test split
SAMPLES_DIR = SET9_DIR / "runs" / "detect" / "test_eval" / "images"

BLOOD_CELL_CLASSES = ("PC", "RBC", "WBC")
# Metrics of model.val(split="test") in the Set 9 notebook (144 images)
TEST_METRICS = {
    "Test precision": "0.958",
    "Test recall": "0.959",
    "Test mAP@0.5": "0.983",
    "Test mAP@0.5:0.95": "0.838",
}

DEFAULT_CONF = 0.25  # Ultralytics' default confidence threshold
IOU_MATCH = 0.5  # a prediction is correct if IoU >= 0.5 with a same-class GT box
MAX_LISTED = 15  # detections listed one by one in the details panel

GT_COLOR = (0, 200, 0)
PRED_COLOR = (30, 110, 255)

ACTION_SELECT = "Select test image…"
OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels; ``conf`` is None for ground truth."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class MatchResult:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0


# ---------------------------------------------------------------- helpers


def find_label_file(image_path: Path) -> Path | None:
    """YOLO label of an image: ``../labels/<stem>.txt`` or ``<stem>.txt`` beside it."""
    image_path = Path(image_path)
    for candidate in (
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
    ):
        if candidate.is_file():
            return candidate
    return None


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    """Read ``class x_center y_center w h`` lines (relative) into pixel boxes."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:5])
        xc, yc, w, h = xc * width, yc * height, w * width, h * height
        boxes.append(Box(cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N, 4) and every box in ``b`` (M, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_boxes(
    gt: Sequence[Box], pred: Sequence[Box], iou_thr: float = IOU_MATCH
) -> MatchResult:
    """Greedy matching, most confident prediction first, same class only.

    Each GT box can be matched once; unmatched predictions are false
    positives and unmatched GT boxes are false negatives.
    """
    if not gt or not pred:
        return MatchResult(tp=0, fp=len(pred), fn=len(gt))
    iou = box_iou([g.xyxy for g in gt], [p.xyxy for p in pred])
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for j in sorted(range(len(pred)), key=lambda k: -(pred[k].conf or 0.0)):
        candidates = [
            i for i in range(len(gt)) if not used[i] and gt[i].cls == pred[j].cls
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: iou[i, j])
        if iou[best, j] >= iou_thr:
            used[best] = True
            tp += 1
    return MatchResult(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp)


def count_per_class(boxes: Sequence[Box], names: Sequence[str]) -> str:
    counts = {name: 0 for name in names}
    for box in boxes:
        counts[names[box.cls]] += 1
    return ", ".join(f"{name}: {n}" for name, n in counts.items())


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int,
    dash: int,
) -> None:
    x1, y1, x2, y2 = xyxy
    for start, end in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        steps = max(1, int(length // dash))
        for k in range(0, steps, 2):  # draw every other segment
            t0, t1 = k / steps, min(1.0, (k + 1) / steps)
            draw.line(
                [
                    (
                        start[0] + (end[0] - start[0]) * t0,
                        start[1] + (end[1] - start[1]) * t0,
                    ),
                    (
                        start[0] + (end[0] - start[0]) * t1,
                        start[1] + (end[1] - start[1]) * t1,
                    ),
                ],
                fill=color,
                width=width,
            )


def draw_gt_and_predictions(
    image: Image.Image,
    gt: Sequence[Box],
    pred: Sequence[Box],
    names: Sequence[str],
) -> Image.Image:
    """GT as solid green boxes, predictions as dashed blue boxes, plus a legend."""
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    line = max(2, result.width // 320)
    font = _font(max(10, result.width // 50))

    for box in gt:
        draw.rectangle(box.xyxy, outline=GT_COLOR, width=line)
        draw.text(
            (box.x1 + line + 1, box.y2 - line - 1),
            names[box.cls],
            fill=GT_COLOR,
            font=font,
            anchor="ld",
        )
    for box in pred:
        _dashed_rectangle(draw, box.xyxy, PRED_COLOR, line, dash=max(4, line * 3))
        label = f"{names[box.cls]} {box.conf:.2f}"
        left, top, right, bottom = draw.textbbox(
            (box.x1, box.y1), label, font=font, anchor="lb"
        )
        draw.rectangle([left, top, right, bottom], fill=PRED_COLOR)
        draw.text((box.x1, box.y1), label, fill=(255, 255, 255), font=font, anchor="lb")

    # Legend in the top-left corner
    legend = [("Ground truth (solid)", GT_COLOR), ("Prediction (dashed)", PRED_COLOR)]
    pad = 4
    y = pad
    for text, color in legend:
        left, top, right, bottom = draw.textbbox((pad * 2, y), text, font=font)
        draw.rectangle(
            [left - pad, top - pad, right + pad, bottom + pad], fill=(0, 0, 0)
        )
        draw.text((pad * 2, y), text, fill=color, font=font)
        y = bottom + pad * 2
    return result


def read_training_summary(run_dir: Path = TRAINED_RUN_DIR) -> dict[str, str]:
    """A few settings from ``args.yaml`` and the best epoch of ``results.csv``."""
    summary: dict[str, str] = {}
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        wanted = {
            "model": "Architecture",
            "epochs": "Epochs",
            "batch": "Batch size",
            "imgsz": "Image size",
            "optimizer": "Optimizer",
            "device": "Device",
        }
        for raw in args_path.read_text(encoding="utf-8").splitlines():
            key, _, value = raw.partition(":")
            if key.strip() in wanted:
                summary[wanted[key.strip()]] = value.strip()
    results_path = run_dir / "results.csv"
    if results_path.is_file():
        with open(results_path, encoding="utf-8") as file:
            rows = [
                {k.strip(): v for k, v in row.items()} for row in csv.DictReader(file)
            ]
        if rows:
            best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
            summary["Best epoch (val)"] = (
                f"{int(float(best['epoch']))}: "
                f"mAP@0.5 = {float(best['metrics/mAP50(B)']):.3f}, "
                f"mAP@0.5:0.95 = {float(best['metrics/mAP50-95(B)']):.3f}"
            )
    return summary


def _load_yolo(weights: Path) -> Any:
    from ultralytics import YOLO  # imported lazily: torch takes seconds to load

    return YOLO(str(weights))


def predict_boxes(model: Any, image: Image.Image, conf: float) -> tuple[Any, list[Box]]:
    """Run YOLO on a PIL image; return the raw result and the pixel boxes."""
    result = model.predict(image.convert("RGB"), conf=conf, verbose=False)[0]
    boxes = [
        Box(int(c), *map(float, xyxy), conf=float(s))
        for xyxy, c, s in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        )
    ]
    return result, boxes


# ------------------------------------------------------------------ tools


class _YoloTool(ForensicsTool):
    category = "Set 9 duties"

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:  # load once and keep it
            self._model = _load_yolo(self.weights_path)
        return self._model

    def _check_model(self) -> ToolResult | None:
        if not self.weights_path.exists():
            return ToolResult(message=f"Model weights not found: {self.weights_path}")
        try:
            self._get_model()
        except ImportError:
            return ToolResult(
                message="Ultralytics is not installed. Run: pip install ultralytics"
            )
        return None


class YoloPretrainedTool(_YoloTool):
    tool_id = "yolo_pretrained"
    title = "YOLO11 detection (pretrained)"
    description = (
        "Set 9.1: run the COCO-pretrained Ultralytics YOLO11n model on the "
        "open image and draw the detected objects."
    )

    def __init__(self, weights_path: Path = PRETRAINED_WEIGHTS) -> None:
        super().__init__(weights_path)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        conf = simpledialog.askfloat(
            self.title,
            "Confidence threshold (0 to 1):",
            parent=parent,
            initialvalue=DEFAULT_CONF,
            minvalue=0.0,
            maxvalue=1.0,
        )
        if conf is None:
            return None
        return self.detect(document, conf)

    def detect(self, document: ImageDocument, conf: float = DEFAULT_CONF) -> ToolResult:
        assert document.current is not None
        if (error := self._check_model()) is not None:
            return error

        start = time.perf_counter()
        result, boxes = predict_boxes(self._get_model(), document.current, conf)
        elapsed = time.perf_counter() - start
        names = result.names  # {class id: COCO class name}

        # result.plot() returns BGR (OpenCV order): flip to RGB for display
        annotated = Image.fromarray(np.ascontiguousarray(result.plot()[..., ::-1]))

        counts: dict[str, int] = {}
        for box in boxes:
            counts[names[box.cls]] = counts.get(names[box.cls], 0) + 1
        details: dict[str, Any] = {
            "Operation": "Object detection (pretrained YOLO)",
            "Model": f"YOLO11n, COCO-pretrained ({self.weights_path.name})",
            "Classes known": f"{len(names)} (COCO)",
            "Confidence threshold": f"{conf:.2f}",
            "Inference time": f"{elapsed:.3f} s",
            "Detections": len(boxes),
            "Per class": ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            or "none",
        }
        for rank, box in enumerate(sorted(boxes, key=lambda b: -(b.conf or 0)), 1):
            if rank > MAX_LISTED:
                break
            x1, y1, x2, y2 = (round(v) for v in box.xyxy)
            details[f"#{rank}"] = (
                f"{names[box.cls]} {box.conf:.2f} at ({x1}, {y1}, {x2}, {y2})"
            )

        return ToolResult(
            image=annotated,
            message=f"YOLO11n found {len(boxes)} object(s), confidence >= {conf:.2f}.",
            details=details,
        )


class YoloBloodCellTool(_YoloTool):
    tool_id = "yolo_blood_cells"
    title = "YOLO11 blood cells (trained)"
    description = (
        "Set 9.3: load the YOLO11n model trained from scratch on the blood "
        "cell dataset, pick an unseen test image and compare the ground "
        "truth boxes (solid green) with the predictions (dashed blue)."
    )
    requires_image = False  # the image is chosen from the model window

    def __init__(
        self,
        weights_path: Path = TRAINED_WEIGHTS,
        run_dir: Path = TRAINED_RUN_DIR,
        samples_dir: Path = SAMPLES_DIR,
    ) -> None:
        super().__init__(weights_path)
        self.run_dir = run_dir
        self.samples_dir = samples_dir

    def model_details(self) -> dict[str, str]:
        details = {
            "Weights": (
                str(self.weights_path.relative_to(SET9_DIR.parent))
                if self.weights_path.is_relative_to(SET9_DIR.parent)
                else str(self.weights_path)
            ),
            "Dataset": "Roboflow blood cells (train 3,027 images)",
            "Classes": ", ".join(BLOOD_CELL_CLASSES),
        }
        details.update(read_training_summary(self.run_dir))
        details.update(TEST_METRICS)
        return details

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not ask_details_action(
            parent, self.title, "Trained model", self.model_details(), ACTION_SELECT
        ):
            return None

        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose an unseen test image",
            message="Select a blood cell test image. Its ground truth is read "
            "from labels/<name>.txt next to the images folder.",
            initialdir=str(self.samples_dir),
            filetypes=OPEN_TYPES,
        )
        if not filename:
            return None
        try:
            document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            return ToolResult(message=f"Could not open image: {error}")
        return self.evaluate(document, Path(filename))

    def evaluate(
        self, document: ImageDocument, image_path: Path, conf: float = DEFAULT_CONF
    ) -> ToolResult:
        assert document.current is not None
        if (error := self._check_model()) is not None:
            return error

        image = document.current
        label_path = find_label_file(image_path)
        gt = read_yolo_labels(label_path, *image.size) if label_path else []

        start = time.perf_counter()
        _, pred = predict_boxes(self._get_model(), image, conf)
        elapsed = time.perf_counter() - start

        names = BLOOD_CELL_CLASSES
        details: dict[str, Any] = {
            "Operation": "Blood cell detection (trained YOLO11n)",
            "Image": image_path.name,
            "Confidence threshold": f"{conf:.2f}",
            "Inference time": f"{elapsed:.3f} s",
            "Predicted boxes": f"{len(pred)} ({count_per_class(pred, names)})",
        }
        if label_path is None:
            details["Ground truth"] = "not found (no labels/<name>.txt)"
            message = f"{len(pred)} cells predicted; no ground truth label found."
        else:
            match = match_boxes(gt, pred)
            details["Ground truth boxes"] = f"{len(gt)} ({count_per_class(gt, names)})"
            details[f"Matched (IoU >= {IOU_MATCH})"] = (
                f"TP {match.tp}, FP {match.fp}, FN {match.fn}"
            )
            details["Precision"] = f"{match.precision:.3f}"
            details["Recall"] = f"{match.recall:.3f}"
            message = (
                f"{len(pred)} predicted vs {len(gt)} ground truth cells: "
                f"precision {match.precision:.2f}, recall {match.recall:.2f}."
            )

        return ToolResult(
            image=draw_gt_and_predictions(image, gt, pred, names),
            message=message,
            details=details,
        )
