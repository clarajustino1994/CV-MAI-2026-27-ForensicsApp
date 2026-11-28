"""Set 11.1: load the U-Net trained in the Set 10 notebook and visualise its
segmentation of a tray image.

The notebook ``Set10/unet_tray_food_segmentation.ipynb`` (section 9) saves a
checkpoint with the weights and the settings needed to use them. This tool
rebuilds the same U-Net, loads that checkpoint, segments the food on a chosen
image and shows the original image, the segmentation mask and the mask laid
over the image. When the image comes from the Tray dataset its ground truth
mask is found automatically and the Dice coefficient is reported.
"""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from forensics_app.core import ImageDocument

from . import filedialog
from .base import ForensicsTool, ToolResult
from .dialogs import ask_details_action

SET10_DIR = Path(__file__).resolve().parents[2] / "Set10"
MODEL_PATH = SET10_DIR / "models" / "unet_tray_food.pt"
SAMPLES_DIR = SET10_DIR / "TrayDataset" / "XTest"
NOTEBOOK = "Set10/unet_tray_food_segmentation.ipynb"

ACTION_SELECT = "Select image to segment…"
OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
MODEL_TYPES = [("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")]

# Colours of the mask panels (ends of matplotlib's viridis, as in the notebook)
BACKGROUND_COLOR = (68, 1, 84)
FOOD_COLOR = (253, 231, 37)
OVERLAY_ALPHA = 0.45
PANEL_GAP = 12


# ---------------------------------------------------------------- model


def build_unet(in_ch: int = 3, out_ch: int = 1, base: int = 16) -> Any:
    """The U-Net of the Set 10 notebook (section 4), layer names included,
    so the saved ``state_dict`` loads without any renaming."""
    import torch  # imported lazily: torch takes seconds to load
    from torch import nn

    def conv_block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
            self.enc1 = conv_block(in_ch, c1)
            self.enc2 = conv_block(c1, c2)
            self.enc3 = conv_block(c2, c3)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = conv_block(c3, c4)
            self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), conv_block(
                c3 * 2, c3
            )
            self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), conv_block(
                c2 * 2, c2
            )
            self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), conv_block(
                c1 * 2, c1
            )
            self.head = nn.Conv2d(c1, out_ch, 1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return UNet()


@dataclass
class LoadedUNet:
    net: Any
    config: dict[str, Any]
    val_dice: float | None
    test_dice: float | None
    path: Path

    @property
    def threshold(self) -> float:
        return float(self.config.get("threshold", 0.5))

    @property
    def input_size(self) -> tuple[int, int]:
        """(width, height) the network was trained on."""
        return int(self.config["W"]), int(self.config["H"])

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def load_unet(path: Path) -> LoadedUNet:
    """Rebuild the network from the checkpoint ``config`` and load its weights."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    net = build_unet(cfg["in_ch"], cfg["out_ch"], cfg["base"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()  # BatchNorm must use the statistics stored during training
    return LoadedUNet(net, cfg, ckpt.get("val_dice"), ckpt.get("test_dice"), path)


# ---------------------------------------------------------------- helpers


def preprocess(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize to ``size`` (W, H) and scale to [0, 1]; shape (1, 3, H, W)."""
    small = image.convert("RGB").resize(size, Image.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[np.newaxis]


def predict_mask(model: LoadedUNet, image: Image.Image) -> np.ndarray:
    """Boolean food mask at the resolution of ``image``."""
    import torch

    with torch.no_grad():
        logits = model.net(torch.from_numpy(preprocess(image, model.input_size)))
        prob = torch.sigmoid(logits)[0, 0].numpy()
    small = Image.fromarray(((prob > model.threshold) * 255).astype(np.uint8))
    # NEAREST keeps the mask binary when it is scaled back to the image size
    return np.asarray(small.resize(image.size, Image.NEAREST)) > 0


def find_mask_file(image_path: Path) -> Path | None:
    """Ground truth of a Tray dataset image: ``XTest/a.jpg`` -> ``yTest/a.png``."""
    image_path = Path(image_path)
    folder = image_path.parent
    if folder.name[:1] not in ("X", "x"):
        return None
    candidate = folder.parent / f"y{folder.name[1:]}" / f"{image_path.stem}.png"
    return candidate if candidate.is_file() else None


def read_food_mask(path: Path, first_food: int, size: tuple[int, int]) -> np.ndarray:
    """Class-index mask -> boolean food mask (class >= ``first_food``)."""
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:  # some masks are saved with 3 identical channels
        mask = mask[..., 0]
    mask_image = Image.fromarray(mask.astype(np.uint8))
    if mask_image.size != size:
        mask_image = mask_image.resize(size, Image.NEAREST)
    return np.asarray(mask_image) >= first_food


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def colorize(mask: np.ndarray) -> np.ndarray:
    colored = np.empty((*mask.shape, 3), dtype=np.uint8)
    colored[:] = BACKGROUND_COLOR
    colored[mask] = FOOD_COLOR
    return colored


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mixed = (1 - OVERLAY_ALPHA) * base + OVERLAY_ALPHA * colorize(mask)
    return Image.fromarray(mixed.astype(np.uint8))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def compose_panels(
    panels: list[tuple[str, Image.Image]], columns: int = 2
) -> Image.Image:
    """Arrange same-size images in a grid (2 per row), each titled above.

    A last row with fewer images is centred.
    """
    width, height = panels[0][1].size
    font_size = max(14, width // 22)
    title_h = font_size + 14
    cell_w, cell_h = width + PANEL_GAP, height + title_h + PANEL_GAP
    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)  # ceiling division
    canvas = Image.new(
        "RGB",
        (columns * cell_w + PANEL_GAP, rows * cell_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(font_size)
    for index, (title, panel) in enumerate(panels):
        row, col = divmod(index, columns)
        in_row = min(columns, len(panels) - row * columns)
        x = PANEL_GAP + (columns - in_row) * cell_w // 2 + col * cell_w
        y = row * cell_h
        left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
        draw.text(
            (
                x + (width - (right - left)) // 2 - left,
                y + (title_h - (bottom - top)) // 2 - top,
            ),
            title,
            fill=(0, 0, 0),
            font=font,
        )
        canvas.paste(panel.convert("RGB"), (x, y + title_h))
    return canvas


# ---------------------------------------------------------------- tool


class UNetSegmentationTool(ForensicsTool):
    tool_id = "unet_segmentation"
    title = "U-Net food segmentation"
    category = "Set 11 duties"
    description = (
        "Set 11.1: load the U-Net trained on the Tray dataset, pick an image "
        "and visualise the food segmentation (original, mask and overlay)."
    )
    requires_image = False  # the image is chosen from the model window

    def __init__(
        self, model_path: Path = MODEL_PATH, samples_dir: Path = SAMPLES_DIR
    ) -> None:
        self.model_path = model_path
        self.samples_dir = samples_dir
        self._model: LoadedUNet | None = None

    def _get_model(self) -> LoadedUNet:
        # Load once and keep it; reload if another checkpoint was chosen
        if self._model is None or self._model.path != self.model_path:
            self._model = load_unet(self.model_path)
        return self._model

    def model_details(self, model: LoadedUNet) -> dict[str, str]:
        cfg = model.config
        width, height = model.input_size
        details = {
            "Checkpoint": (
                str(model.path.relative_to(SET10_DIR.parent))
                if model.path.is_relative_to(SET10_DIR.parent)
                else str(model.path)
            ),
            "Architecture": f"U-Net, base {cfg['base']} channels, "
            f"{model.parameters:,} parameters",
            "Dataset": "Tray Food Segmentation (Kaggle)",
            "Task": f"food (class >= {cfg['first_food']}) vs. not food",
            "Input size": f"{width}x{height}",
            "Threshold": f"{model.threshold:.2f}",
        }
        if model.val_dice is not None:
            details["Validation Dice"] = f"{model.val_dice:.3f}"
        if model.test_dice is not None:
            details["Test Dice"] = f"{model.test_dice:.3f}"
        return details

    def _choose_model_file(self, parent: tk.Misc) -> bool:
        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose the U-Net checkpoint",
            message=f"No trained model at {self.model_path}.\n\n"
            f"Run section 9 of {NOTEBOOK} to save it, or select a "
            "checkpoint (.pt) saved by that notebook.",
            initialdir=str(
                self.model_path.parent if self.model_path.parent.exists() else SET10_DIR
            ),
            filetypes=MODEL_TYPES,
        )
        if not filename:
            return False
        self.model_path = Path(filename)
        return True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not self.model_path.exists() and not self._choose_model_file(parent):
            return None
        try:
            model = self._get_model()
        except ImportError:
            return ToolResult(
                message="PyTorch is not installed. Run: pip install torch"
            )
        except Exception as error:  # wrong or corrupt file: unpickling, keys, shapes
            self._model = None
            return ToolResult(message=f"Could not load the U-Net model: {error}")

        if not ask_details_action(
            parent,
            self.title,
            "Trained model",
            self.model_details(model),
            ACTION_SELECT,
        ):
            return None

        filename = filedialog.askopenfilename(
            parent=parent,
            title="Choose an image to segment",
            message="Select a tray image to segment. For Tray dataset images "
            "the ground truth is read from the matching y folder "
            "(XTest/<name>.jpg -> yTest/<name>.png) and the Dice is shown.",
            initialdir=str(self.samples_dir) if self.samples_dir.exists() else None,
            filetypes=OPEN_TYPES,
        )
        if not filename:
            return None
        try:
            document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            return ToolResult(message=f"Could not open image: {error}")
        return self.segment(document, Path(filename))

    def segment(self, document: ImageDocument, image_path: Path) -> ToolResult:
        assert document.current is not None
        model = self._get_model()
        image = document.current.convert("RGB")

        start = time.perf_counter()
        pred = predict_mask(model, image)
        elapsed = time.perf_counter() - start

        panels = [("Original image", image)]
        details: dict[str, Any] = {
            "Operation": "Semantic segmentation (U-Net)",
            "Image": image_path.name,
            "Image size": f"{image.width} x {image.height}",
            "Inference time": f"{elapsed:.3f} s",
            "Food pixels": f"{pred.mean():.1%}",
        }

        mask_path = find_mask_file(image_path)
        if mask_path is None:
            details["Ground truth"] = "not found"
            message = (
                f"Food covers {pred.mean():.1%} of the image (no ground truth found)."
            )
        else:
            gt = read_food_mask(mask_path, int(model.config["first_food"]), image.size)
            d, j = dice(pred, gt), iou(pred, gt)
            panels.append(("Ground truth mask", Image.fromarray(colorize(gt))))
            details["Ground truth"] = mask_path.name
            details["Dice"] = f"{d:.3f}"
            details["IoU"] = f"{j:.3f}"
            message = f"Segmented {image_path.name}: Dice {d:.3f}, IoU {j:.3f}."

        panels += [
            ("Segmentation mask", Image.fromarray(colorize(pred))),
            ("Overlayed mask", overlay(image, pred)),
        ]
        return ToolResult(
            image=compose_panels(panels), message=message, details=details
        )
