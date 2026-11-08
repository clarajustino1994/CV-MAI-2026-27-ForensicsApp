"""Contract implemented by every feature shown in the sidebar."""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from forensics_app.core import ImageDocument

# Flag stored in ``Image.info``; PIL keeps it through copies, so it survives
# the undo/redo history, and PNG/JPEG saving ignores it
SCROLL_KEY = "forensics_app.scroll_vertically"


def mark_scrollable(image: Image.Image) -> Image.Image:
    """Ask the viewer to fit ``image`` to the width and scroll it vertically."""
    image.info[SCROLL_KEY] = True
    return image


def is_scrollable(image: Image.Image | None) -> bool:
    return image is not None and bool(image.info.get(SCROLL_KEY))


@dataclass(frozen=True)
class ToolResult:
    """A tool may produce an image, textual measurements, or both."""

    message: str
    image: Image.Image | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ForensicsTool(ABC):
    """Base class for a functionality/module in the application."""

    tool_id = "tool"
    title = "Unnamed tool"
    category = "Other"
    description = ""
    requires_image = True

    @abstractmethod
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Run the feature; return ``None`` when the user cancels."""

    def reset(self) -> None:
        """Forget any state kept between runs; called when the image is reset."""
