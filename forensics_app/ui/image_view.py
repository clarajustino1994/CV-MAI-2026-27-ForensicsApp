"""Canvas that displays a PIL image scaled to the available space.

Images marked with ``mark_scrollable`` (e.g. a tall codebook) are fitted to the
width instead and scroll vertically, so their details stay readable.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from forensics_app.tools.base import is_scrollable

MARGIN = 24


class ImageView(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=8)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical")
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
            yscrollcommand=self.scrollbar.set,
        )
        self.scrollbar.configure(command=self.canvas.yview)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")
        self._scrolling = False
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self.show(None)

    @property
    def scrolling(self) -> bool:
        return self._scrolling

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        self.canvas.yview_moveto(0)
        self._render()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _on_wheel(self, event: tk.Event) -> None:
        # Scroll only when the pointer is over this view and it can scroll
        if not self.scrolling:
            return
        hovered = self.winfo_containing(event.x_root, event.y_root)
        if hovered is None or not str(hovered).startswith(str(self)):
            return
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        elif event.delta:
            step = event.delta // 120 if abs(event.delta) >= 120 else event.delta
            self.canvas.yview_scroll(-step, "units")

    def _set_scrollbar(self, visible: bool) -> None:
        if visible and not self._scrolling:
            self.scrollbar.pack(side="right", fill="y", before=self.canvas)
        elif not visible and self._scrolling:
            self.scrollbar.pack_forget()
        self._scrolling = visible

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        if self._source is None:
            self._set_scrollbar(False)
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="Open an image to begin",
                fill="#c7ccd4",
                font=("TkDefaultFont", 16),
            )
            self._photo = None
            return

        if is_scrollable(self._source):
            self._render_scrollable(width, height)
            return

        self._set_scrollbar(False)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        preview = self._source.copy()
        preview.thumbnail(
            (max(width - MARGIN, 1), max(height - MARGIN, 1)),
            Image.Resampling.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(
            width // 2, height // 2, image=self._photo, anchor="center"
        )

    def _render_scrollable(self, width: int, height: int) -> None:
        """Fit the width (never enlarging) and let the height scroll."""
        source = self._source
        assert source is not None
        scale = min(1.0, max(width - MARGIN, 1) / source.width)
        size = (
            max(round(source.width * scale), 1),
            max(round(source.height * scale), 1),
        )
        preview = source.resize(size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview)

        total = preview.height + MARGIN
        if total <= height:  # fits anyway: centre it like any other image
            self._set_scrollbar(False)
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self.canvas.create_image(
                width // 2, height // 2, image=self._photo, anchor="center"
            )
            return

        self._set_scrollbar(True)
        self.canvas.configure(scrollregion=(0, 0, width, total))
        self.canvas.create_image(width // 2, MARGIN // 2, image=self._photo, anchor="n")
