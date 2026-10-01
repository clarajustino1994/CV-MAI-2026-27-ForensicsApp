"""Small modal dialogs shared by tools that need a choice from the user."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk


def ask_choice(
    parent: tk.Misc, title: str, prompt: str, options: Sequence[str]
) -> int | None:
    """Show one button per option; return the picked index, or ``None`` on cancel."""
    dialog = tk.Toplevel(parent)
    dialog.withdraw() 
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    choice: list[int] = []

    def pick(index: int) -> None:
        choice.append(index)
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=12)
    frame.pack()
    ttk.Label(frame, text=prompt).grid(
        row=0, column=0, columnspan=len(options), pady=(0, 8)
    )
    for index, name in enumerate(options):
        ttk.Button(frame, text=name, command=lambda i=index: pick(i)).grid(
            row=1, column=index, padx=4
        )
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    _center_on(dialog, parent.winfo_toplevel())
    dialog.grab_set()
    parent.wait_window(dialog)
    return choice[0] if choice else None


def _center_on(window: tk.Toplevel, owner: tk.Misc) -> None:
    window.update_idletasks()
    x = owner.winfo_rootx() + (owner.winfo_width() - window.winfo_reqwidth()) // 2
    y = owner.winfo_rooty() + (owner.winfo_height() - window.winfo_reqheight()) // 2
    window.geometry(f"+{x}+{y}")
    window.deiconify()
