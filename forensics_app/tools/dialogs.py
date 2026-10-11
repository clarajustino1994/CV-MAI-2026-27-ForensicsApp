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


def ask_odd_size(
    parent: tk.Misc, title: str, prompt: str, maximum: int, initial: int = 3
) -> int | None:
    """Slider limited to odd values in ``1..maximum``.

    Return the chosen size, or ``None`` on cancel.
    """
    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    value = tk.IntVar(value=min(initial, maximum))
    choice: list[int] = []

    def snap_to_odd(_event: object = None) -> None:
        try:
            size = int(float(value.get()))
        except (tk.TclError, ValueError):
            size = 1
        size = max(1, min(size, maximum))
        if size % 2 == 0:
            size = size + 1 if size < maximum else size - 1
        value.set(size)

    def accept() -> None:
        snap_to_odd()
        choice.append(value.get())
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=12)
    frame.pack()
    ttk.Label(frame, text=prompt).grid(row=0, column=0, columnspan=3, pady=(0, 8))
    ttk.Label(frame, text="1").grid(row=1, column=0)
    tk.Scale(
        frame,
        from_=1,
        to=maximum,
        orient="horizontal",
        length=320,
        variable=value,
        showvalue=False,
        command=snap_to_odd,
    ).grid(row=1, column=1, padx=4)
    ttk.Label(frame, text=str(maximum)).grid(row=1, column=2)
    spinbox = ttk.Spinbox(
        frame,
        from_=1,
        to=maximum,
        increment=2,
        textvariable=value,
        width=6,
        command=snap_to_odd,
    )
    spinbox.grid(row=2, column=1, pady=8)
    spinbox.bind("<FocusOut>", snap_to_odd)
    buttons = ttk.Frame(frame)
    buttons.grid(row=3, column=0, columnspan=3)
    ttk.Button(buttons, text="OK", command=accept).pack(side="left", padx=4)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
    dialog.bind("<Return>", lambda _event: accept())
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
