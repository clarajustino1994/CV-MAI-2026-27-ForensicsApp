"""Drop-in replacement for ``tkinter.filedialog`` that says what to select.

The macOS file panel does not show its ``title``, so a bare explorer used to
appear with no hint of what to pick. Each function here first shows a small
window explaining what to select (``message``, or the title if none is given)
and opens the explorer once the user clicks OK.
"""

from __future__ import annotations

from tkinter import filedialog as _tk_filedialog
from tkinter import messagebox
from typing import Any


def _explain(options: dict[str, Any]) -> dict[str, Any]:
    title = options.get("title") or "Select"
    message = options.pop("message", None) or title
    messagebox.showinfo(title, message, parent=options.get("parent"))
    return options


def askdirectory(**options: Any) -> str:
    return _tk_filedialog.askdirectory(**_explain(options))


def askopenfilename(**options: Any) -> str:
    return _tk_filedialog.askopenfilename(**_explain(options))


def asksaveasfilename(**options: Any) -> str:
    return _tk_filedialog.asksaveasfilename(**_explain(options))
