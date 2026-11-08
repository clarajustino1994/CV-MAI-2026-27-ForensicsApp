from __future__ import annotations

import unittest
from unittest import mock

from forensics_app.tools import filedialog

TK = "forensics_app.tools.filedialog._tk_filedialog"
INFO = "forensics_app.tools.filedialog.messagebox.showinfo"


class FileDialogTests(unittest.TestCase):
    def test_explains_what_to_select_before_opening(self) -> None:
        calls = mock.Mock()
        with (
            mock.patch(INFO, calls.info),
            mock.patch(f"{TK}.askdirectory", calls.pick) as pick,
        ):
            pick.return_value = "/data"
            folder = filedialog.askdirectory(
                parent=None, title="Builder", message="Select the dataset."
            )

        self.assertEqual(folder, "/data")
        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.info("Builder", "Select the dataset.", parent=None),
                # message is consumed by the notice, not passed to Tk
                mock.call.pick(parent=None, title="Builder"),
            ],
        )

    def test_title_is_the_message_when_none_is_given(self) -> None:
        with (
            mock.patch(INFO) as info,
            mock.patch(f"{TK}.askopenfilename", return_value="") as pick,
        ):
            filedialog.askopenfilename(title="Choose template image")
        info.assert_called_once_with(
            "Choose template image", "Choose template image", parent=None
        )
        pick.assert_called_once_with(title="Choose template image")


if __name__ == "__main__":
    unittest.main()
