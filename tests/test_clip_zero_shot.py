import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.clip_zero_shot import (
    DEFAULT_TEMPLATE,
    ClipZeroShotTool,
    build_prompts,
    parse_classes,
    render_chart,
)

ASK = "forensics_app.tools.clip_zero_shot.ask_texts"
PICK_FILE = "forensics_app.tools.clip_zero_shot.filedialog.askopenfilename"

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAS_TRANSFORMERS = False


def _document(color=(200, 100, 50)) -> ImageDocument:
    document = ImageDocument()
    document.current = document.original = Image.new("RGB", (64, 48), color)
    return document


class FakeClassifier:
    def probabilities(self, image, prompts):
        probs = np.arange(1, len(prompts) + 1, dtype=np.float32)
        return probs / probs.sum()


class ClipHelpersTests(unittest.TestCase):
    def test_any_number_of_classes_with_any_text(self) -> None:
        self.assertEqual(
            parse_classes(" no cats, one cat ,two  cats,, one cat, cats on a sofa "),
            ["no cats", "one cat", "two cats", "cats on a sofa"],
        )

    def test_template_builds_one_sentence_per_class(self) -> None:
        self.assertEqual(
            build_prompts(["two cats", "a dog"], DEFAULT_TEMPLATE),
            ["a photo of two cats.", "a photo of a dog."],
        )
        with self.assertRaises(ValueError):
            build_prompts(["cat"], "a photo")

    def test_render_chart_is_an_image(self) -> None:
        chart = render_chart(
            Image.new("RGB", (40, 30)), ["a", "b", "c"], np.array([0.2, 0.5, 0.3]), "{}"
        )
        self.assertEqual(chart.mode, "RGB")
        self.assertGreater(chart.width, chart.height)


class ClipToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ClipZeroShotTool()
        self.tool._classifier = FakeClassifier()  # no model download in unit tests
        self.tmp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tmp.name) / "any.png"
        Image.new("RGB", (64, 48), (10, 200, 30)).save(self.image_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cancel_image_picker_returns_none(self) -> None:
        with mock.patch(PICK_FILE, return_value=""), mock.patch(ASK) as ask:
            self.assertIsNone(self.tool.run(None, _document()))
        ask.assert_not_called()

    def test_always_asks_for_an_image_even_if_one_is_open(self) -> None:
        document = _document()
        with (
            mock.patch(PICK_FILE, return_value=str(self.image_path)) as pick,
            mock.patch(ASK, return_value=["cat, dog, fox", "a photo of a {}."]),
        ):
            result = self.tool.run(None, document)
        pick.assert_called_once()
        self.assertEqual(document.path, self.image_path)  # the chosen image
        self.assertEqual(result.details["Number of classes"], 3)
        self.assertEqual(result.details["Prediction"], "fox (50.0%)")
        self.assertIn("1. a photo of a fox.", result.details)
        self.assertIsNotNone(result.image)
        self.assertEqual(self.tool.last_classes, "cat, dog, fox")  # remembered
        self.assertEqual(self.tool.last_dir, self.image_path.parent)

    def test_cancel_classes_keeps_the_open_image(self) -> None:
        document = _document()
        before = document.current
        with (
            mock.patch(PICK_FILE, return_value=str(self.image_path)),
            mock.patch(ASK, return_value=None),
        ):
            self.assertIsNone(self.tool.run(None, document))
        self.assertIs(document.current, before)

    def test_unreadable_image_reports_message(self) -> None:
        with (
            mock.patch(PICK_FILE, return_value=__file__),  # not an image
            mock.patch(ASK, return_value=["cat, dog", DEFAULT_TEMPLATE]),
        ):
            result = self.tool.run(None, ImageDocument())
        self.assertIn("Could not open image", result.message)

    def test_needs_two_classes_and_a_valid_template(self) -> None:
        one = self.tool.classify(_document(), "cat", DEFAULT_TEMPLATE)
        self.assertIn("at least 2", one.message)
        bad = self.tool.classify(_document(), "cat, dog", "a photo")
        self.assertIn("{}", bad.message)
        self.assertIsNone(bad.image)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class ClipModelTests(unittest.TestCase):
    """Real CLIP; skipped when the model cannot be downloaded (offline)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = ClipZeroShotTool()
        try:
            cls.tool._get_classifier()
        except OSError as error:
            raise unittest.SkipTest(f"CLIP model not available: {error}")

    def test_colors_are_recognised_without_training(self) -> None:
        result = self.tool.classify(
            _document((220, 20, 20)), "red, green, blue", "a {} square."
        )
        self.assertTrue(result.details["Prediction"].startswith("red"))


if __name__ == "__main__":
    unittest.main()
