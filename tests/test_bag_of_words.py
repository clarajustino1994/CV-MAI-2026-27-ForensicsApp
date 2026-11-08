from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import bag_of_words as bow
from forensics_app.tools.base import is_scrollable, mark_scrollable
from forensics_app.tools.bow_tools import (
    BowClassifierTool,
    BowHistogramTool,
    BowSession,
    CodebookBuilderTool,
)
from forensics_app.tools.feature_matching import DescriptorParams

MODULE = "forensics_app.tools.bow_tools"
SIFT = DescriptorParams(kind="SIFT", n_octaves=3)
N_PER_CLASS = 8
N_WORDS = 6


def textured(kind: str, seed: int, size: int = 96) -> Image.Image:
    """Dark squares or bright dots on noise, so the two classes differ."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    if kind == "squares":
        pixels = np.full((size, size), 210.0)
        for top, left in rng.integers(4, size - 16, size=(10, 2)):
            pixels[top : top + 12, left : left + 12] = 30
    else:
        pixels = np.full((size, size), 40.0)
        for cy, cx in rng.integers(8, size - 8, size=(25, 2)):
            pixels[(y - cy) ** 2 + (x - cx) ** 2 < 9] = 230
    pixels += rng.normal(0, 8, (size, size))
    return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))


def write_dataset(folder: Path) -> None:
    for kind in ("dots", "squares"):
        (folder / kind).mkdir()
        for i in range(N_PER_CLASS):
            textured(kind, i).save(folder / kind / f"image_{i:04d}.png")
    (folder / ".hidden").mkdir()  # ignored, like .DS_Store folders


class BagOfWordsLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.data = cls.root / "data"
        cls.data.mkdir()
        write_dataset(cls.data)
        cls.dataset = bow.load_dataset(cls.data)
        cls.model, cls.train = bow.build_model(cls.dataset, SIFT, N_WORDS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_dataset_uses_one_class_per_folder(self) -> None:
        self.assertEqual(self.dataset.class_names, ["dots", "squares"])
        self.assertEqual(len(self.dataset.paths), 2 * N_PER_CLASS)
        self.assertEqual(np.bincount(self.dataset.labels).tolist(), [8, 8])

    def test_dataset_needs_two_classes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "only").mkdir()
            textured("dots", 0).save(Path(folder) / "only" / "a.png")
            with self.assertRaises(ValueError):
                bow.load_dataset(Path(folder))

    def test_split_is_stratified_and_disjoint(self) -> None:
        train, test = set(self.model.train_paths), set(self.model.test_paths)
        self.assertFalse(train & test)
        self.assertEqual(len(train) + len(test), 2 * N_PER_CLASS)
        # 25 % of 8 images per class go to the test set
        self.assertEqual(len(test), 4)
        self.assertEqual(
            sorted(Path(p).parent.name for p in test),
            ["dots", "dots", "squares", "squares"],
        )

    def test_codebook_has_one_patch_per_word(self) -> None:
        self.assertEqual(self.model.n_words, N_WORDS)
        self.assertEqual(
            self.model.patches.shape, (N_WORDS, bow.PATCH_SIZE, bow.PATCH_SIZE)
        )

    def test_word_histogram_is_normalised(self) -> None:
        histogram = bow.word_histogram(np.array([0, 2, 2, 3]), 5)
        np.testing.assert_allclose(histogram, [0.25, 0, 0.5, 0.25, 0])
        np.testing.assert_array_equal(bow.word_histogram(np.empty(0, int), 3), 0)

    def test_cut_patch_is_padded_at_the_border(self) -> None:
        gray = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10))
        patch = bow.cut_patch(gray, 0, 0)
        self.assertEqual(patch.shape, (bow.PATCH_SIZE, bow.PATCH_SIZE))
        self.assertEqual(patch[bow.PATCH_SIZE // 2, bow.PATCH_SIZE // 2], 0)

    def test_histogram_maps_each_keypoint_to_a_word(self) -> None:
        words = bow.compute_histogram(textured("squares", 50), self.model)
        self.assertEqual(len(words.words), len(words.features.keypoints))
        self.assertTrue(((words.words >= 0) & (words.words < N_WORDS)).all())
        self.assertAlmostEqual(words.histogram.sum(), 1.0)

    def test_new_images_are_classified_correctly(self) -> None:
        for kind in ("dots", "squares"):
            words = bow.compute_histogram(textured(kind, 100), self.model)
            ranked = bow.classify(words.histogram, self.model)
            self.assertEqual(ranked[0][0], kind)
            self.assertAlmostEqual(sum(p for _, p in ranked), 1.0)
            self.assertGreaterEqual(ranked[0][1], ranked[1][1])

    def test_save_and_load_round_trip(self) -> None:
        folder = self.root / "model"
        bow.save_model(self.model, folder, self.dataset)
        loaded = bow.load_model(folder)

        self.assertEqual(loaded.folder, folder)
        self.assertEqual(loaded.params, SIFT)
        self.assertEqual(loaded.class_names, self.model.class_names)
        np.testing.assert_array_equal(
            loaded.kmeans.cluster_centers_, self.model.kmeans.cluster_centers_
        )
        self.assertTrue((folder / bow.SPLIT_FILE).exists())
        test_path = Path(self.model.test_paths[0])
        self.assertEqual(loaded.split_of(test_path), "test")
        self.assertEqual(loaded.split_of(self.root / "x.png"), "not in dataset")

    def test_load_model_reports_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            bow.load_model(self.root)

    def test_visualisations_render(self) -> None:
        words = bow.compute_histogram(textured("dots", 7), self.model)
        for image in (
            bow.draw_codebook(self.model, self.train),
            bow.draw_histogram(words, self.model, "dots"),
            bow.draw_prediction(textured("dots", 7), "dots", 0.9),
        ):
            self.assertEqual(image.mode, "RGB")
            self.assertGreater(image.width, 100)

    def test_link_sample_is_capped_unique_and_repeatable(self) -> None:
        self.assertEqual(len(bow.link_sample(10)), 10)
        sample = bow.link_sample(500)
        self.assertEqual(len(sample), bow.MAX_LINKS)
        self.assertEqual(len(set(sample.tolist())), bow.MAX_LINKS)
        self.assertTrue(((sample >= 0) & (sample < 500)).all())
        np.testing.assert_array_equal(sample, bow.link_sample(500))

    def test_histogram_links_keypoints_to_their_bars(self) -> None:
        words = bow.compute_histogram(textured("dots", 8), self.model)
        with mock.patch.object(
            bow, "ConnectionPatch", wraps=bow.ConnectionPatch
        ) as line:
            bow.draw_histogram(words, self.model, "dots")

        expected = bow.link_sample(len(words.words))
        self.assertEqual(line.call_count, len(expected))
        for call, k in zip(line.call_args_list, expected):
            row, col = words.features.keypoints[k]
            word = int(words.words[k])
            self.assertEqual(call.kwargs["xyA"], (col, row))
            self.assertEqual(call.kwargs["xyB"], (word, words.histogram[word]))

    def test_assign_words_with_no_descriptors(self) -> None:
        empty = np.empty((0, 0), dtype=np.float32)
        self.assertEqual(len(bow.assign_words(self.model.kmeans, empty)), 0)

    def test_codebook_needs_at_least_k_descriptors(self) -> None:
        with self.assertRaises(ValueError):
            bow.build_codebook(np.zeros((3, 128), dtype=np.float32), 5)

    def test_split_of_and_true_class(self) -> None:
        train_path = Path(self.model.train_paths[0])
        self.assertEqual(self.model.split_of(train_path), "train")
        self.assertEqual(self.model.split_of(None), "unknown")
        self.assertEqual(bow.true_class(train_path, self.model), train_path.parent.name)
        self.assertIsNone(bow.true_class(self.root / "x.png", self.model))
        self.assertIsNone(bow.true_class(None, self.model))


class BagOfWordsToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.data = cls.root / "data"
        cls.output = cls.root / "out"
        cls.data.mkdir()
        cls.output.mkdir()
        write_dataset(cls.data)

        # Build once through the tool, as a user would
        cls.session = BowSession()
        with (
            mock.patch(f"{MODULE}.messagebox.showinfo"),
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                side_effect=[str(cls.data), str(cls.output)],
            ),
            mock.patch(f"{MODULE}.ask_descriptor_params", return_value=SIFT),
            mock.patch(f"{MODULE}.simpledialog.askinteger", return_value=N_WORDS),
            mock.patch(f"{MODULE}.ProgressDialog") as progress,
        ):
            progress.return_value.__enter__.return_value = None
            cls.build_result = CodebookBuilderTool(cls.session).run(
                None, ImageDocument()
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def document(self, kind: str, index: int) -> ImageDocument:
        document = ImageDocument()
        document.load(self.data / kind / f"image_{index:04d}.png")
        return document

    def test_builder_saves_model_and_codebook(self) -> None:
        result = self.build_result
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Visual words (K)"], N_WORDS)
        for name in (bow.MODEL_FILE, bow.CODEBOOK_FILE, bow.SPLIT_FILE):
            self.assertTrue((self.output / name).exists(), name)
        self.assertEqual(self.session.model.folder, self.output)

    def test_builder_suggests_ten_words_per_class(self) -> None:
        with (
            mock.patch(f"{MODULE}.messagebox.showinfo"),
            mock.patch(
                f"{MODULE}.filedialog.askdirectory", return_value=str(self.data)
            ),
            mock.patch(f"{MODULE}.ask_descriptor_params", return_value=SIFT),
            mock.patch(f"{MODULE}.simpledialog.askinteger", return_value=None) as ask_k,
        ):
            result = CodebookBuilderTool(BowSession()).run(None, ImageDocument())
        self.assertIsNone(result)
        self.assertEqual(ask_k.call_args.kwargs["initialvalue"], 20)

    def test_histogram_tool_uses_loaded_model(self) -> None:
        with mock.patch(f"{MODULE}.ask_choice", return_value=0):
            result = BowHistogramTool(self.session).run(None, self.document("dots", 0))
        self.assertIsNotNone(result.image)
        self.assertGreater(result.details["Keypoints"], 0)
        self.assertIn(result.details["Image split"].split()[0], ("train", "test"))

    def test_histogram_tool_loads_model_from_folder(self) -> None:
        session = BowSession()
        with mock.patch(
            f"{MODULE}.filedialog.askdirectory", return_value=str(self.output)
        ):
            result = BowHistogramTool(session).run(None, self.document("dots", 1))
        self.assertEqual(session.model.folder, self.output)
        self.assertEqual(result.details["Model folder"], str(self.output))

    def test_classifier_shows_top_probability_after_classify(self) -> None:
        test_path = Path(self.session.model.test_paths[0])
        document = ImageDocument()
        document.load(test_path)
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=0),
            mock.patch(f"{MODULE}.ask_image_action", return_value=True) as dialog,
        ):
            result = BowClassifierTool(self.session).run(None, document)

        self.assertEqual(dialog.call_args.args[3], "Classify")
        label = result.details["Predicted class"]
        self.assertTrue(result.message.startswith(f"p({label}) = "))
        self.assertEqual(result.details["True class"], test_path.parent.name)
        self.assertEqual(result.details["Image split"], "test")
        self.assertIn("p(dots)", result.details)
        self.assertIn("p(squares)", result.details)

    def test_classifier_cancelled_before_classify(self) -> None:
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=0),
            mock.patch(f"{MODULE}.ask_image_action", return_value=False),
        ):
            result = BowClassifierTool(self.session).run(
                None, self.document("squares", 0)
            )
        self.assertIsNone(result)

    def test_classifier_ignores_previous_plot_in_view(self) -> None:
        document = self.document("squares", 2)
        document.apply(Image.new("RGB", (50, 50), "white"))
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=0),
            mock.patch(f"{MODULE}.ask_image_action", return_value=True),
        ):
            result = BowClassifierTool(self.session).run(None, document)
        self.assertGreater(result.details["Keypoints"], 0)

    def test_builder_cancelled_at_dataset_folder(self) -> None:
        with (
            mock.patch(f"{MODULE}.filedialog.askdirectory", return_value=""),
            mock.patch(f"{MODULE}.ask_descriptor_params") as ask_params,
        ):
            result = CodebookBuilderTool(BowSession()).run(None, ImageDocument())
        self.assertIsNone(result)
        ask_params.assert_not_called()

    def test_builder_keeps_existing_model_when_not_replaced(self) -> None:
        session = BowSession()
        with (
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                side_effect=[str(self.data), str(self.output)],
            ),
            mock.patch(f"{MODULE}.ask_descriptor_params", return_value=SIFT),
            mock.patch(f"{MODULE}.simpledialog.askinteger", return_value=N_WORDS),
            mock.patch(f"{MODULE}.messagebox.askyesno", return_value=False) as ask,
            mock.patch(f"{MODULE}.build_model") as build,
        ):
            result = CodebookBuilderTool(session).run(None, ImageDocument())
        self.assertIsNone(result)
        ask.assert_called_once()
        build.assert_not_called()
        self.assertIsNone(session.model)

    def test_load_another_model_asks_for_folder(self) -> None:
        session = BowSession()
        session.model = self.session.model
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=1),
            mock.patch(
                f"{MODULE}.filedialog.askdirectory", return_value=str(self.output)
            ) as pick,
        ):
            model = session.get_model(None, "test")
        pick.assert_called_once()
        self.assertIsNot(model, self.session.model)
        self.assertEqual(model.folder, self.output)

    def test_cancelled_model_choice(self) -> None:
        with mock.patch(f"{MODULE}.ask_choice", return_value=None):
            result = BowHistogramTool(self.session).run(None, self.document("dots", 0))
        self.assertIsNone(result)

    def test_training_image_is_flagged(self) -> None:
        document = ImageDocument()
        document.load(Path(self.session.model.train_paths[0]))
        with mock.patch(f"{MODULE}.ask_choice", return_value=0):
            result = BowHistogramTool(self.session).run(None, document)
        self.assertIn("not a fair test", result.details["Image split"])

    def test_image_without_keypoints(self) -> None:
        blank = self.root / "blank.png"
        Image.new("L", (96, 96), 128).save(blank)
        document = ImageDocument()
        document.load(blank)
        with (
            mock.patch(f"{MODULE}.ask_choice", return_value=0),
            mock.patch(f"{MODULE}.ask_image_action") as dialog,
        ):
            histogram = BowHistogramTool(self.session).run(None, document)
            classified = BowClassifierTool(self.session).run(None, document)
        self.assertIsNone(histogram.image)
        self.assertIn("No keypoints", histogram.message)
        self.assertIn("No keypoints", classified.message)
        dialog.assert_not_called()

    def build_with_k(self, n_words: int, output: Path):
        """Run the Codebook builder with ``n_words`` visual words."""
        output.mkdir()
        with (
            mock.patch(
                f"{MODULE}.filedialog.askdirectory",
                side_effect=[str(self.data), str(output)],
            ),
            mock.patch(f"{MODULE}.ask_descriptor_params", return_value=SIFT),
            mock.patch(f"{MODULE}.simpledialog.askinteger", return_value=n_words),
            mock.patch(f"{MODULE}.ProgressDialog"),
        ):
            return CodebookBuilderTool(BowSession()).run(None, ImageDocument())

    def test_codebook_up_to_six_rows_is_fitted(self) -> None:
        # 30 words = 6 rows of 5
        result = self.build_with_k(30, self.root / "k30")
        self.assertFalse(is_scrollable(result.image))

    def test_codebook_over_six_rows_scrolls_in_the_view(self) -> None:
        # 35 words = 7 rows of 5
        result = self.build_with_k(35, self.root / "k35")
        self.assertTrue(is_scrollable(result.image))

        # The flag survives the document history but is not saved to disk
        document = self.document("dots", 0)
        document.apply(result.image)
        document.undo()
        document.redo()
        self.assertTrue(is_scrollable(document.current))
        saved = self.root / "codebook_saved.png"
        document.save(saved)
        with Image.open(saved) as image:
            self.assertFalse(is_scrollable(image))


class Set7DialogTests(unittest.TestCase):
    """Classify dialog, progress window and scrolling view, on a hidden Tk root."""

    def setUp(self) -> None:
        import tkinter as tk

        try:
            self.root = tk.Tk()
        except tk.TclError as error:  # no display available
            self.skipTest(str(error))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def press(self, label: str):
        """Replace the modal wait by a click on the button called ``label``."""
        from tkinter import ttk

        def click(dialog) -> None:
            buttons = [
                w
                for w in dialog.winfo_children()[0].winfo_children()[1].winfo_children()
                if isinstance(w, ttk.Button)
            ]
            next(b for b in buttons if b.cget("text") == label).invoke()

        return mock.patch.object(self.root, "wait_window", side_effect=click)

    def test_classify_button_returns_true(self) -> None:
        from forensics_app.tools.dialogs import ask_image_action

        image = Image.new("RGB", (80, 60), "white")
        with self.press("Classify"):
            self.assertTrue(ask_image_action(self.root, "t", image, "Classify"))

    def test_cancel_button_returns_false(self) -> None:
        from forensics_app.tools.dialogs import ask_image_action

        image = Image.new("RGB", (80, 60), "white")
        with self.press("Cancel"):
            self.assertFalse(ask_image_action(self.root, "t", image, "Classify"))

    def test_progress_dialog_updates_and_closes(self) -> None:
        from forensics_app.tools.dialogs import ProgressDialog

        with ProgressDialog(self.root, "Building") as progress:
            progress(3, 10, "Extracting")
            window = progress._window
            self.assertEqual(progress._text.get(), "Extracting")
            self.assertEqual(float(progress._bar.cget("value")), 3)
            self.assertEqual(float(progress._bar.cget("maximum")), 10)
        self.assertFalse(window.winfo_exists())

    def view(self, width: int = 600, height: int = 400):
        """An ImageView whose canvas reports a fixed size."""
        from forensics_app.ui.image_view import ImageView

        view = ImageView(self.root)
        view.pack()
        for name, value in (("winfo_width", width), ("winfo_height", height)):
            patcher = mock.patch.object(view.canvas, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return view

    def scroll_height(self, view) -> float:
        return float(view.canvas.cget("scrollregion").split()[3])

    def test_view_scrolls_marked_tall_image(self) -> None:
        from forensics_app.ui.image_view import MARGIN

        view = self.view()
        view.show(mark_scrollable(Image.new("RGB", (1100, 2800), "white")))

        self.assertTrue(view.scrolling)
        # Fitted to the width (576 px), so the height grows to 1466 px + margin
        self.assertEqual(view._photo.width(), 600 - MARGIN)
        self.assertEqual(self.scroll_height(view), view._photo.height() + MARGIN)
        self.assertGreater(view._photo.height(), 400)
        # The scrollbar is placed and drives the canvas
        self.assertEqual(view.scrollbar.winfo_manager(), "pack")
        self.assertIn("yview", str(view.scrollbar.cget("command")))

    def test_view_fits_unmarked_tall_image(self) -> None:
        view = self.view()
        view.show(Image.new("RGB", (1100, 2800), "white"))
        self.assertFalse(view.scrolling)
        self.assertLessEqual(view._photo.height(), 400)

    def test_view_does_not_scroll_marked_image_that_fits(self) -> None:
        view = self.view()
        view.show(mark_scrollable(Image.new("RGB", (300, 200), "white")))
        self.assertFalse(view.scrolling)
        self.assertEqual(self.scroll_height(view), 400)

    def test_view_stops_scrolling_for_next_image(self) -> None:
        view = self.view()
        view.show(mark_scrollable(Image.new("RGB", (1100, 2800), "white")))
        view.show(Image.new("RGB", (300, 200), "white"))
        self.assertFalse(view.scrolling)

    def test_codebook_rows(self) -> None:
        self.assertEqual(bow.codebook_rows(30), 6)
        self.assertEqual(bow.codebook_rows(31), 7)
        self.assertEqual(bow.codebook_rows(60), 12)


if __name__ == "__main__":
    unittest.main()
