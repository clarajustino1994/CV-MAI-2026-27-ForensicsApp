import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.feature_matching import (
    KEYPOINT_COLOR,
    DescriptorParams,
    FeatureMatchingTool,
    Features,
    MatchParams,
    default_ratio,
    draw_matches,
    extract_features,
    match_features,
    ransac_inliers,
    to_gray,
)
from forensics_app.tools.image_lookup import (
    Candidate,
    ImageLookupTool,
    find_images,
    rank_images,
    score,
)

MATCHING_MODULE = "forensics_app.tools.feature_matching"
LOOKUP_MODULE = "forensics_app.tools.image_lookup"

ORB = DescriptorParams(kind="ORB", n_keypoints=200)
SIFT = DescriptorParams(kind="SIFT")
MATCH = MatchParams(cross_check=True, max_ratio=0.8)

# Course images live next to the project folder, not inside the repository
SET5_IMAGES = Path(__file__).resolve().parents[2] / "Set5_images"

# Where the pattern is pasted in the scene, as (x, y)
OFFSET = (40, 20)


def make_document(image: Image.Image, path: Path | None = None) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    document.path = path
    return document


def blocks_pattern(seed: int = 0, size: int = 160) -> Image.Image:
    # Random 10x10 px grey blocks: lots of corners for ORB and blobs for SIFT
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 256, size=(size // 10, size // 10), dtype=np.uint8)
    pixels = np.kron(cells, np.ones((10, 10), dtype=np.uint8))
    return Image.fromarray(pixels, mode="L").convert("RGB")


def scene_with(pattern: Image.Image) -> Image.Image:
    # The same pattern, moved by OFFSET inside a bigger flat scene
    scene = Image.new("RGB", (240, 200), (128, 128, 128))
    scene.paste(pattern, OFFSET)
    return scene


def write_image(folder: Path, name: str, image: Image.Image) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


class FeatureExtractionTests(unittest.TestCase):
    def test_metric_follows_descriptor_type(self) -> None:
        self.assertEqual(ORB.metric, "hamming")
        self.assertEqual(SIFT.metric, "euclidean")
        self.assertGreater(default_ratio("ORB"), default_ratio("SIFT"))

    def test_to_gray_shrinks_large_images_and_keeps_small_ones(self) -> None:
        large = to_gray(Image.new("RGB", (1280, 640)), size=640)
        self.assertEqual((large.mode, large.size), ("L", (640, 320)))
        small = to_gray(Image.new("RGBA", (100, 50)), size=640)
        self.assertEqual((small.mode, small.size), ("L", (100, 50)))

    def test_orb_respects_keypoint_budget_and_gives_binary_descriptors(self) -> None:
        features = extract_features(blocks_pattern(), DescriptorParams("ORB", 50))
        self.assertGreater(len(features.keypoints), 0)
        self.assertLessEqual(len(features.keypoints), 50)
        self.assertEqual(features.descriptors.dtype, bool)
        self.assertEqual(features.descriptors.shape, (len(features.keypoints), 256))

    def test_sift_gives_128_value_descriptors(self) -> None:
        features = extract_features(blocks_pattern(), SIFT)
        self.assertGreater(len(features.keypoints), 0)
        self.assertEqual(features.descriptors.shape, (len(features.keypoints), 128))

    def test_flat_image_has_no_features_and_no_matches(self) -> None:
        flat = Image.new("RGB", (100, 100), (90, 90, 90))
        for params in (ORB, SIFT):
            empty = extract_features(flat, params)
            self.assertEqual(len(empty.keypoints), 0)
            pattern = extract_features(blocks_pattern(), params)
            matches = match_features(empty, pattern, params.metric, MATCH)
            self.assertEqual(matches.shape, (0, 2))
            self.assertFalse(ransac_inliers(empty, pattern, matches).any())


class MatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = blocks_pattern()
        self.scene = scene_with(self.pattern)

    def test_matches_follow_the_shift_of_the_pattern(self) -> None:
        for params in (ORB, SIFT):
            with self.subTest(descriptor=params.kind):
                first = extract_features(self.pattern, params)
                second = extract_features(self.scene, params)
                matches = match_features(first, second, params.metric, MATCH)
                inliers = ransac_inliers(first, second, matches)
                self.assertGreaterEqual(len(matches), 10)
                self.assertGreater(inliers.mean(), 0.8)
                # Keypoints are (row, col), the offset is (x, y)
                shift = (
                    second.keypoints[matches[inliers, 1]]
                    - first.keypoints[matches[inliers, 0]]
                )
                np.testing.assert_allclose(
                    np.median(shift, axis=0), OFFSET[::-1], atol=1
                )

    def test_unrelated_image_has_fewer_inliers_than_the_same_pattern(self) -> None:
        params = DescriptorParams("ORB", 200)
        loose = MatchParams(cross_check=True, max_ratio=1.0)
        first = extract_features(self.pattern, params)
        same = extract_features(self.scene, params)
        other = extract_features(scene_with(blocks_pattern(seed=7)), params)
        same_inliers = ransac_inliers(
            first, same, match_features(first, same, params.metric, loose)
        )
        other_inliers = ransac_inliers(
            first, other, match_features(first, other, params.metric, loose)
        )
        self.assertGreater(same_inliers.sum(), 3 * other_inliers.sum())

    def test_cross_check_keeps_one_match_per_keypoint(self) -> None:
        first = extract_features(self.pattern, ORB)
        second = extract_features(self.scene, ORB)
        no_ratio = MatchParams(cross_check=True, max_ratio=1.0)
        matches = match_features(first, second, ORB.metric, no_ratio)
        self.assertEqual(len(set(matches[:, 1])), len(matches))
        one_way = MatchParams(cross_check=False, max_ratio=1.0)
        # Without cross check every keypoint of the first image is matched
        self.assertEqual(
            len(match_features(first, second, ORB.metric, one_way)),
            len(first.keypoints),
        )

    def test_stricter_ratio_never_adds_matches(self) -> None:
        first = extract_features(self.pattern, ORB)
        second = extract_features(scene_with(blocks_pattern(seed=3)), ORB)
        counts = [
            len(match_features(first, second, ORB.metric, MatchParams(True, ratio)))
            for ratio in (1.0, 0.9, 0.6)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_ransac_needs_at_least_four_matches(self) -> None:
        first = extract_features(self.pattern, ORB)
        second = extract_features(self.scene, ORB)
        matches = match_features(first, second, ORB.metric, MATCH)[:3]
        self.assertFalse(ransac_inliers(first, second, matches).any())

    def test_draw_matches_puts_images_side_by_side_without_mutating(self) -> None:
        left = Image.new("L", (30, 20), 0)
        right = Image.new("L", (40, 50), 0)
        features1 = Features(left, np.array([[10.0, 10.0]]), np.empty((1, 0)))
        features2 = Features(right, np.array([[10.0, 20.0]]), np.empty((1, 0)))
        result = draw_matches(
            features1, features2, np.array([[0, 0]]), np.array([True])
        )
        self.assertEqual((result.mode, result.size), ("RGB", (70, 50)))
        self.assertEqual(result.getpixel((10, 7)), KEYPOINT_COLOR)  # circle top
        self.assertEqual(result.getpixel((30 + 20, 7)), KEYPOINT_COLOR)  # shifted
        self.assertNotEqual(result.getpixel((30, 10)), (0, 0, 0))  # match line
        self.assertEqual(left.getpixel((10, 7)), 0)


class FeatureMatchingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        self.pattern = blocks_pattern()
        self.second = write_image(folder, "scene.png", scene_with(self.pattern))

    def run_tool(self, path: str, params, match_params):
        document = make_document(self.pattern)
        with (
            mock.patch(f"{MATCHING_MODULE}.ask_second_image_path", return_value=path),
            mock.patch(f"{MATCHING_MODULE}.ask_descriptor_params", return_value=params),
            mock.patch(
                f"{MATCHING_MODULE}.ask_match_params", return_value=match_params
            ),
        ):
            return FeatureMatchingTool().run(None, document), document

    def test_tool_draws_matches_and_reports_counts(self) -> None:
        result, document = self.run_tool(str(self.second), ORB, MATCH)
        self.assertEqual(result.image.size, (160 + 240, 200))
        self.assertEqual(result.details["Descriptor"], "ORB")
        self.assertEqual(result.details["Metric"], "hamming")
        self.assertEqual(result.details["Cross check"], "yes")
        self.assertGreaterEqual(result.details["Matches"], 10)
        self.assertGreater(result.details["RANSAC inliers"], 0)
        self.assertIn("matches", result.message)
        self.assertEqual(document.current.size, (160, 160))

    def test_tool_reports_sift_settings(self) -> None:
        result, _ = self.run_tool(str(self.second), SIFT, MATCH)
        self.assertEqual(result.details["Descriptor"], "SIFT")
        self.assertEqual(result.details["Octaves"], SIFT.n_octaves)
        self.assertEqual(result.details["Metric"], "euclidean")

    def test_tool_returns_none_when_any_dialog_is_cancelled(self) -> None:
        self.assertIsNone(self.run_tool("", ORB, MATCH)[0])
        self.assertIsNone(self.run_tool(str(self.second), None, MATCH)[0])
        self.assertIsNone(self.run_tool(str(self.second), ORB, None)[0])


class ImageLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.pattern = blocks_pattern()
        self.reference = write_image(self.folder, "reference.png", self.pattern)
        self.same = write_image(self.folder, "b/same.jpg", scene_with(self.pattern))
        self.other = write_image(
            self.folder, "a/other.png", scene_with(blocks_pattern(seed=7))
        )
        (self.folder / "notes.txt").write_text("not an image")

    def test_find_images_searches_subfolders_and_skips_reference(self) -> None:
        self.assertEqual(
            find_images(self.folder, exclude=self.reference),
            [self.other, self.same],
        )
        self.assertEqual(len(find_images(self.folder)), 3)

    def test_rank_images_puts_the_same_pattern_first(self) -> None:
        reference = extract_features(self.pattern, ORB)
        ranking = rank_images(reference, [self.other, self.same], ORB, MATCH)
        self.assertEqual([c.path for c in ranking], [self.same, self.other])
        self.assertGreater(ranking[0].n_inliers, ranking[1].n_inliers)

    def test_rank_images_skips_files_that_are_not_images(self) -> None:
        broken = self.folder / "broken.png"
        broken.write_text("not a png")
        reference = extract_features(self.pattern, ORB)
        ranking = rank_images(reference, [broken, self.same], ORB, MATCH)
        self.assertEqual([c.path for c in ranking], [self.same])

    def test_score_uses_chosen_criterion_and_the_other_to_break_ties(self) -> None:
        def candidate(n_matches: int, n_inliers: int) -> Candidate:
            inliers = np.arange(n_matches) < n_inliers
            return Candidate(Path("x"), None, np.zeros((n_matches, 2)), inliers)

        many_matches, many_inliers = candidate(40, 5), candidate(20, 15)
        self.assertGreater(
            score(many_matches, "Matches"), score(many_inliers, "Matches")
        )
        self.assertGreater(
            score(many_inliers, "RANSAC inliers"),
            score(many_matches, "RANSAC inliers"),
        )
        self.assertGreater(
            score(candidate(30, 5), "Matches"), score(candidate(30, 4), "Matches")
        )

    def run_tool(self, folder: str, criterion: str | None = "RANSAC inliers"):
        document = make_document(self.pattern, path=self.reference)
        with (
            mock.patch(f"{LOOKUP_MODULE}.ask_folder", return_value=folder),
            mock.patch(f"{LOOKUP_MODULE}.ask_descriptor_params", return_value=ORB),
            mock.patch(f"{LOOKUP_MODULE}.ask_match_params", return_value=MATCH),
            mock.patch(f"{LOOKUP_MODULE}.ask_criterion", return_value=criterion),
        ):
            return ImageLookupTool().run(None, document)

    def test_tool_returns_best_image_and_ranking(self) -> None:
        result = self.run_tool(str(self.folder))
        self.assertEqual(result.details["Best match"], "b/same.jpg")
        self.assertEqual(result.details["Images compared"], 2)  # reference excluded
        self.assertEqual(result.details["Ranked by"], "RANSAC inliers")
        self.assertIn("1. b/same.jpg", result.details)
        self.assertIn("2. a/other.png", result.details)
        self.assertIn("Best: b/same.jpg", result.message)
        self.assertEqual(result.image.size, (160 + 240, 200))

    def test_tool_reports_folder_without_images(self) -> None:
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty)
        result = self.run_tool(str(empty))
        self.assertIsNone(result.image)
        self.assertIn("No images", result.message)

    def test_tool_returns_none_when_cancelled(self) -> None:
        self.assertIsNone(self.run_tool(""))
        self.assertIsNone(self.run_tool(str(self.folder), criterion=None))


@unittest.skipUnless(SET5_IMAGES.is_dir(), "Set5_images folder not available")
class Set5ImagesTests(unittest.TestCase):
    """End-to-end checks on the course images (slides 5.3 and 5.4)."""

    def features(self, relative: str, params: DescriptorParams) -> Features:
        with Image.open(SET5_IMAGES / relative) as image:
            return extract_features(image, params)

    def test_logo_has_more_inliers_with_cocacola_than_with_starbucks(self) -> None:
        params = DescriptorParams("ORB", 500)
        match_params = MatchParams(cross_check=True, max_ratio=0.9)
        logo = self.features("cocacola/cocacola_logo.png", params)
        counts = {}
        for name in ("cocacola/cocacola2.png", "starbucks/starbucks.jpg"):
            other = self.features(name, params)
            matches = match_features(logo, other, params.metric, match_params)
            counts[name] = int(ransac_inliers(logo, other, matches).sum())
        self.assertGreater(
            counts["cocacola/cocacola2.png"], 3 * counts["starbucks/starbucks.jpg"]
        )

    def test_sift_lookup_finds_the_other_ciudad_ciencias_photo(self) -> None:
        # Same setting as the slide: ciudad_ciencias_1 against the buildings folder
        reference = SET5_IMAGES / "buildings" / "ciudad_ciencias_1.jpg"
        paths = find_images(SET5_IMAGES / "buildings", exclude=reference)
        ranking = rank_images(
            self.features("buildings/ciudad_ciencias_1.jpg", SIFT),
            paths,
            SIFT,
            MATCH,
            criterion="Matches",
        )
        self.assertEqual(ranking[0].path.name, "ciudad_ciencias_2.jpg")

    def test_orb_lookup_of_logo_in_whole_folder_returns_a_cocacola_image(self) -> None:
        params = DescriptorParams("ORB", 500)
        match_params = MatchParams(cross_check=True, max_ratio=1.0)
        reference = SET5_IMAGES / "cocacola" / "cocacola_logo.png"
        paths = find_images(SET5_IMAGES, exclude=reference)
        logo = self.features("cocacola/cocacola_logo.png", params)
        by_inliers = rank_images(logo, paths, params, match_params, "RANSAC inliers")
        self.assertEqual(by_inliers[0].path.parent.name, "cocacola")
        self.assertNotIn(reference, [c.path for c in by_inliers])


if __name__ == "__main__":
    unittest.main()
