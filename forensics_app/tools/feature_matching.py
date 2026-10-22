"""Set 5.3: match ORB or SIFT keypoints between the current and a second image."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from itertools import cycle
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature, measure, transform

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

DESCRIPTORS = ("ORB", "SIFT")

# Big photos are shrunk so that SIFT runs in seconds instead of minutes
WORKING_SIZE = 640

# A homography needs 4 point pairs; inliers must land within 5 px of the model
RANSAC_MIN_SAMPLES = 4
RANSAC_THRESHOLD = 5.0

KEYPOINT_COLOR = (255, 0, 0)
OUTLIER_COLOR = (90, 90, 90)
MATCH_COLORS = (
    (0, 200, 255),
    (255, 160, 0),
    (0, 220, 0),
    (255, 0, 255),
    (255, 255, 0),
    (140, 90, 255),
)
KEYPOINT_RADIUS = 3


@dataclass(frozen=True)
class DescriptorParams:
    kind: str = "ORB"
    n_keypoints: int = 500  # ORB
    n_octaves: int = 8  # SIFT
    n_scales: int = 3  # SIFT
    sigma_min: float = 1.6  # SIFT

    @property
    def metric(self) -> str:
        # ORB gives binary strings, SIFT gives real-valued histograms
        return "hamming" if self.kind == "ORB" else "euclidean"


@dataclass(frozen=True)
class MatchParams:
    cross_check: bool = True
    max_ratio: float = 0.8


@dataclass(frozen=True)
class Features:
    image: Image.Image  # grayscale image the keypoints refer to
    keypoints: np.ndarray  # (row, col) per keypoint
    descriptors: np.ndarray


def default_ratio(kind: str) -> float:
    # Hamming distances of ORB are coarse, so the ratio test needs more room
    return 0.9 if kind == "ORB" else 0.8


def to_gray(image: Image.Image, size: int = WORKING_SIZE) -> Image.Image:
    gray = image.convert("L")
    gray.thumbnail((size, size))  # keeps the aspect ratio, never enlarges
    return gray


def make_extractor(params: DescriptorParams) -> feature.ORB | feature.SIFT:
    if params.kind == "ORB":
        return feature.ORB(n_keypoints=params.n_keypoints)
    return feature.SIFT(
        n_octaves=params.n_octaves,
        n_scales=params.n_scales,
        sigma_min=params.sigma_min,
    )


def extract_features(image: Image.Image, params: DescriptorParams) -> Features:
    gray = to_gray(image)
    extractor = make_extractor(params)
    try:
        extractor.detect_and_extract(np.asarray(gray, dtype=np.float32) / 255.0)
    except RuntimeError:
        # Flat images have no keypoints; treat them as an empty set
        return Features(gray, np.empty((0, 2)), np.empty((0, 0)))
    return Features(gray, extractor.keypoints, extractor.descriptors)


def match_features(
    features1: Features,
    features2: Features,
    metric: str,
    params: MatchParams,
) -> np.ndarray:
    """Return an (N, 2) array of indices into the keypoints of both images."""
    if len(features1.keypoints) == 0 or len(features2.keypoints) == 0:
        return np.empty((0, 2), dtype=int)
    return feature.match_descriptors(
        features1.descriptors,
        features2.descriptors,
        cross_check=params.cross_check,
        max_ratio=params.max_ratio,
        metric=metric,
    )


def ransac_inliers(
    features1: Features, features2: Features, matches: np.ndarray
) -> np.ndarray:
    """Mark the matches that agree with one homography between the images."""
    inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) < RANSAC_MIN_SAMPLES:
        return inliers

    # skimage keypoints are (row, col) but transforms work in (x, y)
    source = features1.keypoints[matches[:, 0]][:, ::-1]
    target = features2.keypoints[matches[:, 1]][:, ::-1]
    _model, found = measure.ransac(
        (source, target),
        transform.ProjectiveTransform,
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=RANSAC_THRESHOLD,
        max_trials=1000,
        rng=0,  # same answer on every run
    )
    return inliers if found is None else found


def draw_matches(
    features1: Features,
    features2: Features,
    matches: np.ndarray,
    inliers: np.ndarray,
) -> Image.Image:
    """Both images side by side, keypoints in red, one line per match."""
    left, right = features1.image, features2.image
    canvas = Image.new(
        "RGB", (left.width + right.width, max(left.height, right.height))
    )
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    draw = ImageDraw.Draw(canvas)

    for keypoints, offset in (
        (features1.keypoints, 0),
        (features2.keypoints, left.width),
    ):
        for row, col in keypoints:
            x, y = col + offset, row
            draw.ellipse(
                [
                    x - KEYPOINT_RADIUS,
                    y - KEYPOINT_RADIUS,
                    x + KEYPOINT_RADIUS,
                    y + KEYPOINT_RADIUS,
                ],
                outline=KEYPOINT_COLOR,
            )

    # Outliers first so the coloured inlier lines are drawn on top
    colors = cycle(MATCH_COLORS)
    order = np.argsort(inliers, kind="stable")
    for index in order:
        row1, col1 = features1.keypoints[matches[index, 0]]
        row2, col2 = features2.keypoints[matches[index, 1]]
        color = next(colors) if inliers[index] else OUTLIER_COLOR
        draw.line([col1, row1, col2 + left.width, row2], fill=color, width=1)
    return canvas


def ask_descriptor_params(parent: tk.Misc, title: str) -> DescriptorParams | None:
    choice = ask_choice(parent, title, "Descriptor:", DESCRIPTORS)
    if choice is None:
        return None
    defaults = DescriptorParams()

    if DESCRIPTORS[choice] == "ORB":
        n_keypoints = simpledialog.askinteger(
            title,
            "Number of ORB keypoints:",
            parent=parent,
            initialvalue=defaults.n_keypoints,
            minvalue=10,
            maxvalue=5000,
        )
        if n_keypoints is None:
            return None
        return DescriptorParams(kind="ORB", n_keypoints=n_keypoints)

    n_octaves = simpledialog.askinteger(
        title,
        "Number of SIFT octaves:",
        parent=parent,
        initialvalue=defaults.n_octaves,
        minvalue=1,
        maxvalue=10,
    )
    if n_octaves is None:
        return None

    n_scales = simpledialog.askinteger(
        title,
        "Scales per octave:",
        parent=parent,
        initialvalue=defaults.n_scales,
        minvalue=1,
        maxvalue=10,
    )
    if n_scales is None:
        return None

    sigma_min = simpledialog.askfloat(
        title,
        "Blur of the first scale (sigma_min):",
        parent=parent,
        initialvalue=defaults.sigma_min,
        minvalue=0.5,
        maxvalue=5.0,
    )
    if sigma_min is None:
        return None

    return DescriptorParams(
        kind="SIFT", n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min
    )


def ask_match_params(parent: tk.Misc, title: str, kind: str) -> MatchParams | None:
    cross_check = ask_choice(
        parent, title, "Keep only mutual best matches (cross check)?", ("Yes", "No")
    )
    if cross_check is None:
        return None

    max_ratio = simpledialog.askfloat(
        title,
        "Maximum distance ratio, closest / second closest (1 = no ratio test):",
        parent=parent,
        initialvalue=default_ratio(kind),
        minvalue=0.1,
        maxvalue=1.0,
    )
    if max_ratio is None:
        return None

    return MatchParams(cross_check=cross_check == 0, max_ratio=max_ratio)


def ask_second_image_path(parent: tk.Misc) -> str:
    return filedialog.askopenfilename(
        parent=parent,
        title="Choose the second image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )


def descriptor_details(params: DescriptorParams) -> dict[str, object]:
    if params.kind == "ORB":
        return {"Descriptor": "ORB", "Keypoints requested": params.n_keypoints}
    return {
        "Descriptor": "SIFT",
        "Octaves": params.n_octaves,
        "Scales per octave": params.n_scales,
        "Sigma min": params.sigma_min,
    }


def match_details(
    params: DescriptorParams, match_params: MatchParams
) -> dict[str, object]:
    return {
        "Metric": params.metric,
        "Cross check": "yes" if match_params.cross_check else "no",
        "Max ratio": match_params.max_ratio,
    }


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Keypoint matching"
    category = "Set 5"
    description = "Match ORB or SIFT keypoints with a second image."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        filename = ask_second_image_path(parent)
        if not filename:
            return None

        params = ask_descriptor_params(parent, "Keypoint matching")
        if params is None:
            return None

        match_params = ask_match_params(parent, "Keypoint matching", params.kind)
        if match_params is None:
            return None

        with Image.open(filename) as second:
            second_features = extract_features(second, params)
        current_features = extract_features(document.current, params)

        matches = match_features(
            current_features, second_features, params.metric, match_params
        )
        inliers = ransac_inliers(current_features, second_features, matches)

        return ToolResult(
            image=draw_matches(current_features, second_features, matches, inliers),
            message=f"{len(matches)} {params.kind} matches, "
            f"{int(inliers.sum())} consistent with one homography.",
            details={
                "Operation": "Keypoint matching",
                **descriptor_details(params),
                **match_details(params, match_params),
                "Keypoints (current)": len(current_features.keypoints),
                "Keypoints (second)": len(second_features.keypoints),
                "Matches": len(matches),
                "RANSAC inliers": int(inliers.sum()),
            },
        )
