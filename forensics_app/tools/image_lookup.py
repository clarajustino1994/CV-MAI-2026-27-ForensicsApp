"""Set 5.4: find the image in a folder that best matches the current image."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .feature_matching import (
    DescriptorParams,
    Features,
    MatchParams,
    ask_descriptor_params,
    ask_match_params,
    descriptor_details,
    draw_matches,
    extract_features,
    match_details,
    match_features,
    ransac_inliers,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# "Matches" is the simplest rule from the slides; "inliers" adds a geometric check
CRITERIA = ("RANSAC inliers", "Matches")

# Number of runner-up images listed in the details panel
RANKING_SIZE = 10


@dataclass(frozen=True)
class Candidate:
    path: Path
    features: Features
    matches: np.ndarray
    inliers: np.ndarray

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def find_images(folder: Path, exclude: Path | None = None) -> list[Path]:
    """Every image file below ``folder``, sorted, without ``exclude`` itself."""
    excluded = exclude.resolve() if exclude is not None else None
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve() != excluded
    )


def score(candidate: Candidate, criterion: str) -> tuple[int, int]:
    # Ties on the main criterion are broken by the other count
    if criterion == "Matches":
        return candidate.n_matches, candidate.n_inliers
    return candidate.n_inliers, candidate.n_matches


def rank_images(
    reference: Features,
    paths: list[Path],
    params: DescriptorParams,
    match_params: MatchParams,
    criterion: str = CRITERIA[0],
) -> list[Candidate]:
    """Compare the reference with every image; best candidate first."""
    candidates = []
    for path in paths:
        try:
            with Image.open(path) as image:
                features = extract_features(image, params)
        except (OSError, UnidentifiedImageError):
            continue  # unreadable or not really an image
        matches = match_features(reference, features, params.metric, match_params)
        inliers = ransac_inliers(reference, features, matches)
        candidates.append(Candidate(path, features, matches, inliers))

    # sorted() is stable, so equal scores keep the alphabetical order
    return sorted(candidates, key=lambda c: score(c, criterion), reverse=True)


def ask_folder(parent: tk.Misc) -> str:
    messagebox.showinfo(
        "Choose the folder to search",
        "Select a folder of images. Every image in it and its subfolders "
        "is compared with the current image to find the closest one.",
        parent=parent,
    )
    return filedialog.askdirectory(
        parent=parent, title="Choose the folder to search", mustexist=True
    )


def ask_criterion(parent: tk.Misc) -> str | None:
    choice = ask_choice(parent, "Image lookup", "Rank the images by:", CRITERIA)
    return None if choice is None else CRITERIA[choice]


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image lookup"
    category = "Set 5 duties"
    description = "Find the image in a folder with the most keypoint matches."

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:
        assert document.current is not None

        folder = ask_folder(parent)
        if not folder:
            return None

        params = ask_descriptor_params(parent, "Image lookup")
        if params is None:
            return None

        match_params = ask_match_params(parent, "Image lookup", params.kind)
        if match_params is None:
            return None

        criterion = ask_criterion(parent)
        if criterion is None:
            return None

        # Do not return the open image as its own closest match
        root = Path(folder)
        paths = find_images(root, exclude=document.path)
        if not paths:
            return ToolResult(message=f"No images found in {folder}.")

        reference = extract_features(document.current, params)
        ranking = rank_images(reference, paths, params, match_params, criterion)
        if not ranking:
            return ToolResult(message=f"None of the images in {folder} could be read.")

        best = ranking[0]
        best_name = best.path.relative_to(root).as_posix()
        reference_name = document.path.name if document.path else "current image"
        return ToolResult(
            image=draw_matches(reference, best.features, best.matches, best.inliers),
            message=f"Reference: {reference_name} <---> Best: {best_name} "
            f"(matches={best.n_matches}, inliers={best.n_inliers})",
            details={
                "Operation": "Closest image lookup",
                **descriptor_details(params),
                **match_details(params, match_params),
                "Ranked by": criterion,
                "Images compared": len(ranking),
                "Best match": best_name,
                **{
                    f"{place}. {c.path.relative_to(root).as_posix()}": (
                        f"{c.n_matches} matches, {c.n_inliers} inliers"
                    )
                    for place, c in enumerate(ranking[:RANKING_SIZE], start=1)
                },
            },
        )
