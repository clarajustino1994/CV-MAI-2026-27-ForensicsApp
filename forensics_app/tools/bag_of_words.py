"""Set 7: Bag of Visual Words — codebook, histograms and classification.

This module holds the logic shared by the three Set 7 tools and has no Tkinter
code, so it can be tested on its own:

* 7.1 ``build_model``: split the dataset, cluster the training descriptors with
  K-Means (each cluster centre is a visual word) and train a classifier on the
  word histograms of the training images.
* 7.2 ``compute_histogram``: map every descriptor of an image to its closest
  centroid and count how often each word appears.
* 7.3 ``classify``: run the trained classifier on that histogram.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from .feature_matching import DescriptorParams, Features, extract_features  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MODEL_FILE = "bow_model.joblib"
CODEBOOK_FILE = "codebook.png"
SPLIT_FILE = "split.csv"

# The slides use K = 10 visual words per class
WORDS_PER_CLASS = 10

# Same split as Set 6: 25 % of every class is kept aside for testing
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Side (px) of the grey patch cut around a keypoint to show a visual word
PATCH_SIZE = 32

# Training images drawn with their keypoints above the codebook
N_SAMPLE_IMAGES = 10

# Visual words per row in the codebook figure
CODEBOOK_COLUMNS = 5

# Lines from keypoints to histogram bars; more than this becomes unreadable
MAX_LINKS = 60

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class Dataset:
    paths: list[Path]
    labels: np.ndarray  # class index per path
    class_names: list[str]


@dataclass
class BowModel:
    """Everything needed to rebuild histograms and classify a new image."""

    params: DescriptorParams
    kmeans: MiniBatchKMeans
    classifier: SVC
    class_names: list[str]
    patches: np.ndarray  # (K, PATCH_SIZE, PATCH_SIZE) uint8, one per word
    train_paths: list[str]
    test_paths: list[str]
    test_accuracy: float
    n_descriptors: int
    folder: Path | None = None  # where it was saved / loaded from

    @property
    def n_words(self) -> int:
        return int(self.kmeans.n_clusters)

    def split_of(self, path: Path | None) -> str:
        """Tell whether an image was used for training, testing, or neither."""
        if path is None:
            return "unknown"
        resolved = str(path.resolve())
        if resolved in self.train_paths:
            return "train"
        if resolved in self.test_paths:
            return "test"
        return "not in dataset"


# ---------------------------------------------------------------- dataset


def load_dataset(folder: Path) -> Dataset:
    """Read a folder with one sub-folder of images per class."""
    class_dirs = sorted(
        d for d in folder.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    paths: list[Path] = []
    labels: list[int] = []
    class_names: list[str] = []
    for class_dir in class_dirs:
        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        label = len(class_names)
        class_names.append(class_dir.name)
        paths.extend(images)
        labels.extend([label] * len(images))

    if len(class_names) < 2:
        raise ValueError(
            f"Found {len(class_names)} class folder(s) with images in:\n{folder}\n\n"
            "Choose a folder with one sub-folder of images per class "
            "(e.g. butterfly/, dalmatian/, kangaroo/)."
        )
    counts = np.bincount(labels)
    if counts.min() < 2:
        small = class_names[int(counts.argmin())]
        raise ValueError(f"Class '{small}' needs at least 2 images to be split.")
    return Dataset(paths, np.array(labels), class_names)


def split_dataset(dataset: Dataset) -> tuple[list[int], list[int]]:
    """Stratified train/test split, returned as indices into ``dataset.paths``."""
    indices = np.arange(len(dataset.paths))
    train, test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.labels,
    )
    return sorted(train.tolist()), sorted(test.tolist())


def image_features(image: Image.Image, params: DescriptorParams) -> Features:
    """Keypoints and descriptors as floats (ORB bits become 0/1) for K-Means."""
    features = extract_features(image, params)
    descriptors = features.descriptors.astype(np.float32)
    if len(features.keypoints) == 0:
        descriptors = np.empty((0, 0), dtype=np.float32)
    return Features(features.image, features.keypoints, descriptors)


def _read_features(path: Path, params: DescriptorParams) -> Features:
    with Image.open(path) as image:
        return image_features(image, params)


# ------------------------------------------------------------- 7.1 codebook


def cut_patch(gray: Image.Image, row: float, col: float) -> np.ndarray:
    """Grey square of PATCH_SIZE px centred on a keypoint (edges replicated)."""
    pixels = np.asarray(gray, dtype=np.uint8)
    half = PATCH_SIZE // 2
    padded = np.pad(pixels, half, mode="edge")
    r, c = int(round(row)) + half, int(round(col)) + half
    return padded[r - half : r + half, c - half : c + half].copy()


def build_codebook(
    descriptors: np.ndarray, n_words: int
) -> tuple[MiniBatchKMeans, np.ndarray]:
    """Cluster descriptors; return the model and, per word, its closest descriptor."""
    if len(descriptors) < n_words:
        raise ValueError(
            f"Only {len(descriptors)} descriptors were found, fewer than the "
            f"{n_words} visual words requested."
        )
    kmeans = MiniBatchKMeans(n_clusters=n_words, random_state=0, n_init=3)
    kmeans.fit(descriptors)

    # A centroid is an average descriptor, not an image. To show it, use the
    # real descriptor nearest to it and cut the patch around its keypoint.
    distances = kmeans.transform(descriptors)  # (N, K)
    representatives = distances.argmin(axis=0)
    return kmeans, representatives


def word_histogram(words: np.ndarray, n_words: int) -> np.ndarray:
    """Normalised frequency of every visual word (sums to 1, or all zeros)."""
    counts = np.bincount(words, minlength=n_words).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def assign_words(kmeans: MiniBatchKMeans, descriptors: np.ndarray) -> np.ndarray:
    """Index of the closest centroid for every descriptor."""
    if len(descriptors) == 0:
        return np.empty(0, dtype=int)
    return kmeans.predict(descriptors.astype(kmeans.cluster_centers_.dtype))


def build_model(
    dataset: Dataset,
    params: DescriptorParams,
    n_words: int,
    progress: ProgressCallback | None = None,
) -> tuple[BowModel, list[Features]]:
    """Train the codebook and classifier; also return the training features."""
    train_idx, test_idx = split_dataset(dataset)
    n_steps = len(dataset.paths) + 2

    def report(step: int, text: str) -> None:
        if progress is not None:
            progress(step, n_steps, text)

    # 1. Descriptors of every image (test ones are kept for evaluation)
    features: dict[int, Features] = {}
    for step, index in enumerate(train_idx + test_idx):
        path = dataset.paths[index]
        report(step, f"Extracting {params.kind} descriptors: {path.name}")
        features[index] = _read_features(path, params)

    train_features = [features[i] for i in train_idx]
    usable = [f for f in train_features if len(f.keypoints)]
    if not usable:
        raise ValueError("No keypoints were detected in the training images.")

    # 2. Codebook from the training descriptors only (no test leakage)
    report(len(dataset.paths), f"Clustering descriptors into {n_words} words")
    descriptors = np.vstack([f.descriptors for f in usable])
    owner = np.concatenate([np.full(len(f.keypoints), i) for i, f in enumerate(usable)])
    keypoints = np.vstack([f.keypoints for f in usable])
    kmeans, representatives = build_codebook(descriptors, n_words)
    patches = np.stack(
        [cut_patch(usable[owner[r]].image, *keypoints[r]) for r in representatives]
    )

    # 3. Histograms and classifier
    report(len(dataset.paths) + 1, "Training the classifier")

    def histograms(indices: list[int]) -> np.ndarray:
        return np.array(
            [
                word_histogram(assign_words(kmeans, features[i].descriptors), n_words)
                for i in indices
            ]
        )

    x_train, y_train = histograms(train_idx), dataset.labels[train_idx]
    x_test, y_test = histograms(test_idx), dataset.labels[test_idx]
    classifier = SVC(kernel="rbf", probability=True, random_state=0)
    classifier.fit(x_train, y_train)
    predicted = classifier.classes_[classifier.predict_proba(x_test).argmax(axis=1)]
    accuracy = float((predicted == y_test).mean())

    model = BowModel(
        params=params,
        kmeans=kmeans,
        classifier=classifier,
        class_names=dataset.class_names,
        patches=patches,
        train_paths=[str(dataset.paths[i].resolve()) for i in train_idx],
        test_paths=[str(dataset.paths[i].resolve()) for i in test_idx],
        test_accuracy=accuracy,
        n_descriptors=len(descriptors),
    )
    return model, train_features


def save_model(model: BowModel, folder: Path, dataset: Dataset) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    state = asdict(model)
    state["params"] = asdict(model.params)
    state.pop("folder")
    target = folder / MODEL_FILE
    joblib.dump(state, target)

    # Human-readable list of the split, so test images are easy to find
    labels = {
        str(p.resolve()): dataset.class_names[label]
        for p, label in zip(dataset.paths, dataset.labels)
    }
    with open(folder / SPLIT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class", "path"])
        for split, paths in (("train", model.train_paths), ("test", model.test_paths)):
            for path in paths:
                writer.writerow([split, labels[path], path])
    model.folder = folder
    return target


def load_model(folder: Path) -> BowModel:
    target = folder / MODEL_FILE
    if not target.exists():
        raise FileNotFoundError(
            f"No Bag of Words model ({MODEL_FILE}) in:\n{folder}\n\n"
            "Build one first with the Codebook builder."
        )
    state = joblib.load(target)
    state["params"] = DescriptorParams(**state["params"])
    return BowModel(**state, folder=folder)


# ------------------------------------------------------------ 7.2 histogram


@dataclass(frozen=True)
class ImageWords:
    features: Features
    words: np.ndarray  # visual word index per keypoint
    histogram: np.ndarray  # normalised, length K


def compute_histogram(image: Image.Image, model: BowModel) -> ImageWords:
    features = image_features(image, model.params)
    words = assign_words(model.kmeans, features.descriptors)
    return ImageWords(features, words, word_histogram(words, model.n_words))


# ------------------------------------------------------- 7.3 classification


def classify(histogram: np.ndarray, model: BowModel) -> list[tuple[str, float]]:
    """Class probabilities, highest first."""
    probabilities = model.classifier.predict_proba(histogram.reshape(1, -1))[0]
    ranked = sorted(
        zip(model.classifier.classes_, probabilities), key=lambda item: -item[1]
    )
    return [(model.class_names[int(label)], float(p)) for label, p in ranked]


def true_class(path: Path | None, model: BowModel) -> str | None:
    """Class given by the image's folder name, when it is one of the classes."""
    if path is None or path.parent.name not in model.class_names:
        return None
    return path.parent.name


# ---------------------------------------------------------- visualisations


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def _word_colors(n_words: int) -> np.ndarray:
    return plt.cm.hsv(np.linspace(0, 1, n_words, endpoint=False))


def codebook_rows(n_words: int, columns: int = CODEBOOK_COLUMNS) -> int:
    """Rows of visual words in the codebook figure."""
    return -(-n_words // columns)


def draw_codebook(
    model: BowModel, samples: list[Features], columns: int = CODEBOOK_COLUMNS
) -> Image.Image:
    """Sample images with their keypoints (red) above the grid of visual words."""
    # Evenly spaced picks, so every class appears (paths are sorted by class)
    if len(samples) > N_SAMPLE_IMAGES:
        picks = np.linspace(0, len(samples) - 1, N_SAMPLE_IMAGES).astype(int)
        samples = [samples[i] for i in picks]
    sample_rows = -(-len(samples) // columns)
    word_rows = codebook_rows(model.n_words, columns)
    rows = sample_rows + word_rows

    fig = plt.figure(figsize=(2.2 * columns, 1.9 * rows + 0.6))
    grid = fig.add_gridspec(rows, columns, hspace=0.45, wspace=0.15)
    fig.suptitle(
        f"Codebook: {model.n_words} visual words ({model.params.kind})", fontsize=14
    )

    for i, sample in enumerate(samples):
        ax = fig.add_subplot(grid[i // columns, i % columns])
        ax.imshow(sample.image, cmap="gray")
        if len(sample.keypoints):
            ax.scatter(sample.keypoints[:, 1], sample.keypoints[:, 0], s=4, c="red")
        ax.axis("off")

    for word, patch in enumerate(model.patches):
        ax = fig.add_subplot(grid[sample_rows + word // columns, word % columns])
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"vw {word}", fontsize=9)
        ax.axis("off")
    return _figure_to_image(fig)


def link_sample(n_keypoints: int, limit: int = MAX_LINKS) -> np.ndarray:
    """Keypoints that get a line to their bar: a fixed random subset of ``limit``."""
    return np.random.default_rng(0).permutation(n_keypoints)[:limit]


def draw_histogram(
    image_words: ImageWords, model: BowModel, title: str = ""
) -> Image.Image:
    """Image with keypoints coloured by word, linked to its word histogram."""
    features = image_words.features
    words, histogram = image_words.words, image_words.histogram
    n_words = model.n_words
    colors = _word_colors(n_words)

    fig, (ax_image, ax_hist) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.9, wspace=0.12)

    ax_image.imshow(features.image, cmap="gray")
    ax_image.set_title(title or "query image")
    ax_image.axis("off")
    if len(words):
        ax_image.scatter(
            features.keypoints[:, 1],
            features.keypoints[:, 0],
            s=14,
            c=colors[words],
            edgecolors="black",
            linewidths=0.3,
        )

    positions = np.arange(n_words)
    ax_hist.bar(
        positions, histogram, color="lightgrey", edgecolor="black", linewidth=0.7
    )
    ax_hist.set_xlim(-0.7, n_words - 0.3)
    ax_hist.set_ylim(0, max(histogram.max() * 1.1, 0.01))
    ax_hist.set_title("Histogram of visual words")
    ax_hist.set_xticks(positions[:: 5 if n_words > 15 else 1])
    ax_hist.set_xlabel("Visual word index", labelpad=26)
    ax_hist.set_ylabel("Normalized frequency")

    # Visual word patches under the x axis
    zoom = min(0.5, 14 / n_words)
    for word in positions:
        thumbnail = OffsetImage(model.patches[word], cmap="gray", zoom=zoom)
        ax_hist.add_artist(
            AnnotationBbox(
                thumbnail,
                (word, 0),
                xybox=(0, -30),
                xycoords=("data", "axes fraction"),
                boxcoords="offset points",
                frameon=False,
                annotation_clip=False,
            )
        )

    # Thin lines from a sample of keypoints to the top of their word's bar
    for k in link_sample(len(words)):
        row, col = features.keypoints[k]
        word = int(words[k])
        fig.add_artist(
            ConnectionPatch(
                xyA=(col, row),
                coordsA=ax_image.transData,
                xyB=(word, histogram[word]),
                coordsB=ax_hist.transData,
                color=colors[word],
                alpha=0.35,
                linewidth=0.6,
            )
        )
    return _figure_to_image(fig)


def draw_prediction(image: Image.Image, label: str, probability: float) -> Image.Image:
    """The image with ``p(label) = 0.64`` written above it, as in the slides."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image.convert("RGB"))
    ax.set_title(f"p({label}) = {probability:.2f}", fontsize=16)
    ax.axis("off")
    fig.tight_layout()
    return _figure_to_image(fig)
