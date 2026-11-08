from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from forensics_app.tools.eigenface_builder import EigenfaceBuilderTool


class TestEigenfaceBuilder:

    def create_dataset(self):
        rng = np.random.RandomState(42)

        images = rng.rand(12, 8, 8).astype(np.float32)
        labels = np.array([0] * 6 + [1] * 6)

        return SimpleNamespace(
            images=images,
            data=images.reshape(12, -1),
            target=labels,
            target_names=np.array(["Person A", "Person B"]),
        )

    @patch("forensics_app.tools.eigenface_builder.messagebox.showerror")
    @patch("forensics_app.tools.eigenface_builder.simpledialog.askinteger")
    @patch("forensics_app.tools.eigenface_builder.filedialog.askdirectory")
    @patch("forensics_app.tools.eigenface_builder.messagebox.showinfo")
    @patch("forensics_app.tools.eigenface_builder.fetch_lfw_people")
    def test_builds_eigenface_model(
        self,
        mock_fetch,
        mock_showinfo,
        mock_directory,
        mock_askinteger,
        mock_showerror,
        tmp_path,
        monkeypatch,
    ):
        output = tmp_path / "eigenface_output"
        output.mkdir()

        mock_fetch.return_value = self.create_dataset()
        mock_directory.return_value = str(output)
        mock_askinteger.return_value = 3

        # Restore the original SSL context after the test.
        import ssl

        original_ssl = ssl._create_default_https_context
        monkeypatch.setattr(
            ssl,
            "_create_default_https_context",
            original_ssl,
        )

        tool = EigenfaceBuilderTool()
        result = tool.run(None, None)

        assert result is not None
        assert result.details["Operation"] == "Eigenface builder"
        assert result.details["Dataset"] == "LFW People"
        assert result.details["Eigenfaces"] == 3
        assert result.details["Eigenfaces shown"] == 3

        # Check generated folders.
        assert (output / "train").is_dir()
        assert (output / "test").is_dir()
        assert (output / "model").is_dir()

        # Check saved images and CSV.
        assert (output / "image_labels.csv").exists()
        assert len(list((output / "train").glob("*.jpg"))) == 9
        assert len(list((output / "test").glob("*.jpg"))) == 3

        # Check saved PCA model files.
        model = output / "model"

        expected_files = [
            "mean_face.npy",
            "eigenfaces.npy",
            "train_coefficients.npy",
            "test_coefficients.npy",
            "train_labels.npy",
            "test_labels.npy",
            "target_names.npy",
            "explained_variance_ratio.npy",
            "explained_variance.npy",
        ]

        for filename in expected_files:
            assert (model / filename).exists()

        # Check eigenface dimensions.
        eigenfaces = np.load(model / "eigenfaces.npy")
        assert eigenfaces.shape == (3, 8, 8)

        # Check visualization.
        assert (output / "eigenfaces.png").exists()
        assert result.image is not None

        mock_showerror.assert_not_called()

    @patch("forensics_app.tools.eigenface_builder.filedialog.askdirectory")
    @patch("forensics_app.tools.eigenface_builder.messagebox.showinfo")
    def test_cancelled_folder_selection(
        self,
        mock_showinfo,
        mock_directory,
    ):
        mock_directory.return_value = ""

        tool = EigenfaceBuilderTool()
        result = tool.run(None, None)

        assert result is None
