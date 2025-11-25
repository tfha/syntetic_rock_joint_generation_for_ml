"""Tests for count-based dataset splitting functionality."""

from pathlib import Path

import pytest

from ml_segmentation.data_loading import split_data


class TestCountBasedSplits:
    """Test suite for count-based dataset splitting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        # Create 100 sample file names
        self.files = [f"image_{i}.png" for i in range(100)]

    def test_count_based_split_basic(self, monkeypatch, tmp_path):
        """Test basic count-based splitting."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.files[:],
            self.files[:],
            train_count=50,
            val_count=20,
            test_count=30,
        )

        # Assert
        assert len(train_list) == 50
        assert len(val_list) == 20
        assert len(test_list) == 30

    def test_count_based_split_small_dataset(self, monkeypatch, tmp_path):
        """Test count-based splitting with small dataset."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.files[:],
            self.files[:],
            train_count=10,
            val_count=5,
            test_count=5,
        )

        # Assert
        assert len(train_list) == 10
        assert len(val_list) == 5
        assert len(test_list) == 5

    def test_count_exceeds_available_data(self, monkeypatch, tmp_path):
        """Test that requesting more data than available raises error."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act & Assert
        with pytest.raises(
            ValueError, match="Requested counts .* exceed available files"
        ):
            split_data(
                self.files[:],
                self.files[:],
                train_count=80,
                val_count=50,
                test_count=50,  # Total 180 > 100 available
            )

    def test_mixing_fractions_and_counts_raises_error(self, monkeypatch, tmp_path):
        """Test that mixing fractions and counts raises error."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot mix fraction-based and count-based"
        ):
            split_data(
                self.files[:],
                self.files[:],
                train_frac=0.8,  # Using fraction
                train_count=50,  # And count - should fail
            )

    def test_disjoint_datasets_with_counts(self, monkeypatch, tmp_path):
        """Test count-based splitting with disjoint train/test sets."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        train_files = [f"train_{i}.png" for i in range(80)]
        test_files = [f"test_{i}.png" for i in range(20)]

        # Act
        train_list, val_list, test_list = split_data(
            train_files,
            test_files,
            train_count=50,
            val_count=10,
            test_count=15,
        )

        # Assert
        assert len(train_list) == 50
        assert len(val_list) == 10
        assert len(test_list) == 15
        # All test samples should come from test_files
        assert all(f.startswith("test_") for f in test_list)
