"""
Tests for data loading module of the ml_segmentation package.

This module contains tests for the functions in the data_loading.py module,
specifically focusing on the split_data function that divides datasets into
train, validation, and test sets.

To run all tests, use the following command from the project root directory:
    pytest -v

To run the tests for this script specifically, use the following command from the project root directory:
    pytest tests/test_data_loading.py -v

For a specific test:
    pytest tests/test_data_loading.py::TestSplitData::test_zero_validation_split -v

For test coverage report:
    pytest tests/test_data_loading.py --cov=ml_segmentation.data_loading
"""

import json
import random
from pathlib import Path

import pytest

from ml_segmentation.data_loading import split_data


class TestSplitData:
    """Test suite for the split_data function."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and tear down after the test."""
        # Create predictable test data
        self.train_files = [f"train_image_{i}.png" for i in range(100)]
        self.test_files = [f"test_image_{i}.png" for i in range(20)]
        self.same_files = [f"same_image_{i}.png" for i in range(100)]

        # Set a fixed seed for reproducibility
        random.seed(42)

        yield

        # Reset random seed
        random.seed()

    def test_standard_split(self, monkeypatch, tmp_path):
        """Test standard case with train/val/test split with default fractions."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.same_files[:],
            self.same_files[:],
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )

        # Assert
        assert len(train_list) == 80
        assert len(val_list) == 10
        assert len(test_list) == 10

        # Check for no duplicates between splits
        assert len(set(train_list) & set(val_list)) == 0
        assert len(set(train_list) & set(test_list)) == 0
        assert len(set(val_list) & set(test_list)) == 0

        # Check that files were saved to disk
        assert (raw_dir / "all_files.json").exists()
        assert (model_ready_dir / "train_files.json").exists()
        assert (model_ready_dir / "val_files.json").exists()
        assert (model_ready_dir / "test_files.json").exists()

    def test_different_files_split(self, monkeypatch, tmp_path):
        """Test case where train_files and test_files are different."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.train_files[:],
            self.test_files[:],
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )

        # Assert
        # When files are different, val comes from train and test remains as provided
        assert len(val_list) == 11  # ~10% of 100 train files
        assert len(train_list) == 89  # remaining train files
        assert test_list == self.test_files  # test files remain unchanged

        # Check for no duplicates between splits
        assert len(set(train_list) & set(val_list)) == 0
        assert len(set(train_list) & set(test_list)) == 0
        assert len(set(val_list) & set(test_list)) == 0

    def test_zero_validation_split(self, monkeypatch, tmp_path):
        """Test case where val_frac = 0 (no validation set)."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.same_files[:],
            self.same_files[:],
            train_frac=0.9,
            val_frac=0.0,
            test_frac=0.1,
        )

        # Assert
        assert len(train_list) == 90
        assert len(val_list) == 0  # Validation set should be empty
        assert len(test_list) == 10

        # Check for no duplicates between splits
        assert len(set(train_list) & set(test_list)) == 0

    def test_zero_validation_different_files(self, monkeypatch, tmp_path):
        """Test case where val_frac = 0 and train/test files are different."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.train_files[:],
            self.test_files[:],
            train_frac=0.9,
            val_frac=0.0,
            test_frac=0.1,
        )

        # Assert
        assert len(train_list) == 100  # All train files stay in train
        assert len(val_list) == 0  # No validation set
        assert test_list == self.test_files

    def test_custom_split_fractions(self, monkeypatch, tmp_path):
        """Test with custom split fractions."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.same_files[:],
            self.same_files[:],
            train_frac=0.7,
            val_frac=0.2,
            test_frac=0.1,
        )

        # Assert
        assert len(train_list) == 70
        assert len(val_list) == 20
        assert len(test_list) == 10

    def test_invalid_fractions(self):
        """Test that an error is raised when fractions don't add up to 1.0."""
        # Act & Assert
        with pytest.raises(ValueError, match="Fractions must sum to 1.0."):
            split_data(
                self.same_files[:],
                self.same_files[:],
                train_frac=0.7,
                val_frac=0.2,
                test_frac=0.2,
            )

    def test_duplicate_files(self, monkeypatch, tmp_path):
        """Test that an error is raised when input files contain duplicates."""
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

        # Create input with duplicates
        files_with_duplicates = self.same_files[:] + [self.same_files[0]]

        # Act & Assert
        with pytest.raises(ValueError, match="Train set contains duplicate files"):
            split_data(
                files_with_duplicates,
                files_with_duplicates,
                train_frac=0.8,
                val_frac=0.1,
                test_frac=0.1,
            )

    def test_verify_file_contents(self, monkeypatch, tmp_path):
        """Test that correct file contents are saved to disk."""
        # Arrange
        model_ready_dir = tmp_path / "model_ready"
        model_ready_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Mock Path to return our directories
        def mock_path(path_str):
            if path_str == "data/raw":
                return raw_dir
            elif path_str == "data/model_ready":
                return model_ready_dir
            return Path(path_str)

        monkeypatch.setattr("ml_segmentation.data_loading.Path", mock_path)

        # Act
        train_list, val_list, test_list = split_data(
            self.same_files[:],
            self.same_files[:],
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )

        # Assert
        # Verify file contents
        with open(raw_dir / "all_files.json") as f:
            all_files = json.load(f)
            assert len(all_files) == 200  # Same files are duplicated

        with open(model_ready_dir / "train_files.json") as f:
            saved_train = json.load(f)
            assert saved_train == train_list

        with open(model_ready_dir / "val_files.json") as f:
            saved_val = json.load(f)
            assert saved_val == val_list

        with open(model_ready_dir / "test_files.json") as f:
            saved_test = json.load(f)
            assert saved_test == test_list
