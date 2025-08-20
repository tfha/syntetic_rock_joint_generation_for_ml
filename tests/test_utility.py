"""
Tests for utility module of the ml_segmentation package.

This module contains tests for the functions in the utility.py module.

To run all tests, use the following command from the project root directory:
    pytest -v

To run the tests for this script specifically, use the following command from the project root directory:
    pytest tests/test_utility.py -v

For a specific test:
    pytest tests/test_utility.py::TestCreateResultsTable::test_standard_table -v

For test coverage report:
    pytest tests/test_utility.py --cov=ml_segmentation.utility
"""

from pathlib import Path

from rich.table import Table

from ml_segmentation.utility import create_results_table, modify_filepath


class TestCreateResultsTable:
    """Test suite for the create_results_table function."""

    def test_standard_table(self):
        """Test creation of a standard table with training metrics."""
        # Arrange
        epoch = 0
        metrics: dict[str, float] = {"loss": 0.5, "accuracy": 0.85, "iou": 0.75}
        session = "Training"

        # Act
        table = create_results_table(epoch, metrics, session)

        # Assert
        assert isinstance(table, Table)
        assert table.title == "Epoch 1 Training Results"
        assert table.columns[0].header == "Metric"
        assert table.columns[1].header == "Value"

        # Unfortunately, we can't directly check the rows in a Table object
        # But we can check if the table has been created with no errors

    def test_validation_session(self):
        """Test creation of a table with validation metrics."""
        # Arrange
        epoch = 3
        metrics: dict[str, float] = {"loss": 0.4, "accuracy": 0.9, "iou": 0.8}
        session = "Validation"

        # Act
        table = create_results_table(epoch, metrics, session)

        # Assert
        assert isinstance(table, Table)
        assert table.title == "Epoch 4 Validation Results"  # Epoch + 1 for display

    def test_empty_metrics(self):
        """Test with empty metrics dictionary."""
        # Arrange
        epoch = 0
        metrics: dict[str, float] = {}

        # Act
        table = create_results_table(epoch, metrics)

        # Assert
        assert isinstance(table, Table)
        assert table.title == "Epoch 1 Training Results"
        # Table should be created but have no rows

    def test_custom_session_name(self):
        """Test with a custom session name."""
        # Arrange
        epoch = 1
        metrics: dict[str, float] = {"loss": 0.3}
        session = "Testing"

        # Act
        table = create_results_table(epoch, metrics, session)

        # Assert
        assert isinstance(table, Table)
        assert table.title == "Epoch 2 Testing Results"

    def test_metrics_formatting(self):
        """Test that metrics values are properly formatted to 2 decimal places."""
        # Arrange
        epoch = 0
        metrics: dict[str, float] = {"precision": 0.12345, "recall": 0.98765}

        # Act
        table = create_results_table(epoch, metrics)

        # Assert
        # We can't directly test row content, but we know table creation
        # should succeed without errors
        assert isinstance(table, Table)


class TestModifyFilepath:
    """Test suite for the modify_filepath function."""

    def test_standard_filepath_modification(self):
        """Test modifying a standard filepath with an endsection."""
        # Arrange
        original_path = Path("/data/train.csv")
        endsection = "_modified"
        expected_path = Path("/data/train_modified.csv")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path

    def test_filepath_with_multiple_dots(self):
        """Test modifying a filepath that has multiple dots in the filename."""
        # Arrange
        original_path = Path("/data/train.data.csv")
        endsection = "_processed"
        expected_path = Path("/data/train.data_processed.csv")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path

    def test_filepath_without_extension(self):
        """Test modifying a filepath without an extension."""
        # Arrange
        original_path = Path("/data/train")
        endsection = "_modified"
        expected_path = Path("/data/train_modified")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path

    def test_filepath_with_empty_endsection(self):
        """Test modifying a filepath with an empty endsection."""
        # Arrange
        original_path = Path("/data/train.csv")
        endsection = ""
        expected_path = Path("/data/train.csv")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path

    def test_filepath_with_directories(self):
        """Test modifying a filepath with multiple directory levels."""
        # Arrange
        original_path = Path("/root/sub/data/train.csv")
        endsection = "_v2"
        expected_path = Path("/root/sub/data/train_v2.csv")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path

    def test_windows_style_filepath(self):
        """Test modifying a Windows-style filepath."""
        # Arrange
        original_path = Path(r"C:\Users\data\train.csv")
        endsection = "_processed"
        expected_path = Path(r"C:\Users\data\train_processed.csv")

        # Act
        modified_path = modify_filepath(original_path, endsection)

        # Assert
        assert modified_path == expected_path
