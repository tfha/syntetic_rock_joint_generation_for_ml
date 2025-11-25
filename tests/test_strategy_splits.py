"""Tests for per-strategy split configuration."""

from ml_segmentation.azure_data_assets import get_split_config_for_strategy


class TestStrategySplits:
    """Test suite for strategy-specific split configuration."""

    def test_get_split_config_uses_strategy_specific_fractions(self):
        """Test that strategy-specific fractions override global settings."""
        strategy_splits = {
            "verification_box": {
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
            }
        }

        result = get_split_config_for_strategy(
            experiment_strategy="verification_box",
            strategy_splits=strategy_splits,
            global_train_fraction=0.8,
            global_val_fraction=0.1,
            global_test_fraction=0.1,
            global_train_count=None,
            global_val_count=None,
            global_test_count=None,
        )

        assert result == (0.7, 0.15, 0.15, None, None, None)

    def test_get_split_config_uses_strategy_specific_counts(self):
        """Test that strategy-specific counts override global settings."""
        strategy_splits = {
            "main_objective_dfn_rock_slope": {
                "train_count": 150,
                "val_count": 25,
                "test_count": 50,
            }
        }

        result = get_split_config_for_strategy(
            experiment_strategy="main_objective_dfn_rock_slope",
            strategy_splits=strategy_splits,
            global_train_fraction=0.8,
            global_val_fraction=0.1,
            global_test_fraction=0.1,
            global_train_count=None,
            global_val_count=None,
            global_test_count=None,
        )

        assert result == (None, None, None, 150, 25, 50)

    def test_get_split_config_falls_back_when_no_strategy_config(self):
        """Test fallback to global when no strategy-specific config."""
        strategy_splits = {
            "verification_box": {
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
            }
        }

        result = get_split_config_for_strategy(
            experiment_strategy="other_strategy",
            strategy_splits=strategy_splits,
            global_train_fraction=0.8,
            global_val_fraction=0.1,
            global_test_fraction=0.1,
            global_train_count=None,
            global_val_count=None,
            global_test_count=None,
        )

        assert result == (0.8, 0.1, 0.1, None, None, None)

    def test_get_split_config_falls_back_when_strategy_splits_is_none(self):
        """Test that global settings are used when strategy_splits is None."""
        result = get_split_config_for_strategy(
            experiment_strategy="verification_box",
            strategy_splits=None,
            global_train_fraction=0.8,
            global_val_fraction=0.1,
            global_test_fraction=0.1,
            global_train_count=None,
            global_val_count=None,
            global_test_count=None,
        )

        assert result == (0.8, 0.1, 0.1, None, None, None)

    def test_get_split_config_with_global_counts(self):
        """Test fallback to global count-based settings."""
        result = get_split_config_for_strategy(
            experiment_strategy="verification_box",
            strategy_splits=None,
            global_train_fraction=None,
            global_val_fraction=None,
            global_test_fraction=None,
            global_train_count=100,
            global_val_count=20,
            global_test_count=30,
        )

        assert result == (None, None, None, 100, 20, 30)
