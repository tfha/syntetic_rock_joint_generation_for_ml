"""
Tests for finetune-specific data leakage validation.
"""

import pytest

from ml_segmentation.data_loading import validate_no_data_leakage


class TestFinetuneValidation:
    """Test FT-specific data leakage validation with stage splits."""

    def test_ft_no_leakage(self):
        """Test FT validation passes when no leakage exists."""
        train_synthetic = ["synth1.png", "synth2.png", "synth3.png"]
        train_real = ["real1.png", "real2.png"]
        combined = train_synthetic + train_real
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        # Should not raise
        validate_no_data_leakage(
            train_list=combined,
            val_list=val_list,
            test_list=test_list,
            experiment_name="test_ft",
            train_synthetic_list=train_synthetic,
            train_real_list=train_real,
        )

    def test_ft_synthetic_test_leakage(self):
        """Test FT validation detects synthetic-test overlap."""
        train_synthetic = ["synth1.png", "synth2.png", "test1.png"]  # test1 leaked
        train_real = ["real1.png", "real2.png"]
        combined = train_synthetic + train_real
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        with pytest.raises(ValueError, match="FT Stage 1: Synthetic"):
            validate_no_data_leakage(
                train_list=combined,
                val_list=val_list,
                test_list=test_list,
                experiment_name="test_ft",
                train_synthetic_list=train_synthetic,
                train_real_list=train_real,
            )

    def test_ft_real_test_leakage(self):
        """Test FT validation detects real-test overlap."""
        train_synthetic = ["synth1.png", "synth2.png", "synth3.png"]
        train_real = ["real1.png", "test1.png"]  # test1 leaked
        combined = train_synthetic + train_real
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        with pytest.raises(ValueError, match="FT Stage 2: Real"):
            validate_no_data_leakage(
                train_list=combined,
                val_list=val_list,
                test_list=test_list,
                experiment_name="test_ft",
                train_synthetic_list=train_synthetic,
                train_real_list=train_real,
            )

    def test_ft_combined_clean_but_stage_leaks(self):
        """Test that FT validation catches stage-specific leakage even when combined is clean."""
        # This scenario shouldn't normally happen, but tests the validation logic
        train_synthetic = ["synth1.png", "test1.png"]
        train_real = ["real1.png", "real2.png"]
        combined = train_synthetic + train_real
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        # Combined check would pass (test1 in combined, but that's expected)
        # But stage-specific check should catch synthetic-test overlap
        with pytest.raises(ValueError, match="FT Stage 1: Synthetic"):
            validate_no_data_leakage(
                train_list=combined,
                val_list=val_list,
                test_list=test_list,
                experiment_name="test_ft",
                train_synthetic_list=train_synthetic,
                train_real_list=train_real,
            )

    def test_sm_without_stage_lists(self):
        """Test SM validation works without stage-specific lists (backward compatibility)."""
        train_list = ["train1.png", "train2.png", "train3.png"]
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        # Should not raise (no stage lists provided = SM mode)
        validate_no_data_leakage(
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
            experiment_name="test_sm",
        )

    def test_sm_train_test_leakage(self):
        """Test SM validation detects train-test overlap."""
        train_list = ["train1.png", "train2.png", "test1.png"]  # test1 leaked
        val_list = ["val1.png", "val2.png"]
        test_list = ["test1.png", "test2.png"]

        with pytest.raises(ValueError, match="train and test"):
            validate_no_data_leakage(
                train_list=train_list,
                val_list=val_list,
                test_list=test_list,
                experiment_name="test_sm",
            )

    def test_simplemixed_val_test_same_allowed(self):
        """Test SimpleMixed validation allows val=test when flag is set."""
        train_list = ["train1.png", "train2.png", "train3.png"]
        test_list = ["test1.png", "test2.png"]
        val_list = test_list  # val and test are the same (Wachter strategy)

        # Should not raise when allow_val_test_overlap=True
        validate_no_data_leakage(
            train_list=train_list,
            val_list=val_list,
            test_list=test_list,
            experiment_name="simplemixed_box_50",
            allow_val_test_overlap=True,
        )

    def test_simplemixed_train_test_leakage_still_detected(self):
        """Test SimpleMixed validation still detects train-test overlap even with val=test allowed."""
        train_list = ["train1.png", "train2.png", "test1.png"]  # test1 leaked
        test_list = ["test1.png", "test2.png"]
        val_list = test_list  # val and test are the same

        # When val=test, train-val overlap also means train-test overlap
        # The validation will detect train-val overlap first
        with pytest.raises(ValueError, match="train and validation"):
            validate_no_data_leakage(
                train_list=train_list,
                val_list=val_list,
                test_list=test_list,
                experiment_name="simplemixed_box_50",
                allow_val_test_overlap=True,
            )
