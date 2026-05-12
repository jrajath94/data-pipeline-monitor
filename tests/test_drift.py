"""Tests for drift detection."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline_monitor import (
    DriftDetector,
    DriftType,
    InsufficientDataError,
)


class TestDriftDetector:
    """Test DriftDetector class."""

    def test_init_valid_threshold(self) -> None:
        """Test initialization with valid threshold."""
        detector = DriftDetector(threshold=0.05)
        assert detector.threshold == 0.05

    def test_init_invalid_threshold(self) -> None:
        """Test initialization with invalid threshold."""
        with pytest.raises(ValueError, match="threshold"):
            DriftDetector(threshold=1.5)

        with pytest.raises(ValueError, match="threshold"):
            DriftDetector(threshold=0.0)

    def test_detect_numerical_drift_no_drift(
        self,
        sample_baseline: pd.DataFrame,
        sample_current_no_drift: pd.DataFrame,
    ) -> None:
        """Test numerical drift detection when no drift present."""
        detector = DriftDetector()
        result = detector.detect_numerical_drift(
            sample_baseline["feature1"],
            sample_current_no_drift["feature1"],
            "feature1",
        )

        assert not result.drift_detected
        assert result.drift_type == DriftType.FEATURE_DRIFT
        assert result.feature_name == "feature1"
        assert result.p_value > 0.05

    def test_detect_numerical_drift_with_drift(
        self,
        sample_baseline: pd.DataFrame,
        sample_current_with_drift: pd.DataFrame,
    ) -> None:
        """Test numerical drift detection when drift is present."""
        detector = DriftDetector()
        result = detector.detect_numerical_drift(
            sample_baseline["feature1"],
            sample_current_with_drift["feature1"],
            "feature1",
        )

        assert result.drift_detected
        assert result.drift_type == DriftType.FEATURE_DRIFT
        assert result.p_value < 0.05

    def test_detect_numerical_drift_insufficient_samples(
        self,
        sample_baseline: pd.DataFrame,
    ) -> None:
        """Test error when insufficient samples."""
        detector = DriftDetector()
        small_sample = sample_baseline["feature1"].head(10)

        with pytest.raises(InsufficientDataError):
            detector.detect_numerical_drift(
                sample_baseline["feature1"],
                small_sample,
                "feature1",
            )

    def test_detect_categorical_drift_no_drift(
        self,
        sample_baseline: pd.DataFrame,
        sample_current_no_drift: pd.DataFrame,
    ) -> None:
        """Test categorical drift detection when no drift present."""
        detector = DriftDetector()
        result = detector.detect_categorical_drift(
            sample_baseline["feature3"],
            sample_current_no_drift["feature3"],
            "feature3",
        )

        assert not result.drift_detected
        assert result.drift_type == DriftType.FEATURE_DRIFT

    def test_detect_categorical_drift_with_drift(
        self,
        sample_baseline: pd.DataFrame,
        sample_current_with_drift: pd.DataFrame,
    ) -> None:
        """Test categorical drift detection when drift is present."""
        detector = DriftDetector()
        result = detector.detect_categorical_drift(
            sample_baseline["feature3"],
            sample_current_with_drift["feature3"],
            "feature3",
        )

        assert result.drift_detected
        assert result.drift_type == DriftType.FEATURE_DRIFT

    def test_detect_target_drift_numerical(
        self,
        sample_baseline: pd.DataFrame,
        sample_target: pd.Series,
        sample_target_with_drift: pd.Series,
    ) -> None:
        """Test target drift detection for numerical target."""
        np.random.seed(42)
        baseline_target = pd.Series(np.random.normal(5, 1, 200))
        current_target = pd.Series(np.random.normal(6.5, 1, 150))

        detector = DriftDetector()
        result = detector.detect_target_drift(baseline_target, current_target)

        assert result.drift_detected
        assert result.drift_type == DriftType.TARGET_DRIFT

    def test_detect_multivariate_drift(
        self,
        sample_baseline: pd.DataFrame,
        sample_current_with_drift: pd.DataFrame,
    ) -> None:
        """Test multivariate drift detection."""
        detector = DriftDetector()
        results = detector.detect_multivariate_drift(
            sample_baseline,
            sample_current_with_drift,
            feature_columns=["feature1", "feature2"],
        )

        assert "feature1" in results
        assert "feature2" in results
        # feature1 should show drift
        assert results["feature1"].drift_detected

    @pytest.mark.parametrize(
        ("threshold", "should_detect"),
        [
            (0.01, True),  # Very strict, should detect strong drift
            (0.5, True),  # Loose, should also detect strong drift with p<0.5
        ],
    )
    def test_detect_numerical_drift_threshold_sensitivity(
        self,
        threshold: float,
        should_detect: bool,
        sample_baseline: pd.DataFrame,
        sample_current_with_drift: pd.DataFrame,
    ) -> None:
        """Test that drift detection respects threshold."""
        detector = DriftDetector(threshold=threshold)
        result = detector.detect_numerical_drift(
            sample_baseline["feature1"],
            sample_current_with_drift["feature1"],
            "feature1",
        )

        # Both thresholds should detect the strong drift in our test data
        assert result.drift_detected == should_detect


class TestFeatureImportanceDrift:
    """Test feature importance drift calculation."""

    def test_feature_importance_drift_no_drift(self) -> None:
        """Test when feature importance hasn't drifted."""
        from data_pipeline_monitor.drift import calculate_feature_importance_drift

        baseline = {"a": 0.5, "b": 0.3, "c": 0.2}
        current = {"a": 0.5, "b": 0.3, "c": 0.2}

        drift = calculate_feature_importance_drift(baseline, current)
        assert drift == 1.0  # Perfect correlation

    def test_feature_importance_drift_with_drift(self) -> None:
        """Test when feature importance has drifted."""
        from data_pipeline_monitor.drift import calculate_feature_importance_drift

        baseline = {"a": 0.5, "b": 0.3, "c": 0.2}
        current = {"a": 0.1, "b": 0.2, "c": 0.7}  # Completely different

        drift = calculate_feature_importance_drift(baseline, current)
        assert drift < 1.0  # Should be low correlation
        assert drift >= -1.0  # Correlation always >= -1
