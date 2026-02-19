"""Drift detection algorithms for ML pipelines."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import DriftDetectionError, InsufficientDataError
from .models import DriftResult, DriftType

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30


class DriftDetector:
    """Detects data and target drift using statistical tests."""

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize drift detector.

        Args:
            threshold: p-value threshold for significance (default 0.05)

        Raises:
            ValueError: if threshold is not in (0, 1)
        """
        if not 0 < threshold < 1:
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold

    def detect_numerical_drift(
        self,
        baseline: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift in numerical feature using Kolmogorov-Smirnov test.

        Args:
            baseline: baseline distribution (reference)
            current: current distribution
            feature_name: name of the feature

        Returns:
            DriftResult with test results

        Raises:
            InsufficientDataError: if fewer than MIN_SAMPLES
        """
        if len(baseline) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need at least {MIN_SAMPLES} samples, got baseline={len(baseline)}, "
                f"current={len(current)}"
            )

        # Remove NaNs
        baseline_clean = baseline.dropna().values
        current_clean = current.dropna().values

        if len(baseline_clean) < MIN_SAMPLES or len(current_clean) < MIN_SAMPLES:
            raise InsufficientDataError(
                "Insufficient non-null samples after cleaning"
            )

        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(baseline_clean, current_clean)

        return DriftResult(
            drift_detected=p_value < self.threshold,
            drift_type=DriftType.FEATURE_DRIFT,
            p_value=p_value,
            statistic=statistic,
            feature_name=feature_name,
            threshold=self.threshold,
        )

    def detect_categorical_drift(
        self,
        baseline: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """Detect drift in categorical feature using Chi-square test.

        Args:
            baseline: baseline distribution (reference)
            current: current distribution
            feature_name: name of the feature

        Returns:
            DriftResult with test results

        Raises:
            InsufficientDataError: if fewer than MIN_SAMPLES
        """
        if len(baseline) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need at least {MIN_SAMPLES} samples, got baseline={len(baseline)}, "
                f"current={len(current)}"
            )

        # Get value counts and normalize to proportions
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()

        # Align on all categories seen in either distribution
        all_categories = sorted(set(baseline_counts.index) | set(current_counts.index))
        baseline_aligned = np.array([baseline_counts.get(cat, 0) for cat in all_categories])
        current_aligned = np.array([current_counts.get(cat, 0) for cat in all_categories])

        if (baseline_aligned == 0).all() or (current_aligned == 0).all():
            raise DriftDetectionError("Cannot perform chi-square test with zero counts")

        # Normalize baseline to expected frequencies matching current sample size
        baseline_props = baseline_aligned / baseline_aligned.sum()
        expected_counts = baseline_props * current_aligned.sum()

        # Chi-square test comparing observed to expected
        chi2, p_value = stats.chisquare(current_aligned, expected_counts)

        return DriftResult(
            drift_detected=p_value < self.threshold,
            drift_type=DriftType.FEATURE_DRIFT,
            p_value=p_value,
            statistic=chi2,
            feature_name=feature_name,
            threshold=self.threshold,
        )

    def detect_target_drift(
        self,
        baseline_target: pd.Series,
        current_target: pd.Series,
    ) -> DriftResult:
        """Detect drift in target variable.

        Args:
            baseline_target: baseline target distribution
            current_target: current target distribution

        Returns:
            DriftResult with test results
        """
        # Check if target is numerical or categorical
        if pd.api.types.is_numeric_dtype(baseline_target):
            result = self.detect_numerical_drift(
                baseline_target,
                current_target,
                feature_name="target",
            )
        else:
            result = self.detect_categorical_drift(
                baseline_target,
                current_target,
                feature_name="target",
            )

        result.drift_type = DriftType.TARGET_DRIFT
        return result

    def detect_multivariate_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, DriftResult]:
        """Detect drift across multiple features.

        Args:
            baseline_df: baseline dataframe
            current_df: current dataframe
            feature_columns: features to check (default: all numeric columns)

        Returns:
            Dictionary mapping feature names to DriftResult
        """
        if feature_columns is None:
            feature_columns = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        for feature in feature_columns:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature {feature} not found in both dataframes")
                continue

            try:
                if pd.api.types.is_numeric_dtype(baseline_df[feature]):
                    result = self.detect_numerical_drift(
                        baseline_df[feature],
                        current_df[feature],
                        feature,
                    )
                else:
                    result = self.detect_categorical_drift(
                        baseline_df[feature],
                        current_df[feature],
                        feature,
                    )
                results[feature] = result
            except (InsufficientDataError, DriftDetectionError) as e:
                logger.warning(f"Could not detect drift for {feature}: {e}")

        return results


def calculate_feature_importance_drift(
    baseline_importances: Dict[str, float],
    current_importances: Dict[str, float],
) -> float:
    """Calculate drift in feature importance rankings using Spearman correlation.

    Args:
        baseline_importances: feature importance from baseline model
        current_importances: feature importance from current model

    Returns:
        Spearman correlation coefficient (closer to 1 means no drift)
    """
    # Common features
    features = set(baseline_importances.keys()) & set(current_importances.keys())

    if len(features) < 2:
        return 0.0

    baseline_vals = np.array([baseline_importances[f] for f in features])
    current_vals = np.array([current_importances[f] for f in features])

    if np.std(baseline_vals) == 0 or np.std(current_vals) == 0:
        return 1.0

    corr, _ = stats.spearmanr(baseline_vals, current_vals)
    return float(corr)
