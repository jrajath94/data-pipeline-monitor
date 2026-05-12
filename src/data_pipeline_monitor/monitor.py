"""Main pipeline monitoring system."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .drift import DriftDetector
from .exceptions import DriftDetectionError, InsufficientDataError, MonitoringConfigError
from .metrics import MetricsExporter
from .models import (
    AlertSeverity,
    DataQualityMetrics,
    MonitoringAlert,
    MonitoringConfig,
    PipelineMetrics,
)

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Real-time monitoring system for ML pipelines with drift detection."""

    def __init__(self, config: MonitoringConfig) -> None:
        """Initialize pipeline monitor.

        Args:
            config: MonitoringConfig with pipeline settings

        Raises:
            MonitoringConfigError: if config is invalid
        """
        if not config.validate():
            raise MonitoringConfigError("Invalid monitoring configuration")

        self.config = config
        self.detector = DriftDetector(threshold=config.drift_threshold)
        self.exporter = MetricsExporter(config.pipeline_name)

        self.baseline_data: Optional[pd.DataFrame] = None
        self.baseline_target: Optional[pd.Series] = None
        self.alerts: list[MonitoringAlert] = []

        logger.info(f"Initialized monitor for pipeline: {config.pipeline_name}")

    def set_baseline(
        self,
        baseline_df: pd.DataFrame,
        baseline_target: Optional[pd.Series] = None,
    ) -> None:
        """Set baseline distribution for drift detection.

        Args:
            baseline_df: baseline feature data
            baseline_target: baseline target data (optional)

        Raises:
            MonitoringConfigError: if baseline too small
        """
        if len(baseline_df) < self.config.min_baseline_samples:
            raise MonitoringConfigError(
                f"Baseline has {len(baseline_df)} samples, "
                f"need at least {self.config.min_baseline_samples}"
            )

        self.baseline_data = baseline_df.copy()
        self.baseline_target = baseline_target.copy() if baseline_target is not None else None

        logger.info(
            f"Baseline set with {len(baseline_df)} samples and "
            f"{baseline_df.shape[1]} features"
        )

    def check_data_quality(self, batch_df: pd.DataFrame) -> DataQualityMetrics:
        """Check data quality of a batch.

        Args:
            batch_df: batch of data to check

        Returns:
            DataQualityMetrics with quality assessment
        """
        n_rows = len(batch_df)

        # Missing values
        missing_count = batch_df.isnull().sum().sum()
        missing_pct = (missing_count / (n_rows * batch_df.shape[1])) * 100 if n_rows > 0 else 0

        # Duplicates
        duplicate_count = batch_df.duplicated().sum()
        duplicate_pct = (duplicate_count / n_rows) * 100 if n_rows > 0 else 0

        # Null features
        null_features = batch_df.columns[batch_df.isnull().any()].tolist()

        # Feature correlations
        numeric_df = batch_df.select_dtypes(include=["number"])
        correlations = {}
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            # Get high correlations (exclude diagonal)
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i + 1 :]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.9:
                        correlations[f"{col1}-{col2}"] = float(corr)

        metrics = DataQualityMetrics(
            missing_values_pct=missing_pct,
            duplicate_rows_pct=duplicate_pct,
            null_features=null_features,
            feature_correlations=correlations,
        )

        # Record metrics
        self.exporter.record_data_quality(missing_pct, duplicate_pct)

        # Alert if quality is poor
        if not metrics.is_healthy(
            self.config.max_missing_pct,
            self.config.max_duplicate_pct,
        ):
            alert = MonitoringAlert(
                severity=AlertSeverity.WARNING,
                message=f"Data quality degradation: {missing_pct:.2f}% missing, "
                f"{duplicate_pct:.2f}% duplicates",
                alert_type="data_quality",
                metadata={
                    "missing_pct": missing_pct,
                    "duplicate_pct": duplicate_pct,
                    "null_features": null_features,
                },
            )
            self.alerts.append(alert)
            self.exporter.record_alert(AlertSeverity.WARNING)

        return metrics

    def detect_feature_drift(self, batch_df: pd.DataFrame) -> dict[str, Any]:
        """Detect drift in features.

        Args:
            batch_df: current batch of data

        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set")

        features_to_check = (
            self.config.features_to_monitor
            if self.config.features_to_monitor
            else self.baseline_data.columns.tolist()
        )

        results = {}
        drift_count = 0

        for feature in features_to_check:
            if feature not in self.baseline_data.columns or feature not in batch_df.columns:
                continue

            try:
                result = (
                    self.detector.detect_numerical_drift(
                        self.baseline_data[feature],
                        batch_df[feature],
                        feature,
                    )
                    if feature not in self.config.categorical_features
                    else self.detector.detect_categorical_drift(
                        self.baseline_data[feature],
                        batch_df[feature],
                        feature,
                    )
                )

                results[feature] = result
                self.exporter.record_drift_result(result)

                if result.drift_detected:
                    drift_count += 1
                    alert = MonitoringAlert(
                        severity=AlertSeverity.WARNING,
                        message=f"Drift detected in feature '{feature}' "
                        f"(p-value: {result.p_value:.4f})",
                        alert_type="feature_drift",
                        metadata={
                            "feature": feature,
                            "p_value": result.p_value,
                            "statistic": result.statistic,
                        },
                    )
                    self.alerts.append(alert)
                    self.exporter.record_alert(AlertSeverity.WARNING)

            except (InsufficientDataError, DriftDetectionError) as e:
                logger.warning(f"Could not detect drift for {feature}: {e}")

        self.exporter.record_drift_summary(drift_count)

        return results

    def detect_target_drift(self, current_target: pd.Series) -> Optional[dict[str, Any]]:
        """Detect drift in target variable.

        Args:
            current_target: current batch of target values

        Returns:
            Dictionary with target drift result or None if no baseline target
        """
        if self.baseline_target is None:
            return None

        try:
            result = self.detector.detect_target_drift(self.baseline_target, current_target)
            self.exporter.record_drift_result(result)

            if result.drift_detected:
                alert = MonitoringAlert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Target drift detected (p-value: {result.p_value:.4f})",
                    alert_type="target_drift",
                    metadata={
                        "p_value": result.p_value,
                        "statistic": result.statistic,
                    },
                )
                self.alerts.append(alert)
                self.exporter.record_alert(AlertSeverity.CRITICAL)

            return {"target": result}
        except (InsufficientDataError, DriftDetectionError) as e:
            logger.warning(f"Could not detect target drift: {e}")
            return None

    def check_pipeline_health(self, metrics: PipelineMetrics) -> bool:
        """Check overall pipeline health.

        Args:
            metrics: PipelineMetrics to evaluate

        Returns:
            True if pipeline is healthy, False otherwise
        """
        healthy = metrics.is_healthy()
        self.exporter.record_pipeline_metrics(metrics)

        if not healthy:
            severity = (
                AlertSeverity.CRITICAL
                if metrics.error_rate >= 5.0
                else AlertSeverity.WARNING
            )
            alert = MonitoringAlert(
                severity=severity,
                message=f"Pipeline performance degradation: "
                f"throughput={metrics.throughput:.1f} rows/sec, "
                f"p99_latency={metrics.latency_p99:.0f}ms, "
                f"error_rate={metrics.error_rate:.2f}%",
                alert_type="pipeline_health",
                metadata={
                    "throughput": metrics.throughput,
                    "latency_p99": metrics.latency_p99,
                    "error_rate": metrics.error_rate,
                },
            )
            self.alerts.append(alert)
            self.exporter.record_alert(severity)

        return healthy

    def get_alerts(self, clear: bool = False) -> list[MonitoringAlert]:
        """Get accumulated alerts.

        Args:
            clear: if True, clear alerts after retrieving

        Returns:
            List of MonitoringAlert objects
        """
        alerts = self.alerts.copy()
        if clear:
            self.alerts.clear()
        return alerts

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus text format metrics
        """
        return self.exporter.get_metrics()

    async def run_monitoring_loop(
        self,
        data_source: Any,
        duration_seconds: Optional[int] = None,
    ) -> None:
        """Run continuous monitoring loop.

        Args:
            data_source: callable that returns (features_df, target_series)
            duration_seconds: how long to run (None = forever)
        """
        start_time = datetime.utcnow()

        try:
            while True:
                if duration_seconds and (
                    datetime.utcnow() - start_time
                ).total_seconds() > duration_seconds:
                    break

                try:
                    batch_df, target = data_source()

                    # Check data quality
                    self.check_data_quality(batch_df)

                    # Detect feature drift
                    self.detect_feature_drift(batch_df)

                    # Detect target drift
                    self.detect_target_drift(target)

                    # Log any alerts
                    for alert in self.get_alerts(clear=True):
                        logger.warning(f"[{alert.alert_type}] {alert.message}")

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                # Wait before next check
                await asyncio.sleep(self.config.check_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Monitoring loop stopped")
            raise
