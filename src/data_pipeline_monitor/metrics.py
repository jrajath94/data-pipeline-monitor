"""Prometheus metrics exporter for pipeline monitoring."""

import logging
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest

from .models import AlertSeverity, DriftResult, PipelineMetrics

logger = logging.getLogger(__name__)

# Drift detection metrics
drift_detections = Counter(
    "pipeline_drift_detections_total",
    "Total number of drift detections",
    ["pipeline", "drift_type"],
)

drift_p_value = Gauge(
    "pipeline_drift_p_value",
    "P-value from drift test",
    ["pipeline", "feature"],
)

features_with_drift = Gauge(
    "pipeline_features_with_drift",
    "Number of features with detected drift",
    ["pipeline"],
)

# Data quality metrics
missing_values_pct = Gauge(
    "pipeline_missing_values_percent",
    "Percentage of missing values",
    ["pipeline"],
)

duplicate_rows_pct = Gauge(
    "pipeline_duplicate_rows_percent",
    "Percentage of duplicate rows",
    ["pipeline"],
)

data_quality_score = Gauge(
    "pipeline_data_quality_score",
    "Overall data quality score (0-100)",
    ["pipeline"],
)

# Pipeline performance metrics
pipeline_throughput = Gauge(
    "pipeline_throughput_rows_per_sec",
    "Pipeline throughput in rows per second",
    ["pipeline"],
)

pipeline_latency_p50 = Histogram(
    "pipeline_latency_p50_ms",
    "Pipeline latency P50 in milliseconds",
    ["pipeline"],
    buckets=(10, 50, 100, 500, 1000, 5000),
)

pipeline_latency_p99 = Histogram(
    "pipeline_latency_p99_ms",
    "Pipeline latency P99 in milliseconds",
    ["pipeline"],
    buckets=(10, 50, 100, 500, 1000, 5000),
)

pipeline_error_rate = Gauge(
    "pipeline_error_rate_percent",
    "Pipeline error rate in percentage",
    ["pipeline"],
)

# Alert metrics
alerts_raised = Counter(
    "pipeline_alerts_raised_total",
    "Total number of alerts raised",
    ["pipeline", "severity"],
)


class MetricsExporter:
    """Exports pipeline monitoring metrics to Prometheus."""

    def __init__(self, pipeline_name: str) -> None:
        """Initialize metrics exporter.

        Args:
            pipeline_name: name of the pipeline being monitored
        """
        self.pipeline_name = pipeline_name

    def record_drift_result(self, result: DriftResult) -> None:
        """Record a drift detection result.

        Args:
            result: DriftResult from drift detector
        """
        drift_detections.labels(
            pipeline=self.pipeline_name,
            drift_type=result.drift_type.value,
        ).inc()

        if result.feature_name:
            drift_p_value.labels(
                pipeline=self.pipeline_name,
                feature=result.feature_name,
            ).set(result.p_value)

    def record_drift_summary(self, num_features_with_drift: int) -> None:
        """Record summary of features with drift.

        Args:
            num_features_with_drift: count of features with drift
        """
        features_with_drift.labels(pipeline=self.pipeline_name).set(num_features_with_drift)

    def record_data_quality(
        self,
        missing_pct: float,
        duplicate_pct: float,
    ) -> None:
        """Record data quality metrics.

        Args:
            missing_pct: percentage of missing values
            duplicate_pct: percentage of duplicate rows
        """
        missing_values_pct.labels(pipeline=self.pipeline_name).set(missing_pct)
        duplicate_rows_pct.labels(pipeline=self.pipeline_name).set(duplicate_pct)

        # Simple quality score
        score = 100 - (missing_pct + duplicate_pct)
        data_quality_score.labels(pipeline=self.pipeline_name).set(max(0, score))

    def record_pipeline_metrics(self, metrics: PipelineMetrics) -> None:
        """Record overall pipeline performance metrics.

        Args:
            metrics: PipelineMetrics object
        """
        pipeline_throughput.labels(pipeline=self.pipeline_name).set(metrics.throughput)
        pipeline_latency_p50.labels(pipeline=self.pipeline_name).observe(metrics.latency_p50)
        pipeline_latency_p99.labels(pipeline=self.pipeline_name).observe(metrics.latency_p99)
        pipeline_error_rate.labels(pipeline=self.pipeline_name).set(metrics.error_rate)

    def record_alert(self, severity: AlertSeverity) -> None:
        """Record an alert.

        Args:
            severity: AlertSeverity level
        """
        alerts_raised.labels(
            pipeline=self.pipeline_name,
            severity=severity.value,
        ).inc()

    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus text format.

        Returns:
            Prometheus metrics as bytes
        """
        return generate_latest()

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary for easy inspection.

        Returns:
            Dictionary with metric names and values
        """
        return {
            "pipeline": self.pipeline_name,
            "drift_detections": dict(drift_detections._metrics),
            "features_with_drift": features_with_drift.labels(pipeline=self.pipeline_name)._value.get(),
            "missing_values_pct": missing_values_pct.labels(
                pipeline=self.pipeline_name
            )._value.get(),
            "duplicate_rows_pct": duplicate_rows_pct.labels(
                pipeline=self.pipeline_name
            )._value.get(),
            "data_quality_score": data_quality_score.labels(
                pipeline=self.pipeline_name
            )._value.get(),
        }
