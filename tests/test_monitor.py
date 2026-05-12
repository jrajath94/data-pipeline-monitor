"""Tests for PipelineMonitor."""

import pandas as pd
import pytest

from data_pipeline_monitor import (
    InsufficientDataError,
    MonitoringConfig,
    MonitoringConfigError,
    PipelineMetrics,
    PipelineMonitor,
)


class TestPipelineMonitor:
    """Test PipelineMonitor class."""

    @pytest.fixture
    def config(self) -> MonitoringConfig:
        """Create a monitoring config."""
        return MonitoringConfig(
            pipeline_name="test_pipeline",
            drift_threshold=0.05,
            min_baseline_samples=50,
            features_to_monitor=["feature1", "feature2"],
        )

    def test_init_valid_config(self, config: MonitoringConfig) -> None:
        """Test initialization with valid config."""
        monitor = PipelineMonitor(config)
        assert monitor.config.pipeline_name == "test_pipeline"
        assert monitor.baseline_data is None

    def test_init_invalid_config(self) -> None:
        """Test initialization with invalid config."""
        invalid_config = MonitoringConfig(
            pipeline_name="",  # Empty name
            drift_threshold=0.05,
        )
        with pytest.raises(MonitoringConfigError, match="Invalid monitoring configuration"):
            PipelineMonitor(invalid_config)

    def test_set_baseline_valid(
        self,
        config: MonitoringConfig,
        sample_baseline: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test setting valid baseline."""
        monitor = PipelineMonitor(config)
        monitor.set_baseline(sample_baseline, sample_target)

        assert monitor.baseline_data is not None
        assert monitor.baseline_target is not None
        assert len(monitor.baseline_data) == len(sample_baseline)

    def test_set_baseline_too_small(self, config: MonitoringConfig) -> None:
        """Test error when baseline is too small."""
        small_df = pd.DataFrame({"feature1": [1, 2, 3]})
        monitor = PipelineMonitor(config)

        with pytest.raises((MonitoringConfigError, InsufficientDataError)):
            monitor.set_baseline(small_df)

    def test_check_data_quality_healthy(
        self,
        config: MonitoringConfig,
        sample_baseline: pd.DataFrame,
    ) -> None:
        """Test data quality check for healthy data."""
        monitor = PipelineMonitor(config)
        metrics = monitor.check_data_quality(sample_baseline)

        assert metrics.missing_values_pct < 5.0
        assert metrics.duplicate_rows_pct < 1.0
        assert metrics.is_healthy()

    def test_check_data_quality_with_nulls(self, config: MonitoringConfig) -> None:
        """Test data quality check with missing values."""
        df = pd.DataFrame({
            "feature1": [1, 2, None, 4, 5] * 40,
            "feature2": [10, 20, 30, None, 50] * 40,
        })
        monitor = PipelineMonitor(config)
        metrics = monitor.check_data_quality(df)

        assert metrics.missing_values_pct > 0
        assert len(metrics.null_features) > 0

    def test_detect_feature_drift(
        self,
        config: MonitoringConfig,
        sample_baseline: pd.DataFrame,
        sample_current_with_drift: pd.DataFrame,
    ) -> None:
        """Test feature drift detection."""
        monitor = PipelineMonitor(config)
        monitor.set_baseline(sample_baseline)

        results = monitor.detect_feature_drift(sample_current_with_drift)

        assert "feature1" in results
        # feature1 should have drift due to shifted mean
        assert results["feature1"].drift_detected

    def test_detect_target_drift(
        self,
        config: MonitoringConfig,
        sample_baseline: pd.DataFrame,
        sample_target: pd.Series,
        sample_target_with_drift: pd.Series,
    ) -> None:
        """Test target drift detection."""
        monitor = PipelineMonitor(config)
        monitor.set_baseline(sample_baseline, sample_target)

        result = monitor.detect_target_drift(sample_target_with_drift)

        assert result is not None
        assert "target" in result
        assert result["target"].drift_detected

    def test_check_pipeline_health_healthy(self, config: MonitoringConfig) -> None:
        """Test pipeline health check for healthy pipeline."""
        monitor = PipelineMonitor(config)
        metrics = PipelineMetrics(
            throughput=1000.0,
            latency_p50=100.0,
            latency_p99=500.0,
            error_rate=0.1,
        )

        assert monitor.check_pipeline_health(metrics)

    def test_check_pipeline_health_unhealthy(self, config: MonitoringConfig) -> None:
        """Test pipeline health check for unhealthy pipeline."""
        monitor = PipelineMonitor(config)
        metrics = PipelineMetrics(
            throughput=10.0,  # Too low
            latency_p50=100.0,
            latency_p99=10000.0,  # Too high
            error_rate=5.0,  # Too high
        )

        assert not monitor.check_pipeline_health(metrics)

    def test_alerts_accumulation(
        self,
        config: MonitoringConfig,
        sample_baseline: pd.DataFrame,
    ) -> None:
        """Test that alerts accumulate."""
        monitor = PipelineMonitor(config)
        monitor.set_baseline(sample_baseline)

        initial_alerts = len(monitor.get_alerts())

        metrics = PipelineMetrics(
            throughput=10.0,
            latency_p50=100.0,
            latency_p99=10000.0,
            error_rate=5.0,
        )
        monitor.check_pipeline_health(metrics)

        new_alerts = monitor.get_alerts()
        assert len(new_alerts) > initial_alerts

    def test_alerts_clear(self, config: MonitoringConfig) -> None:
        """Test clearing alerts."""
        monitor = PipelineMonitor(config)

        metrics = PipelineMetrics(
            throughput=10.0,
            latency_p50=100.0,
            latency_p99=10000.0,
            error_rate=5.0,
        )
        monitor.check_pipeline_health(metrics)

        alerts = monitor.get_alerts(clear=True)
        assert len(alerts) > 0

        alerts_after = monitor.get_alerts()
        assert len(alerts_after) == 0

    def test_export_metrics(self, config: MonitoringConfig) -> None:
        """Test exporting metrics in Prometheus format."""
        monitor = PipelineMonitor(config)
        metrics_bytes = monitor.export_metrics()

        assert isinstance(metrics_bytes, bytes)
        assert b"pipeline" in metrics_bytes
