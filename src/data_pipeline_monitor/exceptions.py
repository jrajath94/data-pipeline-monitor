"""Custom exceptions for data pipeline monitoring."""


class PipelineMonitorError(Exception):
    """Base exception for pipeline monitoring errors."""

    pass


class DriftDetectionError(PipelineMonitorError):
    """Raised when drift detection fails."""

    pass


class DataQualityError(PipelineMonitorError):
    """Raised when data quality checks fail."""

    pass


class MetricsExportError(PipelineMonitorError):
    """Raised when exporting metrics fails."""

    pass


class MonitoringConfigError(PipelineMonitorError):
    """Raised when monitoring configuration is invalid."""

    pass


class InsufficientDataError(PipelineMonitorError):
    """Raised when insufficient data for statistical tests."""

    pass
