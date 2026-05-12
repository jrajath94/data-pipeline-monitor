"""Custom exceptions for data pipeline monitoring."""


class PipelineMonitorError(Exception):
    """Base exception for pipeline monitoring errors."""



class DriftDetectionError(PipelineMonitorError):
    """Raised when drift detection fails."""



class DataQualityError(PipelineMonitorError):
    """Raised when data quality checks fail."""



class MetricsExportError(PipelineMonitorError):
    """Raised when exporting metrics fails."""



class MonitoringConfigError(PipelineMonitorError):
    """Raised when monitoring configuration is invalid."""



class InsufficientDataError(PipelineMonitorError):
    """Raised when insufficient data for statistical tests."""

