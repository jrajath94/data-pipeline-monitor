"""Quickstart example for pipeline monitoring."""

import logging

import numpy as np
import pandas as pd

from data_pipeline_monitor import (
    AlertSeverity,
    MonitoringConfig,
    PipelineMetrics,
    PipelineMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create synthetic baseline data
np.random.seed(42)
baseline_features = pd.DataFrame({
    "customer_value": np.random.normal(100, 15, 500),
    "transaction_count": np.random.normal(50, 10, 500),
    "region": np.random.choice(["US", "EU", "APAC"], 500),
    "product_category": np.random.choice(["A", "B", "C"], 500),
})
baseline_target = pd.Series(np.random.binomial(1, 0.6, 500), name="conversion")

# Initialize monitoring
config = MonitoringConfig(
    pipeline_name="customer_conversion",
    drift_threshold=0.05,
    min_baseline_samples=100,
    features_to_monitor=["customer_value", "transaction_count", "region"],
    categorical_features=["region", "product_category"],
    check_interval_seconds=1,
)

monitor = PipelineMonitor(config)
monitor.set_baseline(baseline_features, baseline_target)

print("✓ Baseline set with 500 samples")

# Simulate current batch with no drift
np.random.seed(123)
current_features_healthy = pd.DataFrame({
    "customer_value": np.random.normal(100, 15, 200),
    "transaction_count": np.random.normal(50, 10, 200),
    "region": np.random.choice(["US", "EU", "APAC"], 200),
    "product_category": np.random.choice(["A", "B", "C"], 200),
})

print("\n--- Checking healthy data ---")
quality = monitor.check_data_quality(current_features_healthy)
print(f"Data quality: missing={quality.missing_values_pct:.2f}%, "
      f"duplicates={quality.duplicate_rows_pct:.2f}%")

drift_results = monitor.detect_feature_drift(current_features_healthy)
drifted_features = [f for f, r in drift_results.items() if r.drift_detected]
print(f"Features with drift: {drifted_features or 'None'}")

# Check pipeline health
health_metrics = PipelineMetrics(
    throughput=1500.0,
    latency_p50=50.0,
    latency_p99=200.0,
    error_rate=0.05,
)
is_healthy = monitor.check_pipeline_health(health_metrics)
print(f"Pipeline health: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

# Simulate current batch with drift
print("\n--- Checking data with drift ---")
np.random.seed(456)
current_features_drift = pd.DataFrame({
    "customer_value": np.random.normal(130, 25, 200),  # Shifted mean and variance
    "transaction_count": np.random.normal(50, 10, 200),
    "region": np.random.choice(["US", "EU", "APAC", "OTHER"], 200),  # New category
    "product_category": np.random.choice(["A", "B", "C"], 200),
})

quality = monitor.check_data_quality(current_features_drift)
drift_results = monitor.detect_feature_drift(current_features_drift)
drifted_features = [f for f, r in drift_results.items() if r.drift_detected]
print(f"Features with drift: {drifted_features}")

# Show alerts
alerts = monitor.get_alerts(clear=True)
print(f"\nAlerts ({len(alerts)}):")
for alert in alerts:
    icon = "⚠" if alert.severity == AlertSeverity.WARNING else "🔴"
    print(f"  {icon} {alert.alert_type}: {alert.message}")

# Export metrics
metrics_text = monitor.export_metrics().decode("utf-8")
print(f"\nPrometheus metrics ({len(metrics_text)} chars exported)")

print("\n✓ Example complete!")
